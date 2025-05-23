import json
from langchain.tools import Tool
from pathlib import Path
import pandas as pd
from services.llm import invoke_model
import sqlite3
import os

# --- read data ---
#db_path = "claritas.db"
base_path = Path(__file__).resolve().parent.parent
db_path = base_path / 'data' / 'claritas.db'
csv_dir = base_path / 'data'

conn = sqlite3.connect(db_path)

# Files: name as {table}.csv
for file in os.listdir(csv_dir):
    if file.endswith(".csv"):
        table_name = os.path.splitext(file)[0]
        df = pd.read_csv(os.path.join(csv_dir, file))
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f"Loaded table: {table_name}")

conn.close()



# Generate SQL code using prompt and schema
def get_sql_from_prompt(prompt, schema):

    #format the prompt that will be sent when calling the LLM
    #importance in structuring well to ensure we get favoravle results
    formatted_prompt = f"""
    \n\nHuman: You are a SQL expert. Given the following database schema and user request, generate the best SQL query to answer the question.

    SCHEMA:
    {schema}

    USER REQUEST:
    {prompt}

    Respond with ONLY the SQL query and nothing else. Do not include explanations or formatting.

    IMPORTANT RULES:
    When using hitcount data, the action types are the following (case sensitive):
        - lead
        - signup
        - registration
        - install

    NOTES:
        - In the pred_actual_data_combined table, values that are labeled as actual are past data points that are used to train the predicted data (future)

    \n\nAssistant:
    """
    #message list format to allow for multi-turn conversations (not sure if this is in the right place)
    messages = [{"role": "user", "content": formatted_prompt}]

    #prep request body for bedrock (select claude, max_tokens, and formatted chat version)
    body={
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 250,
            "messages": messages
        }

    #model type (via claude)
    modelId = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"

    #ensure content type
    accept = "application/json"
    contentType = "application/json"

    #send the formatted response to the LLM
    response = invoke_model(body, modelId, accept, contentType)

    #extract raw response body and parse from JSON
    response_body = json.loads(response.get("body").read())

    #pull actual SQL string generated
    sql_query = response_body["content"][0]["text"].strip()
    return sql_query

def get_query(prompt, schema):

    sql_query = get_sql_from_prompt(prompt, schema)

    #connect to local SQLlite database with path
    conn = sqlite3.connect(db_path)
    try:
        #send query to database and retrieve set, load to pandas df
        df = pd.read_sql_query(sql_query, conn)
        return df
    except Exception as e:
        print("SQL execution error:", e)
        return None
    finally:
        conn.close()

def create_query_tool(schema):
    return Tool(
        name="query_tool",
        func=lambda user_prompt: get_query(user_prompt, schema),
        description="Use this tool to query database using a natural language prompt."
    )
