import sys
from pathlib import Path

# Add the root directory (claritas-agent/) to the Python path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

import json
from langchain.tools import Tool
from services.llm import invoke_model
import pymysql
import pandas as pd
from datetime import date, datetime
import decimal

# --- MySQL connection config ---
MYSQL_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "claritas_v2"
}

# ----------------------------
# MySQL Utilities
# ----------------------------
def convert_mysql_result(row):
    return {
        k: (
            v.isoformat() if isinstance(v, (date, datetime)) else
            float(v) if isinstance(v, decimal.Decimal) else
            v
        )
        for k, v in row.items()
    }

def run_mysql_query(query):
    print(f"🔧 Running SQL against MySQL...")
    conn = pymysql.connect(
        host=MYSQL_CONFIG["host"],
        user=MYSQL_CONFIG["user"],
        password=MYSQL_CONFIG["password"],
        database=MYSQL_CONFIG["database"],
        cursorclass=pymysql.cursors.DictCursor
    )
    with conn.cursor() as cursor:
        cursor.execute(query)
        results = [convert_mysql_result(row) for row in cursor.fetchall()]
    conn.close()
    print(f"✅ Retrieved {len(results)} rows\n")
    return results

# ----------------------------
# Format schema for LLM prompt
# ----------------------------
def format_schema_for_prompt():
    schema_path = BASE_DIR / "data" / "database_schema.json"
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_dict = json.load(f)

    lines = []
    for table, columns in schema_dict.items():
        lines.append(f"Table: {table}")
        for col in columns:
            lines.append(f"  - {col['column']} {col['type']}")
        lines.append("")
    return "\n".join(lines)

# ----------------------------
# Claude SQL Generation
# ----------------------------
def get_sql_from_prompt(prompt, schema):
    formatted_prompt = f"""
    \n\nHuman: You are a SQL expert. Given the following database schema and user request, generate the best SQL query to answer the question.

    SCHEMA:
    {schema}

    USER REQUEST:
    {prompt}


Respond with ONLY the raw SQL query and nothing else. Do NOT wrap it in markdown, triple backticks, or provide any commentary. Just output valid SQL syntax as it would be executed directly in MySQL.

IMPORTANT RULES:
- Use the `pixel_hist` table to get hitcount data broken down by date, action, dma, prectived_or_actual, and top_level_domain. prectived_or_actual represents whether the data is 'predicted' or 'actual'.
- NOTE: The column for actual vs predicted labels is named `prectived_or_actual` (note the typo) in the `pixel_hist` table. Use it as-is when querying.
- Use the `prizm_to_dma` table to match DMAs with PRIZM segments. Join on `pixel_hist.dma = prizm_to_dma.DMA_CODE`.
- Use the `prizm_info` table to describe PRIZM segments. Join on `prizm_to_dma.PRIZM_Segment = prizm_info.PRIZM_Segment`.
- When aggregating (e.g., SUM(hitcount)), use GROUP BY to group by the appropriate dimension (e.g., PRIZM_Segment, DMA_Name).
- Always ensure SELECT, FROM, JOIN, WHERE, and GROUP BY clauses are used correctly and formatted cleanly.
- Never include triple backticks (```), the word "sql", or any markdown formatting.
- In the `pixel_hist` table, values labeled as actual or predicted are stored in the `predicted_or_actual` column.
- To check whether a row is actual or predicted, always use a LIKE filter with wildcards to handle whitespace:
    - For actual values: `WHERE predicted_or_actual LIKE "%actual%"`
    - For predicted values: `WHERE predicted_or_actual LIKE "%predicted%"`
- The most recent day with actual readings is `2025-02-28`. Treat this as "today" when interpreting questions about "next week", "next month", or future forecasts.
FILTERING INSTRUCTIONS:
- If the user references an organization like "RedCrossBlood.org", filter using:
    `pixel_hist.top_level_domain LIKE '%RedCrossBlood.org%'`
- If the user references a geographic region like "New York", filter using:
    `prizm_to_dma.DMA_Name LIKE '%New York%'`

MAPPING NOTES:
- The `action` column in the `pixel_hist` table contains the following case-sensitive values:
    - 'lead'
    - 'signup'
    - 'registration'
    - 'install'
    - 'purchase'
    - 'homepage'

- If the user mentions:
    - "registration", "voter registration", or "campaign signups" → use `action = 'registration'`
    - "signup" → use `action = 'signup'`
    - "lead generation" or "leads" → use `action = 'lead'`
    - "installs" → use `action = 'install'`
    - "purchases" → use `action = 'purchase'`
    - "homepage traffic", "site visits", or "landing page" → use `action = 'homepage'`

- Always use `pixel_hist.action` to filter for interaction types. Ensure the value matches one of the valid case-sensitive options above.
- Prefer fully qualified column names (e.g., `pixel_hist.date`) rather than table aliases.

\n\nAssistant:
    """
    messages = [{"role": "user", "content": formatted_prompt}]
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2000,
        "messages": messages
    }
    modelId = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
    response = invoke_model(body, modelId, "application/json", "application/json")
    response_body = json.loads(response.get("body").read())
    return response_body["content"][0]["text"].strip()

# ----------------------------
# LangChain Tool Wrapper
# ----------------------------
def get_query(prompt):
    schema = format_schema_for_prompt()
    sql_query = get_sql_from_prompt(prompt, schema)
    try:
        result = run_mysql_query(sql_query)
        return pd.DataFrame(result)
    except Exception as e:
        print("❌ SQL execution error:", e)
        return None

def create_query_tool(schema=None):  # schema arg not used now
    return Tool(
        name="query_tool",
        func=lambda user_prompt: get_query(user_prompt),
        description="Use this tool to query the MySQL database using a natural language prompt."
                "Use this tool to query the MySQL database for any questions involving web hitcount activity across all domains and DMA codes. "
        "This includes actual historical data or predicted future trends, segmented by action types such as 'lead', 'signup', 'registration', 'purchase', etc. "
        "Use this tool for analyzing patterns related to PRIZM segments, domains, digital campaigns, and geographic locations. "
        "It should be used whenever the user prompt references specific websites, DMA regions, PRIZM behaviors, or phrases like "
        "'next week', 'next month', 'predictions', 'trends', or 'actual vs predicted'. "
        "This is the primary tool for answering all questions grounded in the structured database schema."
    )
