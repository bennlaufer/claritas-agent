from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_community.chat_models import BedrockChat
from langchain import hub

from tools.query_tool import create_query_tool
from tools.rag_tool import create_rag_tool
from tools.web_search import create_tavily_tool
from config.env import create_aws_client

aws_client = create_aws_client()

def setup_agent(schema):
    tools = [
        create_tavily_tool(),
        create_query_tool(schema),
        create_rag_tool()
    ]

    llm = BedrockChat(
        client=aws_client,
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        model_kwargs={"max_tokens": 2048, "temperature": 0.0}
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    prompt = hub.pull("hwchase17/structured-chat-agent")
    agent = create_structured_chat_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )
    return agent_executor, memory
