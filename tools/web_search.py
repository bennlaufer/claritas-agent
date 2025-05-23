import os
from langchain.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults

def create_tavily_tool():
    return Tool(
        name="tavily_web_search",
        func=TavilySearchResults(max_results=1, TAVILY_API_KEY=os.getenv("TAVILY_API_KEY")),
        description="Use this tool for current web search information."
    )
