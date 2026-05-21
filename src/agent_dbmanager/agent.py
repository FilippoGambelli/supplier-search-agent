from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from config import *
from logger import logger
import json

# Import the final tools
from agent_dbmanager.tools import save_suppliers, semantic_search_suppliers

SYSTEM_PROMPT = """
You are a Database Manager Agent specialized in managing supplier data in PostgreSQL.

You have two main tools at your disposal:
1. save_suppliers: Use this tool IF the user asks you to save or insert data and provides a JSON array. Extract the JSON from the user's request and pass it as a string to this tool.
2. semantic_search_suppliers: Use this tool IF the user asks to search for suppliers (e.g., "search for tile companies in Pisa"). You must extract the location (e.g., "Pisa") from the text request and pass them to this tool. The tool will handle creating and executing the SQL query.

CRITICAL INSTRUCTIONS: 
- When the user makes a request, invoke the appropriate tool ONLY ONCE.
- After the tool returns a response, DO NOT call the tool again under any circumstances.
- For your final answer, you MUST return exactly the JSON string output you received from the tool. Do not modify, introduce, or summarize it. Just output the raw JSON.
"""

LLM = ChatOllama(
    base_url=OLLAMA_BASE_URL,
    model=AGENT_DBMANAGER_MODEL,
    temperature=0,
    timeout=300
)

# List of available tools
tools = [
    save_suppliers,
    semantic_search_suppliers
]

# Bind tools to the LLM
llm_with_tools = LLM.bind_tools(tools)

# State definitions for LangGraph
class InputState(TypedDict):
    query: str

class OutputState(TypedDict):
    answer: str

class OverallState(InputState, OutputState):
    messages: Annotated[list, add_messages]

def init_node(state: InputState):
    user_query = state["query"]
    logger.info("[AGENT-DBMANAGER] Executing init node")
    logger.info(f"[AGENT-DBMANAGER] Received query: {user_query}")
    return {"messages": [HumanMessage(content=user_query)]}

def agent_node(state: OverallState):
    logger.info("[AGENT-DBMANAGER] Executing database manager node")
    messages = state.get("messages", [])
    messages_with_system = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    response = llm_with_tools.invoke(messages_with_system)
    state_update = {"messages": [response]}

    if getattr(response, "tool_calls", None) is None or len(response.tool_calls) == 0:
        state_update["answer"] = response.content

    return state_update

# Build the LangGraph state machine
graph = StateGraph(
    OverallState,
    input_schema=InputState,
    output_schema=OutputState
)

graph.add_node("init", init_node)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "init")
graph.add_edge("init", "agent")
graph.add_conditional_edges("agent", tools_condition)
graph.add_edge("tools", "agent")

app = graph.compile()

def init_database():
    from agent_dbmanager.db.db_manager import init_db
    init_db()
    logger.info("[AGENT-DBMANAGER] Database initialized")

def run_dbmanager(query: str):
    try:
        result = app.invoke({"query": query})
        return result.get("answer", ""), None
    except Exception as e:
        logger.error(f"[DBMANAGER FATAL ERROR] {e}")
        return None, str(e)