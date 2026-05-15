import json
import re
import time
from typing import Annotated, TypedDict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from agent_tool.agent.tools.tools import search_suppliers, is_valid_company, extract_from_paginegialle, research_and_extract_company

from agent_tool.logger import logger
from agent_tool.config import *
from stats import get_stats, reset_stats

from langsmith import traceable

LLM = ChatOllama(
    base_url=OLLAMA_BASE_URL,
    model=MODEL,
    format="json",
    reasoning=True,
    temperature=0
)

tools = [
    search_suppliers, 
    is_valid_company, 
    extract_from_paginegialle, 
    research_and_extract_company
]

llm_with_tools = LLM.bind_tools(tools)


SYSTEM_PROMPT = """You are an autonomous, expert supplier search assistant. 
Your goal is to find potential suppliers based on the user's query and extract their contact information.

CRITICAL WORKFLOW RULES - YOU MUST FOLLOW THESE:
1. Search: Start by using `search_suppliers`.
2. PagineGialle Handling: If ANY URL contains "paginegialle.it", you MUST use the `extract_from_paginegialle` tool on that URL immediately.
3. Validation: For standard URLs, use `is_valid_company`.
4. Deduplication: Keep track of websites you have already processed in the chat history. Skip duplicates.
5. Extraction: For every valid company URL, use `research_and_extract_company`.

NO PREMATURE STOPPING:
You MUST process every single URL you find in the search results. 
Do not stop until you have evaluated every URL and extracted data from all the valid ones.
Once you have finished processing ALL URLs, and ONLY then, formulate your final response presenting all extracted companies as a single structured JSON array.
"""

# STATE
class InputState(TypedDict):
    query: str

class OutputState(TypedDict):
    answer: str

class OverallState(InputState, OutputState):
    messages: Annotated[list, add_messages]

# NODE
@traceable(name="init_node")
def init_node(state: InputState):
    """
    Prende la query testuale passata dall'utente (o da LangSmith)
    e la converte nel formato `messages` richiesto dall'LLM.
    """
    user_query = state["query"]
    return {"messages": [HumanMessage(content=user_query)]}

@traceable(name="agent_node")
def agent_node(state: OverallState):
    """
    The Agent node. It represents the "brain" of the graph where the LLM thinks
    and decides whether to call a tool or provide the final answer.
    """
    stats = get_stats()

    logger.info("="*80)
    logger.info("[AGENT] Executing agent node")

    messages = state.get("messages", [])
    messages_with_system = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    response = llm_with_tools.invoke(messages_with_system)

    # Update stats
    usage = response.usage_metadata or {}
    prompt_tokens = usage.get("input_tokens", 0)
    generated_tokens = usage.get("output_tokens", 0)
    stats.add_request(prompt_tokens, generated_tokens)


    state_update = {"messages": [response]}

    if getattr(response, "tool_calls", None) is None or len(response.tool_calls) == 0:
        state_update["answer"] = response.content
    else:
        for _ in response.tool_calls:
            stats.add_tool_call()

    return state_update

# BUILD GRAPH
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

@traceable(name="run_agent")
def run_agent(query: str):
    logger.info(f"[AGENT-TOOL] Starting agent with query: {query}")
    reset_stats()
    stats = get_stats()
    stats.start()

    try:
        result = app.invoke({"query": query})
        stats.stop()

        raw_answer = result.get("answer", "")
        return raw_answer, None

    except Exception as e:
        stats.stop()
        logger.error(f"[AGENT FATAL ERROR] {e}")
        return None, str(e)