from typing import Annotated, TypedDict, List, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from config import *
from logger import logger

from agent_dbmanager.agent import init_database
from agent_orchestrator.sub_agents import run_search_agent, run_dbmanager_agent

# System prompt defining the orchestrator's behavior
# System prompt defining the orchestrator's behavior with strict workflow and deduplication rules
SYSTEM_PROMPT = """You are an Orchestrator Agent specialized in data retrieval, deduplication, and database synchronization.

Your available tools:
1. run_search_agent(query): Searches the web for external data or suppliers. Returns a JSON array.
2. run_dbmanager_agent(query): Searches the internal database OR inserts new data. Returns a JSON array or a status message.

CRITICAL WORKFLOW INSTRUCTIONS:
You must strictly follow these sequential steps for every user request:

STEP 1: FETCH DATA
- You MUST ALWAYS invoke BOTH `run_search_agent` and `run_dbmanager_agent` using the user's exact query.

STEP 2: IDENTIFY DUPLICATES AND MERGE
- Analyze the JSON results returned by both tools.
- Compare the web results against the database results to find duplicates.
- DEFINITION OF A DUPLICATE: Two records are considered the same entity if they share informations likes same phone number OR if their website hostnames match (e.g., 'example.com' matches 'www.example.com'). They do not need to be 100% identical JSON objects.
- Create a master list merging database suppliers and newly found web suppliers, ensuring NO duplicates exist in this final list.

STEP 3: INSERT NEW SUPPLIERS
- Identify "new" suppliers (those found by `run_search_agent` that are NOT present in the database results).
- If there are new suppliers, you MUST call `run_dbmanager_agent` again to save them.
- Format the query for this tool call exactly as: "insert [JSON array containing ONLY the new suppliers]".
- Wait for the tool to confirm the insertion.

STEP 4: FINAL RESPONSE
- Once the new suppliers have been inserted (or if there were no new suppliers), you must return the final merged and deduplicated list of all companies.
- CRITICAL: Your final answer MUST be a single, strict JSON string containing the array of all unique companies.
- DO NOT use conversational text, explanations, or markdown code blocks (such as ```json). Output only the raw JSON array.
"""

LLM = ChatOllama(
    base_url=OLLAMA_BASE_URL,
    model=AGENT_ORCHESTRATOR_MODEL,
    reasoning=True,
    temperature=0,
    timeout=300
)


# List of available tools for the orchestrator
tools = [
    run_search_agent,
    run_dbmanager_agent
]

# Bind tools to the LLM
llm_with_tools = LLM.bind_tools(tools)


# State definitions for LangGraph
class InputState(TypedDict):
    """Input state schema - requires a query string."""
    query: str


class OutputState(TypedDict):
    """Output state schema - contains the answer."""
    answer: str


class OverallState(InputState, OutputState):
    """Combined state with messages history for the agent."""
    messages: Annotated[list, add_messages]



def init_node(state: InputState) -> OverallState:
    """
    Initialize the agent state with the user's query.
    Converts the input query into a HumanMessage for the LLM.
    """
    user_query = state["query"]
    logger.info("[ORCHESTRATOR] Executing init node")
    logger.info(f"[ORCHESTRATOR] Received query: {user_query}")
    return {"messages": [HumanMessage(content=user_query)]}


def agent_node(state: OverallState) -> OverallState:
    """
    Main agent node that processes messages and decides on tool calls.

    Invokes the LLM with the system prompt and message history,
    then returns the response for tool execution or final answer.
    """
    logger.info("[ORCHESTRATOR] Executing agent node")

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

# Add nodes to the graph
graph.add_node("init", init_node)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))

# Define edges
graph.add_edge(START, "init")
graph.add_edge("init", "agent")
graph.add_conditional_edges("agent", tools_condition)
graph.add_edge("tools", "agent")

# Compile the graph
app = graph.compile()


def run_orchestrator(query: str) -> tuple:
    """
    Execute the orchestrator agent with a given query.

    Args:
        query: The user query to process

    Returns:
        Tuple of (answer, error) where error is None on success
    """
    logger.info(f"[ORCHESTRATOR] Starting orchestration with query: {query}")

    try:
        init_database()         # Initialize the database FIRST

        result = app.invoke({"query": query})
        answer = result.get("answer", "")
        logger.info("[ORCHESTRATOR] Orchestration completed")
        return answer, None
    except Exception as e:
        logger.error(f"[ORCHESTRATOR FATAL ERROR] {e}")
        return None, str(e)