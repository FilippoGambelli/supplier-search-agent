from colorama import Fore, Style, init
from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from config import *
from logger import logger
from stats import get_stats
from agent_dbmanager.tools import save_suppliers, semantic_search_suppliers
from agent_dbmanager import tools as _tools_module

init()

SYSTEM_PROMPT = """
You are a Database Manager Agent responsible for storing and retrieving supplier data.

You do NOT reason over data manually.
You do NOT use memory or previous messages.
You do NOT assume or reconstruct any dataset.

The ONLY source of truth is tool outputs.

All data must be transferred ONLY via artifact_id.

You MUST NEVER:
- pass raw JSON between steps
- use internal memory
- infer missing supplier data
- reuse previous tool outputs without reloading them

INPUT FORMAT
You will receive natural language instructions such as:
- "Find suppliers of building materials in Milan"
- "Retrieve suppliers in Lombardy region"
- "Search suppliers in Italy, Milan province"
- "Store suppliers from artifact_id"

You may also receive an artifact_id.

EXECUTION RULES - You MUST always follow tool-driven execution.

------------------------------------------------------------
INSERT FLOW (SAVE NEW DATA)
------------------------------------------------------------
If the request involves storing supplier data:

STEP 1:
Call save_suppliers(ARTIFACT_ID)

STEP 2:
Return a single string summary such as:
"Saved X suppliers successfully" or "Partial success: X saved, Y failed"

------------------------------------------------------------
RETRIEVE FLOW (SEARCH DATA)
------------------------------------------------------------
If the request involves retrieving supplier data:

STEP 1:
Call semantic_search_suppliers with ANY available geographic filters.

The tool supports:
- country
- region
- province
- city

RULE:
At least ONE filter MUST be provided.
If multiple are present, use all of them.

Example mappings:
- "suppliers in Italy" → country="Italy"
- "suppliers in Lombardy" → region="Lombardy"
- "suppliers in Milan" → city="Milan"
- "suppliers in Milan province" → province="Milan"
- "suppliers in Milan, Lombardy" → city="Milan", region="Lombardy"

STEP 2:
Return BOTH:
- a short status string explaining the operation
- the artifact_id where results are stored

Example output:
"Retrieved supplier dataset. artifact_id: ARTIFACT_ID"

------------------------------------------------------------
STRICT RULES
------------------------------------------------------------
- NEVER pass raw JSON between tools
- NEVER modify artifact content
- NEVER assume database state
- NEVER generate supplier data yourself
- NEVER skip artifact storage
- NEVER return raw tool outputs without formatting
- NEVER call search tool without at least one geographic filter

------------------------------------------------------------
FINAL OUTPUT FORMAT
------------------------------------------------------------
You MUST always return a STRING.

Allowed outputs:
- success message
- error message
- retrieval confirmation including artifact_id

You must NEVER:
- return raw JSON
- return structured objects
- expose internal tool reasoning steps
"""

# List of available tools
tools = [
   save_suppliers,
   semantic_search_suppliers
]

# Bind tools to the LLM
llm_with_tools = AGENT_DBMANAGER_LLM.bind_tools(tools)

# State definitions for LangGraph
class InputState(TypedDict):
    query: str
    verbose: bool

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
    stats = get_stats()

    messages = state.get("messages", [])
    messages_with_system = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    verbose = state.get("verbose", False)

    if verbose:
        full = None
        has_reasoning = False
        for chunk in llm_with_tools.stream(messages_with_system):
            if full is None:
                full = chunk
            else:
                full = full + chunk

            reasoning = chunk.additional_kwargs.get("reasoning_content", "")
            if reasoning:
                if not has_reasoning:
                    print(f"\n{Fore.GREEN}[DB MANAGER]{Style.RESET_ALL} Reasoning:")
                    has_reasoning = True
                print(reasoning, end="", flush=True)

        if has_reasoning:
            print()

        response = full
    else:
        response = llm_with_tools.invoke(messages_with_system)

    # Update stats
    usage = response.usage_metadata or {}
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    stats.add_request(input_tokens, output_tokens)

    state_update = {"messages": [response]}

    if not getattr(response, "tool_calls", None):
        state_update["answer"] = response.content
    else:
        for _ in response.tool_calls:
            stats.add_tool_call()

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

def run_dbmanager(query: str, verbose=True):
    _tools_module.VERBOSE = verbose
    initial_state = {"query": query, "verbose": verbose}
    try:
        result = app.invoke(initial_state)
        error = result.get("error")
        if error:
            logger.error(f"[DBMANAGER ERROR] Graph returned error: {error}")
            return None, error
        messages = result.get("messages", [])
        if messages:
            return getattr(messages[-1], "content", None), None
        return result.get("answer"), None
    except Exception as e:
        logger.error(f"[DBMANAGER FATAL ERROR] Error: {e}")
        return None, str(e)