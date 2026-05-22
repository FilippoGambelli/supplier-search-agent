from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from config import *
from logger import logger
from agent_dbmanager.tools import save_suppliers, semantic_search_suppliers

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
Store results in artifact store (handled by tool)

STEP 3:
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