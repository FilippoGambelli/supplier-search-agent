from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from agent_websearch.tools import search_suppliers, is_valid_company, extract_from_paginegialle, research_and_extract_company
from agent_websearch import tools as _tools_module
from logger import logger
from config import *
from stats import get_stats
from langsmith import traceable


SYSTEM_PROMPT = """You are an autonomous, expert supplier search assistant.
Your goal is to find potential suppliers based on the user's query and extract their contact information.

You MUST follow the workflow below STEP BY STEP, in the EXACT order described.
Do NOT skip steps. Do NOT change the order. Do NOT stop early.

STEP 1 — SEARCH (run exactly once)
Call `search_suppliers` using ONLY the original user query, unchanged.
Do NOT call `search_suppliers` more than once under any circumstances.
Collect every URL returned by the tool. These are your "candidate URLs".

STEP 2 — VALIDATION (run on every candidate URL)
For EACH candidate URL found in Step 1, call `is_valid_company` on that URL.
Process the response as follows:

  CASE A — The response contains:
    "FALSE - This is a PagineGialle directory. ACTION REQUIRED: You MUST use the `extract_from_paginegialle` tool"
    → Do NOT add this URL to the valid list.
    → Immediately call `extract_from_paginegialle` on this URL (see Step 3).

  CASE B — The response contains:
    "TRUE - Valid company website. You can proceed to use `research_and_extract_company` on this URL."
    → Add this URL to your VALID LIST.
    → Do NOT call any other tool on it yet.

  CASE C — The response contains:
    "FALSE - This is a generic aggregator, directory, or social media. Ignore this URL and move to the next one."
    → Discard this URL permanently. Do NOT process it further.

You MUST call `is_valid_company` on every single candidate URL before moving to Step 3.

STEP 3 — PAGINEGIALLE EXTRACTION
For every URL that triggered CASE A in Step 2, call `extract_from_paginegialle`.
This tool will return a list of company URLs extracted from the PagineGialle page.
Collect all these URLs into a separate PAGINEGIALLE LIST.
Do NOT validate or filter these URLs — they are already considered valid supplier leads.

STEP 4 — DEDUPLICATION
Merge your VALID LIST (from Step 2, Case B) and your PAGINEGIALLE LIST (from Step 3)
into a single FINAL LIST, removing duplicates.

Deduplication rule: compare only the hostname (base domain), ignoring paths, query params, and trailing slashes.
Examples:
  - "https://www.example.com/page1" and "https://example.com/contact" → DUPLICATE (same hostname) → keep only one
  - "https://supplier-a.com" and "https://supplier-b.com" → NOT duplicates → keep both

If two URLs share the same hostname, keep the first one encountered and discard the second.

STEP 5 — EXTRACTION (run on every URL in the FINAL LIST)
For EACH URL in the FINAL LIST, call `research_and_extract_company`.
You MUST process every single URL. Do NOT skip any.
Wait for all extractions to complete before proceeding to Step 6.

STEP 6 — FINAL RESPONSE
Only after ALL URLs in the FINAL LIST have been processed in Step 5, produce your final response as a single, 
structured JSON array containing all extracted company data.
DO NOT wrap the response in Markdown code blocks (e.g., do not use json ).
DO NOT add any conversational text before or after the JSON.

Do NOT produce any final response before Step 5 is fully complete.
"""


tools = [
    search_suppliers, 
    is_valid_company, 
    extract_from_paginegialle, 
    research_and_extract_company
]

llm_with_tools = AGENT_WEBSEARCH_LLM.bind_tools(tools)


class InputState(TypedDict):
    query: str
    verbose: bool

class OutputState(TypedDict):
    answer: str

class OverallState(InputState, OutputState):
    messages: Annotated[list, add_messages]


@traceable(name="init_node")
def init_node(state: InputState):
    """
    Takes the user input query (or the query received from LangSmith)
    and converts it into the `messages` format required by the LLM.
    """
    user_query = state["query"]
    return {"messages": [HumanMessage(content=user_query)]}

@traceable(name="agent_node")
def agent_node(state: OverallState):
    """
    The Agent node. It represents the "brain" of the graph where the LLM thinks
    and decides whether to call a tool or provide the final answer.
    """
    verbose = state.get("verbose", False)
    logger.info("[AGENT NODE] Executing agent node")
    stats = get_stats()

    messages = state.get("messages", [])
    messages_with_system = [SystemMessage(content=SYSTEM_PROMPT)] + messages

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
                    print("\n[WEBSEARCHER] Reasoning:")
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

def run_agent(query: str, verbose=True):
    _tools_module.VERBOSE = verbose
    initial_state = {"query": query, "verbose": verbose}
    if verbose:
        try:
            last_event = None
            for event in app.stream(initial_state):
                last_event = event

            if last_event is None:
                logger.error("[AGENT FATAL ERROR] Graph produced no events")
                return None, "Graph produced no events"

            agent_messages = last_event.get("agent", {}).get("messages", [])
            if not agent_messages:
                error_msg = last_event.get("error")
                if error_msg:
                    return None, error_msg
                logger.error("[AGENT FATAL ERROR] No messages in last agent event")
                return None, "No final answer produced"

            first_message = agent_messages[0]
            return getattr(first_message, "content", None), None

        except Exception as e:
            logger.error(f"[AGENT FATAL ERROR] {e}")
            return None, str(e)
    else:
        try:
            result = app.invoke(initial_state)
            error = result.get("error")
            if error:
                logger.error(f"[AGENT ERROR] Graph returned error: {error}")
                return None, error
            return result, None
        except Exception as e:
            logger.error(f"[AGENT FATAL ERROR] Error: {e}")
            return None, str(e)