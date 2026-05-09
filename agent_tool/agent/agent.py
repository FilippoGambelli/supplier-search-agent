from dotenv import load_dotenv
load_dotenv()

import json
from typing import Annotated, TypedDict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from agent_tool.agent.search import search_web
from agent_tool.agent.scrape import scrape_company_website, is_valid_company_result, extract_paginegialle_websites
from agent_tool.agent.llm_extract import extract_data
from agent_tool.agent.utils import extract_and_parse_json

from agent_tool.logger import logger

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL = "gemma4:31b-cloud"

LLM = ChatOllama(
    base_url=OLLAMA_BASE_URL,
    model=MODEL,
    temperature=0
)

@tool
def search_suppliers(query: str) -> str:
    """
    Search for suppliers on the web using SearXNG. 
    Use this first to find potential companies.
    Returns a string containing a list of titles and URLs.
    """
    logger.info(f"[TOOL] search_suppliers - Query: {query}")
    try:
        results = search_web(query, limit=15)
        formatted = "\n".join([f"- {r.get('title', 'N/A')}: {r.get('url', 'N/A')}" for r in results])
        return f"Found {len(results)} results:\n{formatted}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def is_valid_company(title: str, url: str) -> str:
    """
    Check if a search result is a valid company website.
    Returns a string with instructions on how to proceed.
    """
    logger.info(f"[TOOL] is_valid_company - URL: {url}")
    try:
        url_lower = url.lower()
        
        if "paginegialle.it" in url_lower:
            return "FALSE - This is a PagineGialle directory. ACTION REQUIRED: You MUST use the `extract_from_paginegialle` tool on this URL immediately to extract the real company websites."
        
        is_valid = is_valid_company_result(title, url)
        
        if is_valid:
            return "TRUE - Valid company website. You can proceed to use `research_and_extract_company` on this URL."
        else:
            return "FALSE - This is a generic aggregator, directory, or social media. Ignore this URL and move to the next one."
            
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def extract_from_paginegialle(pg_url: str) -> str:
    """
    Extract real company websites from a PagineGialle directory page.
    Use this ONLY when you find a PagineGialle link in your search results.
    """
    logger.info(f"[TOOL] extract_from_paginegialle - URL: {pg_url}")
    try:
        results = extract_paginegialle_websites(pg_url, limit=10)
        if not results:
            return "No real websites found from PagineGialle page."
        formatted = "\n".join([f"- {r.get('name', 'N/A')}: {r.get('website', 'N/A')}" for r in results])
        return f"Found these real websites:\n{formatted}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def research_and_extract_company(url: str, title: str = "") -> str:
    """
    SUB-AGENT TOOL: Deeply analyze a specific company website.
    Use this tool on valid company URLs to extract structured business information.
    It automatically scrapes the website and extracts the data.
    """
    logger.info(f"[TOOL] research_and_extract_company - URL: {url}")
    try:
        data = scrape_company_website(url)
        if not data:
            return f"Failed to scrape {url}. The website might be blocking the connection or is offline."
        
        company_payload = {
            "url": url,
            "title": title,
            "homepage_text": data.get("homepage_text", ""),
            "contact_text": data.get("contact_text", "")
        }
        
        result = extract_data(company_payload)
        return json.dumps(result, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"[TOOL ERROR] research_and_extract_company: {e}")
        return f"Error during extraction for {url}: {str(e)}"
    
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
def init_node(state: InputState):
    """
    Prende la query testuale passata dall'utente (o da LangSmith)
    e la converte nel formato `messages` richiesto dall'LLM.
    """
    user_query = state["query"]
    return {"messages": [HumanMessage(content=user_query)]}

def agent_node(state: OverallState):
    """
    The Agent node. It represents the "brain" of the graph where the LLM thinks
    and decides whether to call a tool or provide the final answer.
    """
    
    # Extract the conversation history from the global state.
    # This gives the LLM context about the user's query and the tool results it has already seen.
    messages = state.get("messages", [])
    
    # Prepend the System Prompt to the beginning of the message history.
    # We do this dynamically in RAM right before calling the LLM, rather than saving it 
    # to the graph's state, to prevent the prompt from duplicating at every loop iteration.
    messages_with_system = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    
    response = llm_with_tools.invoke(messages_with_system)
    
    # Append the LLM's new thought/action to the existing list of messages in the state.
    state_update = {"messages": [response]}
    
    # We check if the LLM decided NOT to call any tools. If the 'tool_calls' list is empty 
    # or missing, it means the LLM has finished researching and wrote the final JSON array.
    if getattr(response, "tool_calls", None) is None or len(response.tool_calls) == 0:
        # We capture this final text and put it into the 'answer' key. 
        # This tells LangGraph to populate the OutputState and finish the process.
        state_update["answer"] = response.content
        
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

def run_agent(query: str):
    logger.info(f"[AGENT START] Query: {query}")
    
    try:
        result = app.invoke({"query": query})
        
        if "answer" not in result:
            return {
                "answer": None,
                "error": "No answer returned from agent"
            }
        
        raw_answer = result.get("answer", "")
        
        parsed_json, error_msg = extract_and_parse_json(raw_answer)
        
        if error_msg:
            logger.error(f"[AGENT PARSE ERROR] {error_msg}\nReceived string: {raw_answer}")
            return {
                "answer": None,
                "raw_answer": raw_answer,
                "error": error_msg
            }
            
        return {
            "answer": parsed_json,
            "error": None
        }
        
    except Exception as e:
        logger.error(f"[AGENT FATAL ERROR] {e}")
        return {
            "answer": None,
            "error": str(e)
        }