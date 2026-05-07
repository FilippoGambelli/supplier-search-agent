from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict, List, Dict, Optional

from langgraph.graph import StateGraph, END

from agent.search import search_web
from agent.scrape import scrape_company_website, is_valid_company_result
from agent.llm_extract import extract_data

from src.logger import logger


# STATES

class InputState(TypedDict):
    query: str


class OutputState(TypedDict):
    final_answer: List[Dict]
    error: Optional[str]


class InternalState(InputState, OutputState):
    search_results: List[Dict]
    scraped_data: List[Dict]
    current_index: int
    extracted_results: List[Dict]


# NODES

def search_node(state: InputState) -> Dict:
    """Search the web using SearXNG."""
    query = state["query"]
    logger.info(f"[SEARCH NODE] Query: {query}")
    try:
        results = search_web(query)
        logger.info(f"[SEARCH NODE] Found {len(results)} results")
        return {"search_results": results}

    except Exception as e:
        logger.error(f"[SEARCH NODE] {e}")
        return {
            "error": "Unexpected error occurred. Please check the logs for more details.",
            "search_results": []
        }


def scrape_node(state: InternalState) -> Dict:
    """Scrape websites from search results."""
    search_results = state.get("search_results", [])
    logger.info(f"[SCRAPE NODE] Processing {len(search_results)} URLs")
    
    scraped_data = []

    for result in search_results:
        title = result.get("title", "")
        url = result.get("url", "")

        # Skip blacklisted sites
        if not is_valid_company_result(title, url):
            logger.warning(f"[SCRAPE NODE] Skipping blacklisted: {url}")
            continue

        try:
            scraped = scrape_company_website(url)
            scraped["title"] = title
            scraped["url"] = url
            scraped_data.append(scraped)

        except Exception as e:
            logger.error(f"[SCRAPE NODE] {url} -> {e}")
            # scraped_data.append({
            #     "title": title,
            #     "url": url,
            #     "homepage_text": None,
            #     "contact_text": None,
            #     "error": str(e)
            # })
    
    if not scraped_data:
        logger.error("[SCRAPE NODE] No valid websites scraped")

        return {
            "error": "No valid company data could be scraped from search results.",
            "scraped_data": []
        }

    logger.info(f"[SCRAPE NODE] Successfully scraped {len(scraped_data)} websites")

    return {
        "scraped_data": scraped_data,
        "current_index": 0,
        "extracted_results": []
    }


def extract_data_llm_node(state: InternalState) -> Dict:
    """Process one company at a time."""

    index = state["current_index"]
    company = state["scraped_data"][index]

    extracted_results = state.get("extracted_results", [])

    try:
        result = extract_data(company)
        logger.info(f"[LLM EXTRACT NODE] Success: {company.get('url')}")
        new_results = extracted_results + [result]
    except Exception as e:
        logger.error(f"[LLM EXTRACT NODE] {company.get('url')} -> {e}")
        new_results = extracted_results
    
    return {
        "extracted_results": new_results,
        "current_index": index + 1
    }


def final_answer_node(state: InternalState) -> Dict:
    """Return final aggregated results."""

    results = state.get("extracted_results", [])

    return {
        "final_answer": results,
        "error": None
    }


def error_node(state: InternalState):

    return {
        "final_answer": [],
        "error": state.get("error", "Unknown error")
    }


# ROUTER FUNCTIONS

def should_continue(state: InternalState):
    if state.get("error"):
        return "error"
    return "continue"

def loop_router(state: InternalState):
    """Decide if we continue looping."""

    if state["current_index"] < len(state["scraped_data"]):
        return "continue"

    return "end"


# BUILD GRAPH
graph = StateGraph(
    InternalState,
    input_schema=InputState,
    output_schema=OutputState
)

graph.add_node("search", search_node)
graph.add_node("scrape", scrape_node)
graph.add_node("extract", extract_data_llm_node)
graph.add_node("final_answer", final_answer_node)
graph.add_node("error", error_node)

graph.add_conditional_edges(
    "search",
    should_continue,
    {
        "continue": "scrape",
        "error": "error"
    }
)

graph.add_conditional_edges(
    "scrape",
    should_continue,
    {
        "continue": "extract",
        "error": "error"
    }
)

graph.add_conditional_edges(
    "extract",
    loop_router,
    {
        "continue": "extract",
        "end": "final_answer"
    }
)

graph.add_edge("final_answer", END)
graph.add_edge("error", END)

graph.set_entry_point("search")

app = graph.compile()

logger.info("[GRAPH] compiled successfully")


# RUNNER

def run_agent(query: str):

    initial_state = {
        "query": query
    }

    return app.invoke(initial_state)