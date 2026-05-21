from typing import TypedDict, List, Dict, Optional
from urllib.parse import urlparse

from langgraph.graph import StateGraph, END

from agent_pipeline.utils.search import search_web
from agent_pipeline.utils.scrape import scrape_company_website, is_valid_company_result, extract_paginegialle_websites
from agent_pipeline.utils.llm_extract import extract_data

from logger import logger
from stats import get_stats, reset_stats

from langsmith import traceable

# STATES
class InputState(TypedDict):
    query: str


class OutputState(TypedDict):
    final_answer: List[Dict]
    error: Optional[str]


class InternalState(InputState, OutputState):
    search_results: List[Dict]
    pg_results: List[Dict]
    scraped_data: List[Dict]
    current_index: int
    current_company: Dict
    should_finish: bool
    extracted_results: List[Dict]


# NODES
@traceable(name="search_node")
def search_node(state: InputState) -> Dict:
    """Search the web using SearXNG."""
    query = state["query"]

    logger.info("=" * 80)
    logger.info(f"[SEARCH] Starting web search with query: {query}")
    try:
        results = search_web(query)
        logger.info(f"[SEARCH] Found {len(results)} search results")
        logger.info("=" * 80)
        return {"search_results": results}

    except Exception as e:
        logger.error(f"[SEARCH NODE] {e}")
        return {
            "error": "Unexpected error occurred. Please check the logs for more details.",
            "search_results": []
        }

@traceable(name="extract_pg_node")
def extract_pg_node(state: InternalState) -> Dict:
    """
    Dedicated node to handle PagineGialle links.
    Extracts real company URLs and saves them separately in pg_results.
    """
    search_results = state.get("search_results", [])
    pg_results = []

    for result in search_results:
        url = result.get("url", "")
        title = result.get("title", "")

        if "paginegialle.it" in url.lower():
            logger.info("=" * 80)
            logger.info(f"[EXTRACT PG] PagineGialle directory detected, extracting real websites from: {url}")
            logger.info("=" * 80)

            try:
                discovered_urls = extract_paginegialle_websites(url)
                for item in discovered_urls:
                    pg_results.append({
                        "title": item["title"],
                        "url": item["url"]
                    })
                logger.info(f"[EXTRACT PG] Extracted {len(discovered_urls)} real company websites from PagineGialle")
                logger.info("=" * 80)
            except Exception as e:
                logger.error(f"[EXTRACT PG NODE] Error extracting URL {url}: {e}")

    return {"pg_results": pg_results}

@traceable(name="scrape_node")
def scrape_node(state: InternalState) -> Dict:
    """
    Scrape websites from search results.
    Combines search_results and pg_results, removes duplicates, applies blacklist, and runs the standard scrape flow.
    """
    search_results = state.get("search_results", [])
    pg_results = state.get("pg_results", [])
    scraped_data = []

    combined_results = search_results + pg_results
    logger.info("=" * 80)
    logger.info(f"[SCRAPE] Combining {len(search_results)} search results with {len(pg_results)} PagineGialle results")
    logger.info("=" * 80)    

    # Delete duplicates based on hostname
    unique_hostnames = set()
    deduplicated_results = []
    duplicates_count = 0

    for result in combined_results:
        url = result.get("url", "")

        try:
            parsed = urlparse(url)
            hostname = parsed.netloc.lower()
        except Exception:
            hostname = url.lower()

        if hostname in unique_hostnames:
            duplicates_count += 1
        else:
            unique_hostnames.add(hostname)
            deduplicated_results.append(result)

    if duplicates_count > 0:
        logger.info(f"[SCRAPE] Removed {duplicates_count} duplicate URLs")

    # Standard blacklist check for other results
    for result in deduplicated_results:
        title = result.get("title", "")
        url = result.get("url", "")

        # Check the blacklist and make sure to skip paginegialle.it 
        if "paginegialle.it" in url.lower() or not is_valid_company_result(title, url):
            logger.warning(f"[SCRAPE NODE] Skipping blacklisted or directory: {url}")
            logger.info("=" * 80)
            continue

        try:
            data = scrape_company_website(url)
            if data is None:
                continue
            data.update({"title": title, "url": url})
            scraped_data.append(data)
        except Exception as e:
            logger.error(f"[SCRAPE NODE] {url} -> {e}")
    
    logger.info("=" * 80)

    if not scraped_data:
        return {"error": "No valid data scraped.", "scraped_data": []}

    return {
        "scraped_data": scraped_data,
        "current_index": 0,
        "extracted_results": []
    }

@traceable(name="extract_node")
def extract_node(state: InternalState) -> Dict:

    index = state["current_index"]
    companies = state["scraped_data"]

    if index >= len(companies):
        return {
            "should_finish": True
        }

    current_company = companies[index]

    return {
        "current_company": current_company,
        "should_finish": False
    }

@traceable(name="llm_extract_node")
def llm_node(state: InternalState) -> Dict:
    company = state["current_company"]

    extracted_results = state.get("extracted_results", [])

    try:
        result = extract_data(company)

        logger.info(f"[AGENT-TOOL] Data successfully extracted by LLM from: {company.get('url')}")

        new_results = extracted_results + [result]

    except Exception as e:
        logger.error(
            f"[LLM NODE] {company.get('url')} -> {e}"
        )

        new_results = extracted_results

    return {
        "extracted_results": new_results,
        "current_index": state["current_index"] + 1
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

def route_after_search(state: InternalState):
    """
    Route search results: 
    If there is an error, go to error.
    If PagineGialle is in the results, go to extract_pg node.
    Otherwise, go directly to scrape.
    """
    if state.get("error"):
        return "error"
        
    search_results = state.get("search_results", [])
    has_pg = any("paginegialle.it" in r.get("url", "").lower() for r in search_results)
    
    if has_pg:
        return "extract_pg"
    return "scrape"


def should_continue(state: InternalState):
    if state.get("error"):
        return "error"
    return "continue"

def loop_router(state: InternalState):
    """Decide if we continue looping."""
    if state["current_index"] < len(state["scraped_data"]):
        return "continue"

    return "end"

def extract_router(state: InternalState):

    if state.get("should_finish"):
        return "end"

    return "continue"


# BUILD GRAPH
graph = StateGraph(
    InternalState,
    input_schema=InputState,
    output_schema=OutputState
)

# Nodes
graph.add_node("search", search_node)
graph.add_node("extract_pg", extract_pg_node)
graph.add_node("scrape", scrape_node)
graph.add_node("extract", extract_node)
graph.add_node("llm", llm_node)
graph.add_node("final_answer", final_answer_node)
graph.add_node("error", error_node)

# Edges & Conditional Edges
graph.add_conditional_edges("search", route_after_search, {"extract_pg": "extract_pg", "scrape": "scrape", "error": "error"})
graph.add_edge("extract_pg", "scrape") 
graph.add_conditional_edges("scrape", should_continue, {"continue": "extract", "error": "error"})
graph.add_conditional_edges("extract", extract_router, {"continue": "llm", "end": "final_answer"})
graph.add_edge("llm", "extract")
graph.add_edge("final_answer", END)
graph.add_edge("error", END)
graph.set_entry_point("search")

app = graph.compile()

logger.info("[GRAPH] Pipeline compiled successfully")
logger.info("=" * 80)


# RUNNER
@traceable(name="pipeline_run_agent")
def run_agent(query: str):
    reset_stats()               # Reset stats at the beginning of each run
    stats = get_stats()         # Get stats instance

    stats.start()               # Start timing the execution

    initial_state = {"query": query}

    try:
        result = app.invoke(initial_state)
        stats.stop()            # Stop timing after execution finishes
    except Exception as e:
        stats.stop()
        result = {"final_answer": [], "error": str(e)}

    return result