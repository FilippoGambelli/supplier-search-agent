from typing import TypedDict, List, Dict, Optional
from urllib.parse import urlparse
from langgraph.graph import StateGraph, END
from agent_websearch.utils.search import search_web
from agent_websearch.utils.scrape import scrape_company_website, is_valid_company_result, extract_paginegialle_websites
from agent_websearch.utils.extract import extract_data
from logger import logger
from config import SEARXNG_RESULTS_LIMIT
from main import print_node_pipeline
from langsmith import traceable


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


@traceable(name="search_node")
def search_node(state: InputState) -> Dict:
    """
    Performs a web search using SearXNG based on the input query in the state.

    Logs the search process, retrieves results, and returns a simplified list
    of search results. In case of failure, returns an error message and an
    empty result set.
    """
    query = state["query"]
    try:
        logger.info(f"[SEARCH NODE] Starting web search with query: {query}")
        results = search_web(query, limit=SEARXNG_RESULTS_LIMIT)
        return {"search_results": results}
    except Exception as e:
        logger.error(f"[SEARCH NODE] Unexpected error: {e}")
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
    logger.info("=" * 80)
    search_results = state.get("search_results", [])
    pg_results = []

    for result in search_results:
        url = result.get("url", "")
        
        if "paginegialle.it" in url.lower():
            logger.info(f"[EXTRACT PG NODE] PagineGialle detected, extracting real websites from: {url}")

            try:
                discovered_urls = extract_paginegialle_websites(url)
                for item in discovered_urls:
                    pg_results.append({
                        "title": item["title"],
                        "url": item["url"]
                    })
            except Exception as e:
                logger.error(f"[EXTRACT PG NODE] Error extracting URL {url}: {e}")

    return {"pg_results": pg_results}

@traceable(name="scrape_node")
def scrape_node(state: InternalState) -> Dict:
    """
    Scrape websites from search results.
    Combines search_results and pg_results, removes duplicates, applies blacklist, and runs the standard scrape flow.
    """
    logger.info("=" * 80)
    search_results = state.get("search_results", [])
    pg_results = state.get("pg_results", [])
    scraped_data = []

    combined_results = search_results + pg_results
    logger.info(f"[SCRAPE NODE] Combining {len(search_results)} search results with {len(pg_results)} PagineGialle results")    

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
        logger.info(f"[SCRAPE NODE] Removed {duplicates_count} duplicate URLs")

    # Standard blacklist check for other results
    for result in deduplicated_results:
        title = result.get("title", "")
        url = result.get("url", "")

        # Check the blacklist and make sure to skip paginegialle.it 
        if "paginegialle.it" in url.lower() or not is_valid_company_result(title, url):
            logger.warning(f"[SCRAPE NODE] Skipping blacklisted or directory: {url}")
            continue

        try:
            data = scrape_company_website(url)
            if data is None:
                continue
            data.update({"title": title, "url": url})
            scraped_data.append(data)
        except Exception as e:
            logger.error(f"[SCRAPE NODE] {url} -> {e}")

    if not scraped_data:
        return {"error": "No valid data scraped.", "scraped_data": []}
    
    logger.info("=" * 80)

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
        logger.info(f"[LLM NODE] Data successfully extracted by LLM from: {company.get('url')}")
        new_results = extracted_results + [result]
    except Exception as e:
        logger.error(f"[LLM NODE] {company.get('url')} -> {e}")
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
    if state["current_index"] < len(state["scraped_data"]):
        return "continue"
    return "end"

def extract_router(state: InternalState):
    if state.get("should_finish"):
        return "end"
    return "continue"


graph = StateGraph(
    InternalState,
    input_schema=InputState,
    output_schema=OutputState
)

graph.add_node("search", search_node)
graph.add_node("extract_pg", extract_pg_node)
graph.add_node("scrape", scrape_node)
graph.add_node("extract", extract_node)
graph.add_node("llm", llm_node)
graph.add_node("final_answer", final_answer_node)
graph.add_node("error", error_node)

graph.add_conditional_edges("search", route_after_search, {"extract_pg": "extract_pg", "scrape": "scrape", "error": "error"})
graph.add_edge("extract_pg", "scrape") 
graph.add_conditional_edges("scrape", should_continue, {"continue": "extract", "error": "error"})
graph.add_conditional_edges("extract", extract_router, {"continue": "llm", "end": "final_answer"})
graph.add_edge("llm", "extract")
graph.add_edge("final_answer", END)
graph.add_edge("error", END)
graph.set_entry_point("search")

app = graph.compile()

@traceable(name="pipeline_run_agent")
def run_agent(query: str, verbose = True):
    if verbose:
        try:
            last_event = None
            for event in app.stream({"query": query}):
                print_node_pipeline(event)
                last_event = event
            message = last_event.get("final_answer", {}).get("final_answer")
            return message, None
            
        except Exception as e:
            logger.error(f"[PIPELINE] Error: {e}")
            return None, str(e)
    else:
        try:
            result = app.invoke({"query": query})
            return result, None
            
        except Exception as e:
            logger.error(f"[PIPELINE] Error: {e}")
            return None, str(e)