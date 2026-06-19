import json
from typing import TypedDict, List, Dict, Optional
from urllib.parse import urlparse
from langgraph.graph import StateGraph, END
from agent_websearch.utils.search import search_web
from agent_websearch.utils.scrape import scrape_company_website, is_valid_company_result
from agent_websearch.utils.paginegialle import extract_paginegialle_websites
from agent_websearch.utils.extract import extract_data
from agent_websearch.exceptions import WebSearchError, InsufficientDataError, NotACompanyError
from logger import logger
from config import SEARXNG_RESULTS_LIMIT
from langsmith import traceable


class InputState(TypedDict):
    query: str
    verbose: bool

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
    verbose = state.get("verbose", False)
    try:
        logger.info(f"[SEARCH NODE] Starting web search with query: {query}")
        results = search_web(query, limit=SEARXNG_RESULTS_LIMIT)
        if verbose:
            print(f"\n[SEARCH] \"{query}\" - {len(results)} result{'' if len(results) == 1 else 's'}")
            for r in results:
                url = r.get("url", "")
                title = r.get("title", "")
                label = title[:80] + "..." if len(title) > 80 else title
                print(f"  \u2022 {label}")
                print(f"    {url}")
        return {"search_results": results}
    except WebSearchError as e:
        logger.error(f"[SEARCH NODE] {e}")
        if verbose:
            print(f"\n[SEARCH] Error: {e}")
        return {
            "error": str(e),
            "search_results": []
        }

@traceable(name="extract_pg_node")
def extract_pg_node(state: InternalState) -> Dict:
    """
    Dedicated node to handle PagineGialle links.
    Extracts real company URLs and saves them separately in pg_results.
    """
    verbose = state.get("verbose", False)
    search_results = state.get("search_results", [])
    pg_results = []

    pg_urls = [r.get("url", "") for r in search_results if "paginegialle.it" in r.get("url", "").lower()]

    for url in pg_urls:
        logger.info(f"[EXTRACT PG NODE] PagineGialle detected, extracting real websites from: {url}")

        try:
            discovered_urls = extract_paginegialle_websites(url)
            for item in discovered_urls:
                pg_results.append({
                    "title": item["title"],
                    "url": item["url"]
                })
        except WebSearchError as e:
            logger.error(f"[EXTRACT PG NODE] Error extracting URL {url}: {e}")
            if verbose:
                print(f"\n[EXTRACT PG] {url} - error: {e}")

    if verbose and pg_results:
        pg_url_short = pg_urls[0][:70] + "..." if len(pg_urls[0]) > 70 else pg_urls[0]
        print(f"\n[EXTRACT PG] {pg_url_short} - extracted {len(pg_results)} website{'' if len(pg_results) == 1 else 's'}")
        for r in pg_results:
            title = r.get("title", "")
            url = r.get("url", "")
            label = title[:70] + "..." if len(title) > 70 else title
            print(f"  \u2022 {label}")
            print(f"    {url}")
    elif verbose and pg_urls:
        pg_url_short = pg_urls[0][:70] + "..." if len(pg_urls[0]) > 70 else pg_urls[0]
        print(f"\n[EXTRACT PG] {pg_url_short} - no websites found")

    return {"pg_results": pg_results}

@traceable(name="scrape_node")
def scrape_node(state: InternalState) -> Dict:
    """
    Scrape websites from search results.
    Combines search_results and pg_results, removes duplicates, applies blacklist, and runs the standard scrape flow.
    """
    verbose = state.get("verbose", False)
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

    lines = []
    for result in deduplicated_results:
        title = result.get("title", "")
        url = result.get("url", "")

        if not is_valid_company_result(title, url):
            logger.warning(f"[SCRAPE NODE] Skipping blacklisted or directory: {url}")
            if verbose:
                lines.append(f"\u2717 {url} (blacklisted)")
            continue

        try:
            data = scrape_company_website(url)
            data.update({"title": title, "url": url})
            scraped_data.append(data)
            if verbose:
                hl = len(data.get("homepage_text", "") or "")
                cl = len(data.get("contact_text", "") or "")
                label = title[:60] + "..." if len(title) > 60 else title
                lines.append(f"\u2713 {label} ({hl}+{cl} chars)")
        except WebSearchError as e:
            logger.error(f"[SCRAPE NODE] {url} -> {e}")
            if verbose:
                err_str = str(e)
                if "403" in err_str:
                    short_err = "403 Forbidden"
                elif "404" in err_str:
                    short_err = "404 Not Found"
                elif "timeout" in err_str.lower() or "timed out" in err_str.lower():
                    short_err = "timeout"
                elif "robots.txt" in err_str.lower():
                    short_err = "blocked by robots.txt"
                else:
                    short_err = err_str[:60] + "..." if len(err_str) > 60 else err_str
                label = title[:60] + "..." if len(title) > 60 else title
                lines.append(f"\u2717 {label} - {short_err}")

    if verbose:
        total = len(combined_results)
        dup_info = f", {duplicates_count} duplicate{'' if duplicates_count == 1 else 's'} removed" if duplicates_count > 0 else ""
        print(f"\n[SCRAPE] {total} URL{'' if total == 1 else 's'} ({len(search_results)} search + {len(pg_results)} PG{dup_info})")
        for line in lines:
            print(f"  {line}")

    if not scraped_data:
        if verbose:
            print(f"  No valid data scraped.")
        return {"error": "No valid data scraped.", "scraped_data": []}

    logger.info("=" * 80)

    return {
        "scraped_data": scraped_data,
        "current_index": 0,
        "extracted_results": [],
    }

@traceable(name="extract_node")
def extract_node(state: InternalState) -> Dict:
    verbose = state.get("verbose", False)
    scraped_data = state.get("scraped_data", [])
    current_index = state.get("current_index", 0)

    if not scraped_data:
        if verbose:
            print(f"\n[EXTRACT] No data to process")
        return {
            "should_finish": True,
            "error": "No scraped data available for extraction"
        }

    if current_index >= len(scraped_data):
        return {
            "should_finish": True
        }

    current_company = scraped_data[current_index]
    url = current_company.get("url", "")
    title = current_company.get("title", "")
    label = title[:80] + "..." if len(title) > 80 else title

    if verbose:
        print(f"\n[EXTRACT] ({current_index+1}/{len(scraped_data)}) {label}")
        print(f"  {url}")

    return {
        "current_company": current_company,
        "should_finish": False,
        "current_index": current_index,
        "total_companies": len(scraped_data),
    }

@traceable(name="llm_extract_node")
def llm_node(state: InternalState) -> Dict:
    verbose = state.get("verbose", False)
    company = state["current_company"]
    extracted_results = state.get("extracted_results", [])
    was_extracted = True
    url = company.get("url", "")
    title = company.get("title", "")

    try:
        result = extract_data(company)
        new_results = extracted_results + [result]
        logger.info(f"[LLM NODE] Data successfully extracted from: {url}")
        if verbose:
            emails = result.get("email", [])
            phones = result.get("phone", [])
            ei = f"{len(emails)} email" if emails else "no email"
            pi = f"{len(phones)} phone" if phones else "no phone"
            print(f"\n[LLM] {title} - {ei}, {pi}")
    except InsufficientDataError as e:
        logger.warning(f"[LLM NODE] {e}")
        new_results = extracted_results
        was_extracted = False
        if verbose:
            print(f"[LLM] ({state['current_index']+1}) {title} - skipped (no email/phone)")
    except NotACompanyError as e:
        logger.info(f"[LLM NODE] {e}")
        new_results = extracted_results
        was_extracted = False

        if verbose:
            print(f"[LLM] ({state['current_index']+1}) {title} - skipped (not a company)")
    except WebSearchError as e:
        logger.error(f"[LLM NODE] {url} -> {e}")
        new_results = extracted_results
        was_extracted = False
        if verbose:
            print(f"[LLM] ({state['current_index']+1}) {title} - error: {e}")
    except Exception as e:
        logger.error(f"[LLM NODE] {url} -> {e}")
        new_results = extracted_results
        was_extracted = False
        if verbose:
            print(f"[LLM] ({state['current_index']+1}) {title} - error: {e}")

    return {
        "extracted_results": new_results,
        "current_index": state["current_index"] + 1,
        "company_url": url,
        "company_title": title,
        "was_extracted": was_extracted,
    }


def final_answer_node(state: InternalState) -> Dict:
    """Return final aggregated results."""
    verbose = state.get("verbose", False)
    results = state.get("extracted_results", [])

    if verbose:
        if not results:
            print(f"\n[FINAL ANSWER] No companies extracted")
        else:
            print(f"\n[FINAL ANSWER] {len(results)} compan{'' if len(results) == 1 else 'ies'} returned")
            for c in results:
                name = c.get("name", "")
                web = c.get("website", "")
                em = c.get("email", [])
                ph = c.get("phone", [])
                print(f"  \u2022 {name}")
                print(f"    {web} | {len(em)} email{'' if len(em) == 1 else 's'}, {len(ph)} phone{'' if len(ph) == 1 else 's'}")

    return {
        "final_answer": results,
        "error": None
    }


def error_node(state: InternalState):
    verbose = state.get("verbose", False)
    err = state.get("error", "Unknown error")
    if verbose:
        print(f"\n[ERROR] {err}")
    return {
        "final_answer": [],
        "error": err
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
def run_agent(query: str, verbose=True):
    initial_state = {"query": query, "verbose": verbose}
    try:
        result = app.invoke(initial_state)
        error = result.get("error")
        if error:
            logger.error(f"[PIPELINE ERROR] Graph returned error: {error}")
            return None, error
        final_answer = result.get("final_answer", [])
        return json.dumps(final_answer, indent=2, ensure_ascii=False), None
    except Exception as e:
        logger.error(f"[PIPELINE] Error: {e}")
        return None, str(e)