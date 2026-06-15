import json
from langchain_core.tools import tool
from agent_websearch.utils.search import search_web
from agent_websearch.utils.scrape import scrape_company_website, is_valid_company_result
from agent_websearch.utils.paginegialle import extract_paginegialle_websites
from agent_websearch.utils.extract import extract_data
from agent_websearch.exceptions import WebSearchError, InsufficientDataError
from logger import logger
from config import PAGINEGIALLE_RESULTS_LIMIT, SEARXNG_RESULTS_LIMIT


@tool
def search_suppliers(query: str) -> str:
    """
    Search for suppliers on the web using SearXNG.
    Use this first to find potential companies.
    Returns a string containing a list of titles and URLs.
    """
    logger.info(f"[AGENT TOOL] Tool search_suppliers called with query: {query}")
    try:
        results = search_web(query, limit=SEARXNG_RESULTS_LIMIT)
        if not results:
            return "No search results returned. The SearXNG instance may be unavailable."
        formatted = "\n".join([f"- {r.get('title', 'N/A')}: {r.get('url', 'N/A')}" for r in results])
        return f"Found {len(results)} results:\n{formatted}"
    except WebSearchError as e:
        logger.error(f"[TOOL ERROR] search_suppliers: {e}")
        return f"Error: {e}"
    except Exception as e:
        logger.error(f"[TOOL ERROR] search_suppliers unexpected error: {e}")
        return f"Unexpected error: {e}"


@tool
def is_valid_company(title: str, url: str) -> str:
    """
    Check if a search result is a valid company website.
    Returns a string with instructions on how to proceed.
    """
    logger.info(f"[AGENT TOOL] Tool is_valid_company called for URL: {url}")
    try:
        url_lower = url.lower()

        if "paginegialle.it" in url_lower:
            return f"FALSE [{url}] - PagineGialle directory. ACTION REQUIRED: You MUST use the `extract_from_paginegialle` tool on this URL immediately to extract the real company websites."

        is_valid = is_valid_company_result(title, url)

        if is_valid:
            return f"TRUE [{url}] - Valid company website. You can proceed to use `research_and_extract_company` on this URL."
        else:
            return f"FALSE [{url}] - Generic aggregator, directory, or social media. Ignore this URL and move to the next one."

    except Exception as e:
        logger.error(f"[TOOL ERROR] is_valid_company failed for {url}: {e}")
        return f"Error: Unable to validate company: {e}"


@tool
def extract_from_paginegialle(pg_url: str) -> str:
    """
    Extract real company websites from a PagineGialle directory page.
    Use this ONLY when you find a PagineGialle link in your search results.
    """
    logger.info(f"[AGENT TOOL] Tool extract_from_paginegialle called for URL: {pg_url}")
    try:
        results = extract_paginegialle_websites(pg_url, limit=PAGINEGIALLE_RESULTS_LIMIT)
        if not results:
            return "No real websites found from PagineGialle page."
        formatted = "\n".join([f"- {r.get('title', 'N/A')}: {r.get('url', 'N/A')}" for r in results])
        return f"Found these real websites:\n{formatted}"
    except WebSearchError as e:
        logger.error(f"[TOOL ERROR] extract_from_paginegialle: {e}")
        return f"Error: {e}"
    except Exception as e:
        logger.error(f"[TOOL ERROR] extract_from_paginegialle unexpected error: {e}")
        return f"Unexpected error: {e}"


@tool
def research_and_extract_company(url: str, title: str = "") -> str:
    """
    SUB-AGENT TOOL: Deeply analyze a specific company website.
    Use this tool on valid company URLs to extract structured business information.
    It automatically scrapes the website and extracts the data.
    """
    logger.info(f"[AGENT TOOL] Tool research_and_extract_company called for URL: {url}")
    try:
        data = scrape_company_website(url)

        company_payload = {
            "url": url,
            "title": title,
            "homepage_text": data.get("homepage_text", ""),
            "contact_text": data.get("contact_text", "")
        }

        result = extract_data(company_payload)
        return json.dumps(result, ensure_ascii=False)

    except InsufficientDataError as e:
        logger.warning(f"[TOOL ERROR] research_and_extract_company insufficient data for {url}: {e}")
        return f"INSUFFICIENT_DATA: {e}"
    except WebSearchError as e:
        logger.error(f"[TOOL ERROR] research_and_extract_company: {e}")
        return f"Error: {e}"
    except Exception as e:
        logger.error(f"[TOOL ERROR] research_and_extract_company unexpected error: {e}")
        return f"Unexpected error: {e}"