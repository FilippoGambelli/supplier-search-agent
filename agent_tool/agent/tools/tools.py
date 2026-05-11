import json
from langchain_core.tools import tool

from agent_tool.agent.tools.search import search_web
from agent_tool.agent.tools.scrape import scrape_company_website, is_valid_company_result, extract_paginegialle_websites
from agent_tool.agent.tools.llm_extract import extract_data

from agent_tool.logger import logger


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
        formatted = "\n".join([f"- {r.get('title', 'N/A')}: {r.get('url', 'N/A')}" for r in results])
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