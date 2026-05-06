"""
LangGraph-based AI Agent for supplier search.
Implements a sequential graph with nodes: Search -> Scrape -> Extract -> FinalAnswer
"""
from dotenv import load_dotenv
import os
load_dotenv()

from typing import TypedDict, List, Dict, Optional, Annotated
import operator

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama


from agent.search import search_web
from agent.scrape import scrape_company_website, is_valid_company_result
from agent.extract import extract_company_data

from src.logger import logger

# =========================================================
# AGENT STATE
# =========================================================

class AgentState(TypedDict):
    """State passed between nodes in the graph."""
    query: str                      # User query
    search_results: List[Dict]      # Results from SearXNG
    scraped_data: List[Dict]        # Scraped homepage + contact pages
    extracted: List[Dict]           # Extracted company data from LLM
    final_answer: List[Dict]        # Final aggregated answer
    error: Optional[str]            # Any error that occurred


# =========================================================
# NODE FUNCTIONS
# =========================================================

def search_node(state: AgentState, config: RunnableConfig) -> Dict:
    """Search the web using SearXNG."""
    query = state.get("query", "")

    logger.info(f"[SEARCH NODE] Query: {query}")

    try:
        results = search_web(query)
        logger.info(f"[SEARCH NODE] Found {len(results)} results")

        return {"search_results": results}

    except Exception as e:
        logger.error(f"[SEARCH NODE ERROR] {e}")
        return {"error": str(e), "search_results": []}


def scrape_node(state: AgentState, config: RunnableConfig) -> Dict:
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
            logger.error(f"[SCRAPE NODE ERROR] {url} -> {e}")
            scraped_data.append({
                "title": title,
                "url": url,
                "homepage_text": None,
                "contact_text": None,
                "error": str(e)
            })

    logger.info(f"[SCRAPE NODE] Successfully scraped {len(scraped_data)} websites")

    return {"scraped_data": scraped_data}


def extract_node(state: AgentState, config: RunnableConfig) -> Dict:
    """Extract company data from scraped content using LLM."""
    scraped_data = state.get("scraped_data", [])

    logger.info(f"[EXTRACT NODE] Extracting data from {len(scraped_data)} companies")

    extracted = []

    for company in scraped_data:
        try:
            data = extract_company_data(company)
            extracted.append(data)

        except Exception as e:
            logger.error(f"[EXTRACT NODE ERROR] {company.get('url')} -> {e}")
            extracted.append({
                "company_name": company.get("title"),
                "website": company.get("url"),
                "error": str(e)
            })

    logger.info(f"[EXTRACT NODE] Extracted data for {len(extracted)} companies")

    return {"extracted": extracted}


def final_answer_node(state: AgentState, config: RunnableConfig) -> Dict:
    """Create final answer from extracted data."""
    extracted = state.get("extracted", [])
    error = state.get("error")

    logger.info(f"[FINAL ANSWER NODE] Generating final answer with {len(extracted)} items")

    # Filter out errors if any
    valid_results = [e for e in extracted if "error" not in e or e.get("error") != "Invalid JSON returned by model"]

    return {
        "final_answer": valid_results,
        "error": error
    }


# =========================================================
# BUILD GRAPH
# =========================================================

# Create graph
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("search", search_node)
graph.add_node("scrape", scrape_node)
graph.add_node("extract", extract_node)
graph.add_node("final_answer", final_answer_node)

# Define edges (sequential flow)
graph.add_edge("search", "scrape")
graph.add_edge("scrape", "extract")
graph.add_edge("extract", "final_answer")

# Set entry and end
graph.set_entry_point("search")
graph.add_edge("final_answer", END)

# Compile
app = graph.compile()

logger.info("[BUILD AGENT GRAPH] Graph compiled successfully")


# =========================================================
# AGENT RUNNER
# =========================================================

def run_agent(query: str) -> Dict:
    """
    Run the full agent pipeline for a given query.
    Returns the final answer.
    """
    logger.info(f"[AGENT RUN] Starting pipeline for: '{query}'")

    initial_state = {
        "query": query,
        "search_results": [],
        "scraped_data": [],
        "extracted": [],
        "final_answer": [],
        "error": None
    }

    result = app.invoke(initial_state)

    logger.info(f"[AGENT RUN] Completed pipeline for: '{query}'")

    return result