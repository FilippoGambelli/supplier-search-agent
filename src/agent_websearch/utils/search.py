import requests
from logger import logger
from config import SEARXNG_URL
from agent_websearch.exceptions import SearchTimeoutError, SearchConnectionError, SearchError


def search_web(query: str, limit: int = 2) -> list[dict]:
    """
    Perform a web search using a SearXNG instance.

    This function queries the configured SearXNG endpoint and extracts only the
    most relevant fields (title and URL) from the search results.

    Args:
        query (str): The search query string.
        limit (int, optional): Maximum number of results to return. Defaults to 2.

    Returns:
        list[dict]: A list of simplified search results, each containing:
            - title (str): The result title
            - url (str): The result URL

    Raises:
        SearchTimeoutError: If the SearXNG request times out.
        SearchConnectionError: If the network request fails.
        SearchError: For any other search failure (JSON parse, unexpected).
    """
    try:
        response = requests.get(
            f"{SEARXNG_URL}/search",
            params={
                "q": query,
                "format": "json",
                "engines": "google"
            }
        )

        response.raise_for_status()
        data = response.json()

        results = data.get("results", [])[:limit]

        simplified = [
            {
                "title": r.get("title", ""),
                "url": r.get("url", "")
            }
            for r in results
        ]

        logger.info(f"[SEARXNG] Found {len(simplified)} search results")

        return simplified

    except requests.exceptions.Timeout:
        raise SearchTimeoutError(f"SearXNG request timed out for query: {query}")

    except requests.exceptions.RequestException as e:
        raise SearchConnectionError(f"SearXNG request failed: {e}")

    except ValueError as e:
        raise SearchError(f"Invalid JSON from SearXNG: {e}")

    except Exception as e:
        raise SearchError(f"Unexpected search error: {e}")