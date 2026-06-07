import requests
from logger import logger
from config import SEARXNG_URL

def search_web(query: str, limit: int = 2):
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

        Returns an empty list in case of failure.
    """
    try:
        response = requests.get(
            f"{SEARXNG_URL}/search",
            params={
                "q": query,
                "format": "json"
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
        logger.error(f"[SEARX SEARCH] Request timed out. Error: {e}")
        return []

    except requests.exceptions.RequestException as e:
        logger.error(f"[SEARX SEARCH] Request failed. Error: {e}")
        return []

    except ValueError as e:
        logger.error(f"[SEARX SEARCH] Invalid JSON response. Error: {e}")
        return []

    except Exception as e:
        logger.error(f"[SEARX SEARCH] Unexpected error. Error: {e}")
        return []