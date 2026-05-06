import requests

from .config import SEARXNG_URL

def search_web(query: str, limit: int = 15):
    """
    Query SearXNG and return only title + url.
    """

    response = requests.get(
        f"{SEARXNG_URL}/search",
        params={"q": query, "format": "json"}
    )

    response.raise_for_status()
    data = response.json()

    results = data.get("results", [])[:limit]

    # Keep only essential fields
    simplified = []

    for r in results:
        simplified.append({
            "title": r.get("title", ""),
            "url": r.get("url", "")
        })

    return simplified