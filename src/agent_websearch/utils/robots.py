from urllib.parse import urlparse
import requests
import protego
from logger import logger

ROBOTS_CACHE = {}


def can_fetch(url: str, user_agent: str = "MyCrawler") -> bool:
    parsed = urlparse(url)
    domain = parsed.scheme + "://" + parsed.netloc

    if domain not in ROBOTS_CACHE:
        robots_url = domain + "/robots.txt"
        try:
            response = requests.get(robots_url, timeout=5)
            response.raise_for_status()
            rp = protego.Protego.parse(response.text)
        except Exception as e:
            logger.warning(f"[ROBOTS] Failed to fetch robots.txt for {robots_url}: {e}")
            rp = protego.Protego.parse("")
        ROBOTS_CACHE[domain] = rp

    return ROBOTS_CACHE[domain].can_fetch(url, user_agent)
