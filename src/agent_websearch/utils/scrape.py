from typing import List, Dict, Optional
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from logger import logger
from agent_websearch.utils.robots import can_fetch
from agent_websearch.exceptions import WebSearchError, RobotsBlockedError, ScrapeError


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    ),
    "Accept-Language": "it-IT,it;q=0.9,en;q=0.8",
}


BLACKLIST = [
    "pagine bianche",
    "paginebianche",
    "paginegialle",
    "yelp",
    "virgilio",
    "prontopro",
    "facebook",
    "instagram",
    "linkedin",
]


CONTACT_KEYWORDS = [
    "contatti",
    "contact",
    "contacts",
    "about",
    "chi-siamo",
    "azienda",
]


def is_valid_company_result(title: str, url: str) -> bool:
    """
    Returns True if the result is a valid company entry.

    Filters out results coming from blacklisted or unwanted websites
    based on keywords found in the title or URL.
    """
    text = f"{title} {url}".lower()

    return not any(bad in text for bad in BLACKLIST)


def fetch_html(url: str) -> str:
    """
    Downloads the raw HTML content of a webpage.

    Performs an HTTP GET request with a timeout and custom headers,
    returning the page content as a string if successful.

    Raises:
        RobotsBlockedError: If robots.txt disallows scraping.
        ScrapeError: If the request fails or an error occurs.
    """
    try:
        if not can_fetch(url):
            raise RobotsBlockedError(f"robots.txt disallows: {url}")

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=10
        )

        response.raise_for_status()
        return response.text
    except requests.exceptions.Timeout:
        raise ScrapeError(f"Request timed out for {url}")
    except requests.exceptions.RequestException as e:
        raise ScrapeError(f"Request failed for {url}: {e}")


def html_to_text(html: str) -> str:
    """
    Convert HTML into clean readable text.
    """

    soup = BeautifulSoup(html, "html.parser")

    # Remove non-readable tags
    for tag in soup([
        "script",
        "style",
        "noscript",
        "svg",
        "img",
        "footer",
        "header",
    ]):
        tag.decompose()

    # Extract visible text
    text = soup.get_text(separator="\n")

    # Normalize whitespace
    lines = [line.strip() for line in text.splitlines()]

    clean_text = "\n".join(line for line in lines if line)

    return clean_text


def find_contact_url(base_url: str, html: str) -> Optional[str]:
    """
    Discover internal contact/about page.
    """
    soup = BeautifulSoup(html, "html.parser")

    for a in soup.find_all("a", href=True):
        href = a["href"].lower()
        if any(keyword in href for keyword in CONTACT_KEYWORDS):
            return urljoin(base_url, href)

    logger.info(f"[CONTACT PAGE NOT FOUND] {base_url}")
    return None


def scrape_company_website(url: str) -> Dict:
    """
    Scrape homepage + contact page.

    Raises:
        ScrapeError: If the homepage cannot be fetched (robots or network).
    """
    logger.info(f"[SCRAPE] Scrape company website: {url}")

    homepage_html = fetch_html(url)
    homepage_text = html_to_text(homepage_html)

    logger.info(f"[SCRAPE] Homepage text extracted: {len(homepage_text)} characters")

    contact_page = find_contact_url(url, homepage_html)
    contact_text = None

    if contact_page:
        try:
            contact_html = fetch_html(contact_page)
            contact_text = html_to_text(contact_html)
            logger.info(f"[SCRAPE] Contact page text extracted: {len(contact_text)} characters")
        except WebSearchError as e:
            logger.warning(f"[SCRAPE] Could not fetch contact page {contact_page}: {e}")
    else:
        logger.info(f"[SCRAPE] Contact page not found in {url}")

    return {
        "homepage_text": homepage_text,
        "contact_text": contact_text,
    }