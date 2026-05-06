from typing import List, Dict, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from src.logger import logger


# =========================================================
# HTTP HEADERS
# =========================================================
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    ),
    "Accept-Language": "it-IT,it;q=0.9,en;q=0.8",
}


# =========================================================
# BLACKLIST
# Ignore directories / aggregators
# =========================================================
BLACKLIST = [
    "paginegialle",
    "pagine bianche",
    "paginebianche",
    "yelp",
    "virgilio",
    "prontopro",
    "facebook",
    "instagram",
    "linkedin",
]


# =========================================================
# CONTACT PAGE KEYWORDS
# =========================================================
CONTACT_KEYWORDS = [
    "contatti",
    "contact",
    "contacts",
    "about",
    "chi-siamo",
    "azienda",
]


# =========================================================
# FILTER SEARCH RESULTS
# =========================================================
def is_valid_company_result(title: str, url: str) -> bool:
    """
    Return False if result belongs to unwanted websites.
    """

    text = f"{title} {url}".lower()

    return not any(bad in text for bad in BLACKLIST)


# =========================================================
# FETCH HTML
# =========================================================
def fetch_html(url: str) -> Optional[str]:
    """
    Download raw HTML from a webpage.
    """

    try:

        logger.info(f"[FETCH] Downloading: {url}")

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=10
        )

        response.raise_for_status()

        logger.info(f"[FETCH] Success: {url}")

        return response.text

    except Exception as e:

        logger.error(f"[FETCH ERROR] {url} -> {e}")

        return None


# =========================================================
# HTML -> TEXT
# =========================================================
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

    clean_text = "\n".join(
        line for line in lines
        if line
    )

    return clean_text


# =========================================================
# FIND CONTACT PAGE
# =========================================================
def find_contact_page(base_url: str, html: str) -> Optional[str]:
    """
    Discover internal contact/about page.
    """

    soup = BeautifulSoup(html, "html.parser")

    for a in soup.find_all("a", href=True):

        href = a["href"].lower()

        if any(keyword in href for keyword in CONTACT_KEYWORDS):

            full_url = urljoin(base_url, href)

            logger.info(f"[CONTACT PAGE FOUND] {full_url}")

            return full_url

    logger.warning(f"[CONTACT PAGE NOT FOUND] {base_url}")

    return None


# =========================================================
# SCRAPE COMPANY WEBSITE
# =========================================================
def scrape_company_website(url: str) -> Dict:
    """
    Scrape homepage + contact page.
    """

    logger.info("=" * 80)
    logger.info(f"[COMPANY SCRAPING START] {url}")

    homepage_html = fetch_html(url)

    if not homepage_html:

        logger.warning(f"[SCRAPING FAILED] Could not fetch homepage: {url}")

        return {
            "homepage_text": None,
            "contact_text": None,
        }

    # Homepage text
    homepage_text = html_to_text(homepage_html)

    logger.info(
        f"[HOMEPAGE TEXT EXTRACTED] "
        f"{len(homepage_text)} characters"
    )

    # Find contact page
    contact_page = find_contact_page(url, homepage_html)

    contact_text = None

    # Contact page scraping
    if contact_page:

        contact_html = fetch_html(contact_page)

        if contact_html:

            contact_text = html_to_text(contact_html)

            logger.info(
                f"[CONTACT TEXT EXTRACTED] "
                f"{len(contact_text)} characters"
            )

    logger.info(f"[COMPANY SCRAPING END] {url}")
    logger.info("=" * 80)

    return {
        "homepage_text": homepage_text,
        "contact_text": contact_text,
    }