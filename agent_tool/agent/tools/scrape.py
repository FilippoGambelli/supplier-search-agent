from typing import List, Dict, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from agent_tool.logger import logger


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
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=10
        )

        response.raise_for_status()

        logger.info(f"[FETCH] Successfully downloaded: {url}")

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

            logger.info(f"[CONTACT] Found contact page: {full_url}")

            return full_url

    logger.warning(f"[CONTACT PAGE NOT FOUND] {base_url}")

    return None


# =========================================================
# EXTRACT PAGINEGIALLE
# =========================================================
def extract_paginegialle_websites(pg_url: str, limit: int = 10) -> List[Dict[str, str]]:
    """
    Sub-problem: PagineGialle Crawler
    1. Scrapes the search result page.
    2. Collects the first N company profile links and their names.
    3. Visits each profile to find the 'Sito Web' (Website) button.
    4. Returns a list of dictionaries with company names and websites.
    """

    logger.info(f"[PAGINEGIALLE] Processing directory: {pg_url}")
    html = fetch_html(pg_url)
    if not html:
        logger.warning(f"[PAGINEGIALLE] Failed to fetch directory page: {pg_url}")
        return []

    soup = BeautifulSoup(html, "html.parser")
    
    # company_name and href paginegialle
    profiles_data = [] 
    
    # Extract company profile links using common PagineGialle selectors
    for a_tag in soup.select('div.search-itm__dx > div a.remove_blank_for_app'):
        href_pg = a_tag.get('href')
        
        # FInd h2 tag
        h2_tag = a_tag.select_one('h2')
        company_name = h2_tag.get_text(strip=True) if h2_tag else "Unknown name"

        if href_pg and "paginegialle.it" in href_pg:
            profiles_data.append({
                "title": company_name,
                "url": href_pg
            })

        if len(profiles_data) >= limit:
            break

    logger.info(f"[PAGINEGIALLE] Found {len(profiles_data)} profiles to inspect.")
    logger.info("=" * 80)

    # company_name and website
    real_websites = []

    for profile in profiles_data:
        profile_url = profile["url"]
        company_name = profile["title"]
        
        logger.info(f"[PAGINEGIALLE] Checking profile: {profile_url} for '{company_name}'")
        p_html = fetch_html(profile_url)
        
        if not p_html:
            logger.warning(f"[PAGINEGIALLE] Could not fetch HTML for profile.")
            continue
        
        p_soup = BeautifulSoup(p_html, "html.parser")
        
        website_found = False

        website_button = p_soup.select_one(
            'a[data-tr="scheda_azienda__cta_sitoweb"]'
        )

        if website_button:
            web_url = website_button.get("href")

            if (web_url and web_url.startswith("http") and "paginegialle" not in web_url):
                logger.info(f"[PAGINEGIALLE] SUCCESS: Found real website -> {web_url}")

                real_websites.append({
                    "title": company_name,
                    "url": web_url
                })
                website_found = True

        if not website_found: 
            logger.info(f"[PAGINEGIALLE] No website found in this profile.")
            
        logger.info("=" * 80)
    
    return real_websites


# =========================================================
# SCRAPE COMPANY WEBSITE
# =========================================================
def scrape_company_website(url: str) -> Dict:
    """
    Scrape homepage + contact page.
    """
    logger.info(f"[COMPANY SCRAPING START] {url}")

    homepage_html = fetch_html(url)

    if not homepage_html:
        logger.warning(f"[SCRAPING FAILED] Could not fetch homepage: {url}")
        return None

    # Homepage text
    homepage_text = html_to_text(homepage_html)

    logger.info(
        f"[SCRAPE] Homepage text extracted: {len(homepage_text)} characters"
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
                f"[SCRAPE] Contact page text extracted: {len(contact_text)} characters"
            )

    logger.info("=" * 80)

    return {
        "homepage_text": homepage_text,
        "contact_text": contact_text,
    }