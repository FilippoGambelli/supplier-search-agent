from typing import List, Dict, Optional
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from logger import logger
import protego

ROBOTS_CACHE = {}

def can_fetch(url: str, user_agent: str = "MyCrawler") -> bool:
    """
    Check robots.txt before requesting URL.
    Uses cache per domain to avoid re-downloading robots.txt.
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    domain = parsed.scheme + "://" + parsed.netloc

    if domain not in ROBOTS_CACHE:
        robots_url = domain + "/robots.txt"
        try:
            response = requests.get(robots_url, timeout=5)
            rp = protego.Protego.parse(response.text)
        except Exception:
            rp = protego.Protego.parse("")
        ROBOTS_CACHE[domain] = rp

    return ROBOTS_CACHE[domain].can_fetch(url, user_agent)


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


def fetch_html(url: str) -> Optional[str]:
    """
    Downloads the raw HTML content of a webpage.

    Performs an HTTP GET request with a timeout and custom headers,
    returning the page content as a string if successful.

    Returns None if the request fails or an error occurs.
    """
    try:
        if not can_fetch(url):
            logger.warning(f"[ROBOTS BLOCKED] {url}")
            return None

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=10
        )

        response.raise_for_status()
        return response.text
    except Exception as e:
        logger.error(f"[FETCH ERROR] {url} -> {e}")
        return None


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


def find_contact_page(base_url: str, html: str) -> Optional[str]:
    """
    Discover internal contact/about page.
    """

    soup = BeautifulSoup(html, "html.parser")

    for a in soup.find_all("a", href=True):
        href = a["href"].lower()
        if any(keyword in href for keyword in CONTACT_KEYWORDS):
            full_url = urljoin(base_url, href)
            return full_url

    logger.warning(f"[CONTACT PAGE NOT FOUND] {base_url}")

    return None


def extract_paginegialle_websites(pg_url: str, limit: int = 10) -> List[Dict[str, str]]:
    """
    PagineGialle Crawler. Steps:
    1. Scrapes the search result page.
    2. Collects the first N company profile links and their names.
    3. Visits each profile to find the 'Sito Web' (Website) button.
    4. Returns a list of dictionaries with company names and websites.
    """

    # STEP 1: SCRAPE SEARCH PAGE
    logger.info(f"[PAGINEGIALLE] Processing directory: {pg_url}")
    html = fetch_html(pg_url)
    if not html:
        logger.warning(f"[PAGINEGIALLE] Failed to fetch directory page: {pg_url}")
        return []

    soup = BeautifulSoup(html, "html.parser")

    # Stores extracted company profiles (name + paginegialle URL)
    profiles_data = [] 
    
    # Extract company profile links from search results page
    for a_tag in soup.select('div.search-itm__dx > div a.remove_blank_for_app'):
        href_pg = a_tag.get('href')
        
        # Extract company name from <h2>
        h2_tag = a_tag.select_one('h2')
        company_name = h2_tag.get_text(strip=True) if h2_tag else "Unknown name"

        if href_pg and "paginegialle.it" in href_pg:
            profiles_data.append({
                "name": company_name,
                "url": href_pg
            })

        if len(profiles_data) >= limit:
            break

    logger.info(f"[PAGINEGIALLE] Found {len(profiles_data)} profiles to inspect...")

    # STEP 2: VISIT PROFILES AND EXTRACT WEBSITE
    real_websites = []

    for profile in profiles_data:
        profile_url = profile["url"]
        company_name = profile["name"]
        
        p_html = fetch_html(profile_url)
        if not p_html:
            logger.warning(f"[PAGINEGIALLE] Could not fetch HTML for profile: {profile_url}")
            continue
        
        p_soup = BeautifulSoup(p_html, "html.parser")
        
        website_found = False

        # Look for official website button in profile page
        website_button = p_soup.select_one(
            'a[data-tr="scheda_azienda__cta_sitoweb"]'
        )

        if website_button:
            web_url = website_button.get("href")

            # Validate extracted URL (avoid paginegialle internal links)
            if (web_url and web_url.startswith("http") and "paginegialle" not in web_url):
                logger.info(f"[PAGINEGIALLE] Found website for {company_name} -> {web_url}")

                real_websites.append({
                    "title": company_name,
                    "url": web_url
                })
                website_found = True

        if not website_found: 
            logger.info(f"[PAGINEGIALLE] No website found for {company_name}")
    
    return real_websites


def scrape_company_website(url: str) -> Dict:
    """
    Scrape homepage + contact page.
    """

    logger.info(f"[SCRAPE] Scrape company website: {url}")

    homepage_html = fetch_html(url)
    if not homepage_html:
        logger.warning(f"[SCRAPE] Could not fetch homepage: {url}")
        return None

    # Homepage text
    homepage_text = html_to_text(homepage_html)

    logger.info(f"[SCRAPE] Homepage text extracted: {len(homepage_text)} characters")

    # Find contact page
    contact_page = find_contact_page(url, homepage_html)

    contact_text = None

    # Contact page scraping
    if contact_page:
        contact_html = fetch_html(contact_page)
        if contact_html:
            contact_text = html_to_text(contact_html)
            logger.info(f"[SCRAPE] Contact page text extracted: {len(contact_text)} characters")
    else:
        logger.info(f"[SCRAPE] Contact page not found in {url}")

    return {
        "homepage_text": homepage_text,
        "contact_text": contact_text,
    }