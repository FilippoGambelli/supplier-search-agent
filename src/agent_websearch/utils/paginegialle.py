from typing import List, Dict, Optional
from bs4 import BeautifulSoup
from logger import logger
from agent_websearch.utils.scrape import fetch_html
from agent_websearch.exceptions import WebSearchError, PagineGialleParseError


def _parse_profiles_from_directory(html: str, limit: int) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    profiles = []

    for a_tag in soup.select('div.search-itm__dx > div a.remove_blank_for_app'):
        href_pg = a_tag.get('href')
        h2_tag = a_tag.select_one('h2')
        company_name = h2_tag.get_text(strip=True) if h2_tag else "Unknown name"

        if href_pg and "paginegialle.it" in href_pg:
            profiles.append({
                "name": company_name,
                "url": href_pg
            })

        if len(profiles) >= limit:
            break

    return profiles


def _extract_website_from_profile(profile_url: str, company_name: str) -> Optional[Dict[str, str]]:
    try:
        p_html = fetch_html(profile_url)
    except WebSearchError as e:
        logger.warning(f"[PAGINEGIALLE] Could not fetch profile {profile_url}: {e}")
        return None

    p_soup = BeautifulSoup(p_html, "html.parser")
    website_button = p_soup.select_one('a[data-tr="scheda_azienda__cta_sitoweb"]')

    if not website_button:
        logger.info(f"[PAGINEGIALLE] No website button found for {company_name}")
        return None

    web_url = website_button.get("href")
    if not (web_url and web_url.startswith("http") and "paginegialle" not in web_url):
        logger.info(f"[PAGINEGIALLE] No valid website found for {company_name}")
        return None

    logger.info(f"[PAGINEGIALLE] Found website for {company_name} -> {web_url}")
    return {
        "title": company_name,
        "url": web_url
    }


def extract_paginegialle_websites(pg_url: str, limit: int = 10) -> List[Dict[str, str]]:
    """
    PagineGialle Crawler. Steps:
    1. Scrapes the search result page.
    2. Collects the first N company profile links and their names.
    3. Visits each profile to find the 'Sito Web' (Website) button.
    4. Returns a list of dictionaries with company names and websites.
    """
    logger.info(f"[PAGINEGIALLE] Processing directory: {pg_url}")
    html = fetch_html(pg_url)

    try:
        profiles = _parse_profiles_from_directory(html, limit)
    except Exception as e:
        raise PagineGialleParseError(f"Failed to parse {pg_url}: {e}")

    logger.info(f"[PAGINEGIALLE] Found {len(profiles)} profiles to inspect...")

    websites = []
    for profile in profiles:
        result = _extract_website_from_profile(profile["url"], profile["name"])
        if result:
            websites.append(result)

    return websites
