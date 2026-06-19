class WebSearchError(Exception):
    """Base exception for the agent_websearch module."""


class SearchError(WebSearchError):
    """Error during web search."""


class SearchTimeoutError(SearchError):
    """Search request timed out."""


class SearchConnectionError(SearchError):
    """Cannot connect to the search instance."""


class ScrapeError(WebSearchError):
    """Error during website scraping."""


class RobotsBlockedError(ScrapeError):
    """robots.txt disallows scraping."""


class PagineGialleParseError(WebSearchError):
    """Error parsing PagineGialle pages."""


class ExtractionError(WebSearchError):
    """Error during LLM-based data extraction."""


class InsufficientDataError(WebSearchError):
    """Extracted data lacks required fields (email/phone)."""

class NotACompanyError(WebSearchError):
    """Not a company"""
