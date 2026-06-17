from urllib.parse import urlparse
import re

COMPANY_WORDS = {
    "spa",
    "srl",
    "ltd",
    "inc",
    "group",
    "company"
}


def normalize_website(url):
    """
    Normalize a website URL by extracting only the core domain name
    (without protocol, subdomains like www, or TLD extensions).

    Example:
        https://www.example.com -> example
    """

    if not url:
        return None

    url = url.lower().strip()

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    host = urlparse(url).hostname or ""

    host = host.removeprefix("www.")

    parts = host.split(".")

    if len(parts) >= 2:
        return parts[-2]

    return host


def normalize_emails(emails):
    """
    Normalize a list of email addresses by:
    - converting to lowercase
    - stripping whitespace
    - removing duplicates

    Returns a unique list of normalized emails.
    """

    if not emails:
        return []

    return list({e.strip().lower() for e in emails if e})


def normalize_phones(phones):
    """
    Normalize phone numbers by:
    - removing all non-numeric characters
    - removing duplicates

    Returns a unique list of cleaned phone numbers.
    """

    if not phones:
        return []

    normalized = []

    for p in phones:

        p = re.sub(r"\D", "", p)

        if p:
            normalized.append(p)

    return list(set(normalized))


def normalize_name(name):
    """
    Normalize a company name by:
    - converting to lowercase
    - removing punctuation
    - removing common company suffix words (e.g. srl, spa, ltd)
    - producing a simplified canonical name for matching purposes
    """

    if not name:
        return None

    name = name.lower()

    name = re.sub(r"[^\w\s]", "", name)

    words = [
        w
        for w in name.split()
        if w not in COMPANY_WORDS
    ]

    return " ".join(words)

def normalize_categories(categories):
    result = []
    for c in categories or []:
        c = c.strip().lower()
        if c:
            result.append(c)
    return list(set(result))

def build_supplier_embedding_text(description, categories):
    return f"""
description:
{description}

categories:
{" | ".join(categories)}
""".strip()