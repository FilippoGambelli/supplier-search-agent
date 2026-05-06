import json
import requests
from typing import Dict, List

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma4:31b-cloud"


def build_company_prompt(company: Dict) -> str:
    """
    Build extraction prompt for a single company.
    """

    title = company.get("title", "")
    url = company.get("url", "")

    homepage_text = company.get("homepage_text", "") or ""
    contact_text = company.get("contact_text", "") or ""

    # Limit text size to avoid LLM overload
    homepage_text = homepage_text[:12000]
    contact_text = contact_text[:8000]

    prompt = f"""
You are an expert business information extraction system.

Your task is to extract company contact information
from the provided website content.

IMPORTANT RULES:
- Return ONLY valid JSON
- Do NOT explain anything
- Do NOT use markdown
- If information is missing, use null
- Emails must be returned as a list
- Phone numbers must be returned as a list
- Be precise and conservative
- Extract only information clearly present in the text

TARGET WEBSITE:
{url}

COMPANY TITLE:
{title}

HOMEPAGE CONTENT:
{homepage_text}

CONTACT PAGE CONTENT:
{contact_text}

Extract the following information:

{{
  "company_name": string | null,
  "website": string | null,
  "emails": [string],
  "phone_numbers": [string],
  "vat_number": string | null,
  "address": string | null,
  "city": string | null,
  "postal_code": string | null,
  "country": string | null,
  "description": string | null,
  "contact_person": string | null
}}

Return ONLY valid JSON.
"""

    return prompt


def extract_company_data(company: Dict) -> Dict:
    """
    Send company scraped content to Ollama
    and extract structured business information.
    """

    prompt = build_company_prompt(company)

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        },
        timeout=120
    )

    response.raise_for_status()

    raw_output = response.json()["response"]

    try:
        return json.loads(raw_output)

    except Exception:
        return {
            "error": "Invalid JSON returned by model",
            "raw_output": raw_output
        }