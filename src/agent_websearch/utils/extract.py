import json
from typing import Dict
from langchain_ollama import ChatOllama
from logger import logger
from config import *
from stats import get_stats

LLM = ChatOllama(
    base_url=OLLAMA_BASE_URL,
    model=AGENT_TOOL_MODEL,
    format="json",
    reasoning=False,
    temperature=0,
    timeout=300
)


def build_company_prompt(company: Dict) -> str:
    """
    Build extraction prompt for a single company.
    """

    title = company.get("title", "")
    url = company.get("url", "")

    homepage_text = company.get("homepage_text", "") or ""
    contact_text = company.get("contact_text", "") or ""

    combined_text = f"""
HOMEPAGE:
{homepage_text}

CONTACT PAGE:
{contact_text}
"""

    prompt = f"""
You are an information extraction system.

Extract ONLY factual company information from the provided text.

Return ONLY valid JSON matching EXACTLY this schema:

{{
  "name": "string",
  "website": "string",
  "description": "string",

  "email": ["string"],
  "phone": ["string"],

  "vat_number": "string",

  "locations": [
    {{
      "country": "string",
      "city": "string",
      "address": "string"
    }}
  ]
}}

RULES:
- Output ONLY JSON
- No markdown
- No explanations
- No comments
- Use empty arrays [] when data is missing
- Use empty string "" for missing strings
- Do not invent data
- Extract ALL emails found
- Extract ALL phone numbers found
- VAT number may appear as:
  - VAT
  - Partita IVA
  - P.IVA
  - IVA
  - ITXXXXXXXXXXX
- Locations must contain:
  - country
  - city
  - full address if available
- If multiple locations exist, include all of them
- Description must be a short factual summary of the company

COMPANY WEBSITE:
{url}

COMPANY TITLE:
{title}

TEXT:
{combined_text}

JSON:
"""

    return prompt


def extract_data(company: Dict) -> Dict:
    """
    Send the scraped content to the model using ChatOllama
    and extract structured business information.
    """
    stats = get_stats()
    prompt = build_company_prompt(company)

    try:
        response = LLM.invoke(prompt)
        raw_output = response.content

        # Update stats
        usage = response.usage_metadata or {}
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        stats.add_request(input_tokens, output_tokens)

        return json.loads(raw_output)

    except Exception as e:
        logger.error(f"[LLM EXTRACTION ERROR] {e}")
        return {
            "error": f"Invalid JSON returned by model or connection error: {str(e)}",
            "raw_output": raw_output if 'raw_output' in locals() else None
        }