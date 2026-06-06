import json
from typing import Dict
from logger import logger
from config import *
from stats import get_stats


def build_company_prompt(company: Dict) -> str:
    """
    Build extraction prompt for a single company.

    The goal is to extract structured supplier data that fully matches
    the database schema used by the ingestion pipeline.
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

  "category": ["string", "..."],

  "email": ["string", "..."],
  "phone": ["string", "..."],

  "vat_number": "string",

  "locations": [
    {{
      "country": "string",
      "region": "string",
      "province": "string",
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
- Do NOT hallucinate business data
- Use "" for missing strings
- Use [] for missing arrays
- Extract ALL emails found in text
- Extract ALL phone numbers found in text

CATEGORY RULES (IMPORTANT):
- Extract categories representing what the company DOES, SELLS, INSTALLS,
  PRODUCES, DISTRIBUTES, or SPECIALIZES IN
- Categories are used for semantic company search
- Output categories as short noun phrases
- Categories MUST be lowercase
- Categories MUST be generic and reusable
- Include multiple categories if relevant
- Prefer 2–10 categories
- Do NOT invent unsupported categories
- Do NOT include locations, company names, brands, slogans, certifications,
  legal forms, or marketing language
- Remove duplicates
- Prefer categories at different abstraction levels

Examples:
- plumber
- hydraulic systems
- construction
- masonry
- tile sales
- industrial automation
- electrical installation
- logistics
- packaging
- furniture manufacturing
- metalworking
- interior design
- solar energy
- software development
- mechanical engineering
- wholesale distribution

Good:
["construction", "masonry", "tile sales"]

Bad:
["best company", "quality", "professional team"]

VAT NUMBER RULES:
May appear as:
- VAT
- Partita IVA
- P.IVA
- IVA
- ITXXXXXXXXXXX

LOCATION RULES:
- Extract ALL available locations from the text
- Each location MUST include at least:
  - city if available
- address should be full structured address if available

GEOGRAPHIC INFERENCE RULE (IMPORTANT):
- If country, region, or province are NOT explicitly present in the text,
  infer them using general geographic knowledge
- Inference must remain conservative and realistic
- If city is known, infer:
  - correct country
  - correct region/state (if applicable)
  - correct province (if applicable)

- If multiple interpretations are possible, choose the MOST PROBABLE one

CRITICAL GEOGRAPHY NORMALIZATION RULE:
- All geographic names MUST be written in full form
- Do NOT use abbreviations

Examples:
- "USA" → "United States"
- "UK" → "United Kingdom"
- "UAE" → "United Arab Emirates"
- "NY" → "New York"
- "CA" → "California" or "Canada"

DESCRIPTION RULE:
- Short factual summary of the company
- Maximum 2 sentences
- Based ONLY on provided text

LANGUAGE RULES (IMPORTANT):
- All extracted textual values inside the JSON MUST be written in the company's native language whenever it can be determined from the source text or company location

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
        response = EXTRACT_LLM.invoke(prompt)
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