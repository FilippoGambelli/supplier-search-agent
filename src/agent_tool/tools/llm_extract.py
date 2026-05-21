import json
import re
from typing import Dict
from langchain_ollama import ChatOllama
from logger import logger
from config import *
from stats import get_stats
# from agent_tool.agent.tools.db.db import save_supplier_to_db

LLM_EXTRACT = ChatOllama(
    base_url=OLLAMA_BASE_URL,
    model=AGENT_TOOL_MODEL,
    format="json",
    reasoning=False,
    temperature=0.1,
    timeout=300
)


def build_company_prompt(company: Dict) -> str:
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


def parse_json_response(raw_output: str) -> Dict:
    """Parse JSON from model response with multiple fallback strategies."""
    json_str = raw_output.strip()
    
    # Remove markdown code blocks
    if json_str.startswith("```json"):
        json_str = json_str[7:]
    elif json_str.startswith("```"):
        json_str = json_str[3:]
    
    if json_str.endswith("```"):
        json_str = json_str[:-3]
    
    json_str = json_str.strip()
    
    # Try to find JSON in the text
    try:
        return json.loads(json_str)
    except:
        pass
    
    # Try to extract JSON from anywhere in the text
    match = re.search(r'\{[\s\S]*\}', json_str)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
    
    # Return what we have with error
    return {"error": "Could not parse JSON", "raw": raw_output[:500]}


def extract_data(company: Dict) -> Dict:
    """Send the scraped content to the model and extract structured business information."""
    
    stats = get_stats()
    prompt = build_company_prompt(company)
    
    logger.info(f"[AGENT-TOOL] Starting data extraction from: {company.get('url', 'unknown')}")

    try:
        response = LLM_EXTRACT.invoke(prompt)

        # Update stats
        usage = response.usage_metadata or {}
        input_tokens = usage.get("input_tokens", 0)
        generated_tokens = usage.get("output_tokens", 0)
        stats.add_request(input_tokens, generated_tokens)

        raw_output = response.content if hasattr(response, 'content') else str(response)
        
        logger.info(f"[AGENT-TOOL] Data successfully extracted by LLM from {company.get('url', 'unknown')}")
        
        result = parse_json_response(raw_output)

        # fallback website
        if not result.get("website"):
            result["website"] = company.get("url", "")

        # fallback name
        if not result.get("name"):
            result["name"] = company.get("title", "Unknown")
        
        # Validate result has at least some data
        has_data = any([
            result.get("name"),
            result.get("email") and len(result.get("email", [])) > 0,
            result.get("phone") and len(result.get("phone", [])) > 0,
            result.get("locations") and len(result.get("locations", [])) > 0
        ])
        
        if not has_data and "error" not in result:
            logger.warning(f"[AGENT-TOOL] LLM did not extract any useful data")
            result["website"] = company.get("url", "")
            if not result.get("name"):
                result["name"] = company.get("title", "Unknown")
        
        return result

    except Exception as e:
        logger.error(f"[AGENT-TOOL] Error: {e}")
        return {
            "error": str(e),
            "name": company.get("title", "Unknown"),
            "website": company.get("url", "")
        }