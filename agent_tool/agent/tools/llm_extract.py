import json
import re
import time
from typing import Dict
from langchain_ollama import ChatOllama
from agent_tool.logger import logger
from agent_tool.config import *
from stats import get_stats

LLM_EXTRACT = ChatOllama(
    base_url=OLLAMA_BASE_URL,
    model=MODEL,
    format="json",
    reasoning=False,
    temperature=0.1
)


def build_company_prompt(company: Dict) -> str:
    """Build extraction prompt for a single company."""
    title = company.get("title", "")
    url = company.get("url", "")

    homepage_text = company.get("homepage_text", "") or ""
    contact_text = company.get("contact_text", "") or ""

    combined_text = f"HOMEPAGE:\n{homepage_text}\n\nCONTACT PAGE:\n{contact_text}"

    prompt = f"""Extract company information from the text below. 
Look for: company name, website, email addresses, phone numbers, VAT number (partita IVA), address, city, postal code.

Website: {url}
Title: {title}

Text:
{combined_text}

Search carefully for ANY email addresses (patterns like @ and .com, .it)
Search for ANY phone numbers (patterns like +39, 0XX, XXX-XXXXXXX)
Search for ANY VAT numbers (patterns like ITXXXXXXXXX)
Search for addresses with street names and city names

Output ONLY valid JSON, no explanation. Use empty array [] for missing lists, null for missing strings.

JSON:"""

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
    
    logger.info(f"[EXTRACT] Starting data extraction from: {company.get('url', 'unknown')}")

    try:
        response = LLM_EXTRACT.invoke(prompt)

        # Update stats
        usage = response.usage_metadata or {}
        prompt_tokens = usage.get("input_tokens", 0)
        generated_tokens = usage.get("output_tokens", 0)
        stats.add_request(prompt_tokens, generated_tokens)

        raw_output = response.content if hasattr(response, 'content') else str(response)
        
        logger.info(f"[EXTRACT] Data successfully extracted by LLM from {company.get('url', 'unknown')}")
        
        result = parse_json_response(raw_output)
        
        # Validate result has at least some data
        has_data = any([
            result.get("company_name"),
            result.get("emails") and len(result.get("emails", [])) > 0,
            result.get("phone_numbers") and len(result.get("phone_numbers", [])) > 0,
            result.get("address")
        ])
        
        if not has_data and "error" not in result:
            logger.warning("[EXTRACT_DATA] Model returned empty result, adding URL as fallback")
            result["website"] = company.get("url", "")
            if not result.get("company_name"):
                result["company_name"] = company.get("title", "Unknown")
        
        return result

    except Exception as e:
        logger.error(f"[EXTRACT_DATA] Error: {e}")
        return {
            "error": str(e),
            "company_name": company.get("title", "Unknown"),
            "website": company.get("url", "")
        }