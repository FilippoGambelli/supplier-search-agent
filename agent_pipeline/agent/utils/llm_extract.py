import json
from typing import Dict
from langchain_ollama import ChatOllama
from agent_pipeline.config import *
from stats import get_stats
from agent_pipeline.logger import logger

LLM = ChatOllama(
    base_url=OLLAMA_BASE_URL,
    model=MODEL,
    format="json",
    reasoning=False,
    temperature=0
)


def build_company_prompt(company: Dict) -> str:
    """
    Build extraction prompt for a single company.
    """

    title = company.get("title", "")
    url = company.get("url", "")

    homepage_text = company.get("homepage_text", "") or ""
    contact_text = company.get("contact_text", "") or ""

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
        generated_tokens = usage.get("output_tokens", 0)
        stats.add_request(input_tokens, generated_tokens)

        return json.loads(raw_output)

    except Exception as e:
        logger.error(f"[LLM EXTRACTION ERROR] {e}")
        return {
            "error": f"Invalid JSON returned by model or connection error: {str(e)}",
            "raw_output": raw_output if 'raw_output' in locals() else None
        }