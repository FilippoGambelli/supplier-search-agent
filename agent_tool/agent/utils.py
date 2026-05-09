import json
import re
from typing import Any, Tuple, Optional

def extract_and_parse_json(raw_text: str) -> Tuple[Optional[Any], Optional[str]]:
    """
    Extracts a JSON payload from a text string (e.g., LLM output)
    by removing any Markdown formatting (```json ... ```).

    Args:
        raw_text (str): The raw string output from the LLM.

    Returns:
        Tuple[Optional[Any], Optional[str]]: A tuple containing:
            - The parsed data (list or dictionary) if parsing succeeds, otherwise None.
            - An error message string if parsing fails, otherwise None.
    """
    if not raw_text:
        return None, "Input string is empty or null."

    # 1. Remove leading and trailing whitespace
    cleaned_text = raw_text.strip()

    # 2. Search for a Markdown code block and capture only the inner content
    match = re.search(r'```(?:json)?(.*?)```', cleaned_text, re.DOTALL | re.IGNORECASE)
    if match:
        cleaned_text = match.group(1).strip()

    # 3. Attempt to parse the JSON
    try:
        parsed_data = json.loads(cleaned_text)
        return parsed_data, None
    except json.JSONDecodeError as e:
        return None, f"JSON Decode Error: {str(e)}"