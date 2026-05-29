import os
from langchain_ollama import ChatOllama

# SEARXNG CONFIG
SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://localhost:8080")

# SEARCH LIMITS
SEARXNG_RESULTS_LIMIT = 3
PAGINEGIALLE_RESULTS_LIMIT = 2

# OLLAMA CONFIG
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# LLM MODELS
AGENT_DBMANAGER_LLM = ChatOllama(
    base_url=OLLAMA_BASE_URL,
    model="gemma4:31b-cloud",
    reasoning=True,
    temperature=0,
    timeout=300
)

AGENT_ORCHESTRATOR_LLM = ChatOllama(
    base_url=OLLAMA_BASE_URL,
    model="gemma4:31b-cloud",
    reasoning=True,
    temperature=0,
    timeout=300
)

AGENT_WEBSEARCH_LLM = ChatOllama(
    base_url=OLLAMA_BASE_URL,
    model="gemma4:31b-cloud",
    reasoning=True,
    temperature=0,
    timeout=300
)

EXTRACT_LLM = ChatOllama(
    base_url=OLLAMA_BASE_URL,
    model="gemma4:31b-cloud",
    reasoning=True,
    temperature=0,
    timeout=300
)