import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from sentence_transformers import SentenceTransformer

load_dotenv()

# SEARXNG CONFIG
SEARXNG_URL = os.environ.get("SEARXNG_URL")

# POSTGRESQL CONFIG
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg2://admin:admin@localhost:5432/suppliersearchagentdb"
)

MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# SEARCH LIMITS
SEARXNG_RESULTS_LIMIT = 5
PAGINEGIALLE_RESULTS_LIMIT = 2

# OLLAMA CONFIG
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL")

# LLM MODELS
AGENT_DBMANAGER_LLM = ChatOllama(
    base_url=OLLAMA_BASE_URL,
    model="gemma4:31b",
    reasoning=True,
    temperature=0,
    timeout=300
)

AGENT_ORCHESTRATOR_LLM = ChatOllama(
    base_url=OLLAMA_BASE_URL,
    model="gemma4:31b",
    reasoning=True,
    temperature=0,
    timeout=300
)

AGENT_WEBSEARCH_LLM = ChatOllama(
    base_url=OLLAMA_BASE_URL,
    model="gemma4:31b",
    reasoning=True,
    temperature=0,
    timeout=300
)

EXTRACT_LLM = ChatOllama(
    base_url=OLLAMA_BASE_URL,
    model="gemma4:31b",
    reasoning=True,
    temperature=0,
    timeout=300
)