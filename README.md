# **Supplier Search Agent**

AI-powered supplier discovery and contact extraction platform for construction and procurement workflows.

The Supplier Search Agent is a multi-agent system designed to automate the supplier sourcing process for companies looking for products, materials, or services.

Instead of manually searching across multiple websites and directories, users can provide a simple natural language request describing what they need, and the system automatically discovers relevant suppliers, collects business information, and stores the results for future reuse.

> Powered by **LangChain**, **LangGraph**, **Ollama**, **SearXNG**, and **PostgreSQL + pgvector**.

The platform is built on top of **LangChain** and **LangGraph** to support structured LLM workflows, tool execution, and multi-agent orchestration.

- **LangChain** provides the abstraction layer for LLM interactions, tool execution, prompt management, and agent composition.
- **LangGraph** manages stateful execution and enables graph-based multi-agent orchestration, execution control, and observability.

This architecture enables the system to coordinate multiple specialized agents while maintaining full execution state and traceability.

---

# Overview

The system is designed to help companies identify and contact suppliers more efficiently.

Users submit a textual query describing their procurement needs (for example: *"Find steel beam suppliers in Northern Italy"* or *"Search for companies providing industrial HVAC systems"*).

The platform then:

- Searches for relevant supplier companies
- Extracts company information and contact details
- Stores newly discovered suppliers and updates existing ones for future searches
- Retrieves suppliers from the database
- Returns a consolidated list of supplier candidates

This significantly reduces manual supplier research effort and accelerates quotation and procurement workflows.

---

# Architecture

The platform is implemented as a **multi-agent system** designed to automate supplier discovery and supplier data management.

The architecture is composed of:

- **Orchestrator Agent**
- **Web Searcher Agent**
- **Database Manager Agent**

The overall objective is to combine **live supplier discovery** with **persistent knowledge accumulation**, allowing the system to become increasingly valuable over time.

---

## Orchestrator

The Orchestrator acts as the central coordinator of the system.

It receives the user request in natural language and decides how to execute the supplier search workflow.

Its responsibilities include:

- Understanding procurement requirements
- Coordinating the execution of sub-agents
- Combining results from multiple sources
- Managing data retrieval and persistence
- Returning consolidated supplier results

When a request is received, the orchestrator simultaneously leverages:

- historical supplier information already stored in the database
- newly discovered suppliers collected from the web

This allows users to benefit from both past knowledge and fresh market data.

---

## Web Searcher Agent

The Web Searcher is responsible for supplier discovery from the web.

It transforms the user request into targeted web searches and performs all extraction and processing activities required to identify relevant suppliers.

Main responsibilities:

- Performing supplier searches
- Collecting supplier websites
- Extracting company information
- Identifying business categories
- Gathering company locations
- Structuring extracted data

The platform supports **two alternative execution strategies**:

- **Option 1 - Agentic Web Search** — A dedicated web search agent autonomously searches, validates, filters, and extracts supplier data using its own tools, maximizing retrieval quality.
- **Option 2 - Deterministic Pipeline** — A fixed sequence of predefined steps (search, scrape, extract, and structure data) without agent reasoning. Useful for controlled and reproducible runs.

#### Web Search Engine - SearXNG

Both execution modes rely on **SearXNG** as the web search provider.

SearXNG is an open-source metasearch engine that aggregates results from multiple search providers instead of relying on a single search engine.

Using SearXNG allows the system to:

- Aggregate results from multiple sources
- Improve supplier coverage
- Reduce vendor lock-in
- Increase flexibility in search strategies

The Web Searcher uses SearXNG as the entry point for supplier discovery and then processes the retrieved pages to extract structured supplier data.

---

## Database Manager Agent

The Database Manager handles all persistence and retrieval operations.

Its purpose is to maintain a continuously growing supplier knowledge base.

Main responsibilities:

- Storing newly discovered suppliers
- Retrieving historical suppliers
- Updating existing records
- Executing structured and semantic queries
- Maintaining supplier metadata

#### Database - PostgreSQL + pgvector

The system uses **PostgreSQL** as its primary database.

PostgreSQL provides reliable relational storage for supplier information and application data.

Additionally, it integrates the **pgvector** extension.

pgvector enables vector storage and similarity search directly inside PostgreSQL, allowing the system to perform **semantic retrieval**.

This enables:

- Semantic supplier search
- Similarity matching between queries
- Retrieval of relevant historical suppliers
- Improved reuse of previously collected supplier data

By combining relational storage with vector search, the platform supports both structured filtering and AI-powered retrieval.

---

## LLM Provider - Ollama

The platform uses **Ollama** as the Large Language Model (LLM) provider.

Ollama is executed locally, on the same machine as the application, providing fast inference without external API dependencies.

Current configuration:

```text
Model: qwen3.6:35b
Hardware: NVIDIA GH200 480GB
CUDA: 12.3
Ollama: v0.24.0
Performance: ~100 tk/s
```

Ollama keeps the architecture model-agnostic and highly flexible depending on infrastructure constraints.

---

# Using the System

The Supplier Search Agent provides two main ways to interact with the system depending on the desired level of control and observability.

### Option 1 - Command Line Interface (CLI)

The primary interface is a **Command Line Interface (CLI)**.

Start the application from the project root:

```bash
python src/main.py
```

The CLI allows direct interaction with the system and supports multiple execution modes selectable at runtime with `/mode <number>`:

1. **WebSearch + Store** *(default)* — Full multi-agent orchestration. Coordinates the Web Searcher Agent and Database Manager Agent to discover new suppliers and store them for future searches.
2. **Agentic WebSearch** — Runs the agentic Web Searcher alone. Autonomously searches and extracts supplier data without database persistence.
3. **Deterministic WebSearch** — Executes a deterministic search pipeline. Follows a fixed sequence of search, scrape, and extract steps. Useful for reproducibility and debugging.

### Option 2 - LangGraph Development Interface

For development and debugging, the system can be run using **LangGraph Dev**:

```bash
langgraph dev
```

This launches the **LangSmith web interface**, providing deep visibility into execution.

It allows:

* Visual graph execution
* Step-by-step workflow tracing
* LLM input/output inspection
* Tool execution monitoring
* State inspection
* Full orchestration debugging

This mode is primarily used for development, testing, and system analysis.

---

# Quick Start

1. Clone the repository:

```bash
git clone <url>
cd supplier-search-agent
```

2. Create a Python virtual environment and install dependencies:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

3. Start the required Docker services (SearXNG, PostgreSQL, pgAdmin):

```bash
docker compose up -d
```

4. Pull the LLM model and start Ollama:

```bash
ollama pull qwen3.6:35b
ollama serve
```

5. Run the application:

```bash
python src/main.py
```

### Environment Configuration

Create a `.env` file in the project root with the following variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `LANGSMITH_TRACING` | Yes | Set to `true` or `false` to enable/disable LangSmith tracing |
| `LANGSMITH_API_KEY` | Yes* | Your LangSmith API key |
| `LANGCHAIN_PROJECT` | Yes | Project name, set to `supplier-search-agent` |
| `LANGCHAIN_ENDPOINT` | Yes* | LangSmith endpoint URL. See [LangChain documentation](https://docs.langchain.com/oss/python/langchain/studio) |
| `OLLAMA_BASE_URL` | Yes | Ollama server URL (e.g. `http://localhost:11434`) |
| `HF_TOKEN` | No | HuggingFace token (optional, for gated models) |

*\*Required only if `LANGSMITH_TRACING=true`.*