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
- Retrieves previously discovered suppliers from the database
- Stores newly discovered suppliers for future searches
- Returns a consolidated list of supplier candidates

This significantly reduces manual supplier research effort and accelerates quotation and procurement workflows.

---

# Architecture

The platform is implemented as a **multi-agent system** designed to automate supplier discovery and supplier data management.

The architecture is composed of:

- **LangGraph** for multi-agent orchestration and execution graphs
- **LangChain** for LLM and agent abstractions
- **1 Orchestrator Agent**
- **1 Web Searcher Agent**
- **1 Database Manager Agent**
- **SearXNG** for web search aggregation
- **PostgreSQL + pgvector** for persistent and semantic data retrieval
- **Ollama** as the LLM provider

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

The Web Searcher is responsible for supplier discovery from external sources.

It transforms the user request into targeted web searches and performs all extraction and processing activities required to identify relevant suppliers.

Main responsibilities:

- Performing supplier searches
- Collecting supplier websites
- Extracting company information
- Retrieving contacts
- Identifying business categories
- Gathering company locations
- Structuring extracted data

The platform supports **two alternative execution strategies** for web search and information extraction.

---

### Option 1 - Agentic Web Search (Default in Current Architecture)

This is the default mode used inside the multi-agent system.

In this configuration, the **Orchestrator dynamically coordinates the Web Searcher Agent**, deciding how searches, extraction, validation, and persistence should be executed.

This approach provides:

- Adaptive decision making
- Flexible search strategies
- Dynamic execution flows
- Better handling of heterogeneous search results
- Improved extensibility for future agents and capabilities

The agent autonomously decides how to explore and process web information to maximize supplier retrieval quality.

---

### Option 2 - Deterministic Search Pipeline

As an alternative, the platform also supports a fully deterministic execution pipeline.

In this mode, supplier discovery follows a predefined sequence of steps instead of relying on agent-based reasoning:

1. Generate search queries
2. Execute web searches
3. Collect result pages
4. Extract company information
5. Normalize and structure data
6. Store results in the database

This approach provides:

- Predictable execution
- Easier debugging
- Reproducible outputs
- Lower orchestration complexity

Although not used by default, this pipeline remains available for experimentation or controlled deployments.

---

### Web Search Engine - SearXNG

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

Each discovered company is stored with:

- Company details
- Contact information
- Categories
- Descriptions
- Locations
- Additional metadata useful for procurement workflows

---

## Database - PostgreSQL + pgvector

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

Ollama abstracts model execution and enables flexible deployment options.

This allows the system to operate with:

- Local models running on local hardware
- Cloud-hosted models
- Easy switching between different LLMs

Current configuration:

```text
gemma4:31b-cloud
````

Ollama keeps the architecture model-agnostic and highly flexible depending on infrastructure constraints.

---

# Using the System

The Supplier Search Agent provides two main ways to interact with the system depending on the desired level of control and observability.

---

## Option 1 - Command Line Interface (CLI)

The primary interface is a **Command Line Interface (CLI)**.

Start the application from the project root:

```bash
python src/main.py
```

The CLI allows direct interaction with the system and supports multiple execution modes.

### Full Multi-Agent System (Default)

This is the recommended mode.

User requests are processed by the **Orchestrator**, which coordinates:

* Web Searcher Agent
* Database Manager Agent

It combines:

* historical supplier data from PostgreSQL
* newly discovered suppliers from the web

This is the full system experience.

---

### Web Search Only - Agentic Mode

This mode runs only the **agentic Web Searcher** without full orchestration.

The agent autonomously:

* decides search strategy
* processes results
* extracts supplier information

Useful for focused supplier discovery tasks.

---

### Web Search Only - Deterministic Pipeline

This mode executes a predefined search pipeline without agent reasoning.

Useful for:

* reproducibility
* debugging
* controlled execution

---

## Option 2 - LangGraph Development Interface

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

1. Open the project in VS Code
2. Click **"Reopen in Container"**
   or use:

   ```bash
   Ctrl + Shift + P -> Dev Containers: Reopen in Container
   ```
3. Wait for container build
4. Run:

```bash
python src/main.py
```

---

# Ollama Model Setup (First Run Only)

This project uses a cloud-hosted model:

```text
gemma4:31b-cloud
```

### Setup

Open a shell inside the Ollama container:

```bash
docker exec -it ssa-ollama sh
```

Pull the model:

```bash
ollama pull gemma4:31b-cloud
```

Run it to trigger authentication:

```bash
ollama run gemma4:31b-cloud
```

**Open the browser link and complete authentication.**

---

# Developer Notes (Internal)

## Connect PostgreSQL from pgAdmin

Register a new server:

```text
Host: postgres
Port: 5432
Database: suppliersearchagentdb
Username: admin
Password: admin
```

---

## Reset Database

To completely reset the database:

```bash
docker compose down
docker volume rm supplier-search-agent_postgres_data
```