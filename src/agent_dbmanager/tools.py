from langchain_core.tools import tool
import json
from logger import logger
from agent_dbmanager.db.db_manager import save_supplier_to_db, execute_search_query
from artifact_store import artifact_store


@tool
def save_suppliers(artifact_id: str) -> str:
    """
    Save suppliers into the database using data loaded from an artifact.

    This tool:
    - Loads supplier data from an artifact
    - Parses JSON content
    - Persists each supplier into the database
    - Returns a human-readable status string

    IMPORTANT:
    - Input MUST be an artifact_id
    - The artifact MUST contain JSON data (list or single object)
    """

    try:
        raw_data = artifact_store.load(artifact_id)
        data = json.loads(raw_data) if isinstance(raw_data, str) else raw_data

        if not isinstance(data, list):
            data = [data]

        saved_count = 0
        error_count = 0
        errors = []

        for item in data:
            try:
                save_supplier_to_db(item)
                saved_count += 1
            except Exception as e:
                error_count += 1
                errors.append(str(e))
                logger.error(f"[DBMANAGER TOOL] Save error: {e}")

        logger.info(f"[DBMANAGER TOOL] save_suppliers completed - saved: {saved_count}, errors: {error_count}")

        if error_count == 0:
            return f"SUCCESS: {saved_count} suppliers saved to database."

        return (f"PARTIAL SUCCESS: {saved_count} saved, {error_count} failed. Errors: {errors}")

    except json.JSONDecodeError:
        logger.error("[DBMANAGER TOOL] Invalid JSON in artifact")
        return "ERROR: Artifact content is not valid JSON."

    except Exception as e:
        logger.error(f"[DBMANAGER TOOL] Unexpected error: {e}")
        return f"ERROR: {str(e)}"


@tool
def semantic_search_suppliers(country: str = None, region: str = None, province: str = None, city: str = None, semantic_query: str = None) -> str:
    """
    Search suppliers in the database using hierarchical geographic filters and optional semantic search.

    This tool:
    - Executes a database search using the DB engine
    - Filters suppliers by geographic fields (country, region, province, city)
    - Optionally ranks results using semantic similarity via embedding search (semantic_query)
    - Stores results in an artifact
    - Returns only the artifact_id as a string

    Args:
        country:
            Optional country filter (e.g. "Italy").

        region:
            Optional region/state filter (e.g. "Lombardy").

        province:
            Optional province filter (e.g. "Milan").

        city:
            Optional city filter (e.g. "Milan").

        semantic_query:
            Optional natural language query used for semantic search over supplier embeddings.
            If provided, results are ranked by similarity between the query embedding
            and supplier embedding (vector-based search).

            Examples:
                - "suppliers of concrete and cement for infrastructure projects"
                - "construction material distributors for residential buildings"
                - "companies providing steel reinforcement bars for civil engineering"
                - "building material suppliers specialized in sustainable construction"
                - "suppliers of insulation materials for energy-efficient buildings"

    Requirements:
        At least ONE of the filters must be provided:
        country OR region OR province OR city OR semantic_query

    Returns:
        artifact_id (str): reference to stored search results

    Raises:
        Exception: if database query or artifact storage fails
    """

    try:
        # Validate that at least one filter is provided
        if not any([country, region, province, city, semantic_query]):
            return "ERROR: at least one location filter must be provided"

        results = execute_search_query(
            country=country,
            region=region,
            province=province,
            city=city,
            semantic_query=semantic_query
        )

        artifact_id = artifact_store.save(
            data=json.dumps(results, ensure_ascii=False, indent=4)
        )

        return artifact_id

    except Exception as e:
        logger.error(f"[DBMANAGER TOOL] Semantic search error: {e}")
        return f"ERROR: {str(e)}"