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
def semantic_search_suppliers(location: str) -> str:
    """
    Search suppliers in the database filtered by location.

    This tool:
    - Executes a database search using the DB engine
    - Stores results in an artifact
    - Returns only the artifact_id as a string

    Args:
        location:
            Geographic filter (e.g. city, region).
            Can be empty for broader search.
    """

    try:
        results = execute_search_query(location)

        artifact_id = artifact_store.save(
            data=json.dumps(results, ensure_ascii=False, indent=4),
            meta={
                "type": "db_search",
                "location": location
            }
        )

        return artifact_id

    except Exception as e:
        logger.error(f"[DBMANAGER TOOL] Semantic search error: {e}")
        return f"ERROR: {str(e)}"