from langchain_core.tools import tool
import json
from logger import logger
from agent_dbmanager.db.db_manager import save_supplier_to_db, execute_search_query
from agent_dbmanager.exceptions import (
    DbManagerError, DatabaseError, IntegrityError,
    ArtifactError, ValidationError
)
from artifact_store import artifact_store

VERBOSE = False


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
            supplier_name = item.get("name", "Unknown")
            try:
                save_supplier_to_db(item)
                saved_count += 1
            except IntegrityError as e:
                error_count += 1
                errors.append(f"Supplier '{supplier_name}': Integrity constraint violation: {str(e)}")
                logger.error(f"[DBMANAGER TOOL] Integrity error saving supplier '{supplier_name}': {e}")
            except DatabaseError as e:
                error_count += 1
                errors.append(f"Supplier '{supplier_name}': Database error: {str(e)}")
                logger.error(f"[DBMANAGER TOOL] Database error saving supplier '{supplier_name}': {e}")
            except Exception as e:
                error_count += 1
                errors.append(f"Supplier '{supplier_name}': Unexpected error: {str(e)}")
                logger.error(f"[DBMANAGER TOOL] Save error for supplier '{supplier_name}': {e}")

        logger.info(f"[DBMANAGER TOOL] save_suppliers completed - saved: {saved_count}, errors: {error_count}")

        if error_count == 0:
            result = f"SUCCESS: {saved_count} suppliers saved to database."
        else:
            result = f"PARTIAL SUCCESS: {saved_count} saved, {error_count} failed. Failed suppliers: {errors}"

    except json.JSONDecodeError as e:
        logger.error(f"[DBMANAGER TOOL] Invalid JSON in artifact: {e}")
        result = "ERROR: Artifact content is not valid JSON."

    except DbManagerError as e:
        logger.error(f"[DBMANAGER TOOL] {e}")
        result = f"ERROR: {str(e)}"

    except Exception as e:
        logger.error(f"[DBMANAGER TOOL] Unexpected error: {e}")
        result = f"ERROR: {str(e)}"

    if VERBOSE:
        print(f"\n[TOOLS] save_suppliers\n  Arguments: artifact_id=\"{artifact_id}\"\n  Result: {result}")
    return result


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
            raise ValidationError("at least one location filter must be provided")

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

        result = artifact_id

    except ArtifactError as e:
        logger.error(f"[DBMANAGER TOOL] Artifact error during search: {e}")
        result = f"ERROR: Artifact store error: {str(e)}"
    except DatabaseError as e:
        logger.error(f"[DBMANAGER TOOL] Database error during search: {e}")
        result = f"ERROR: Database query failed: {str(e)}"
    except DbManagerError as e:
        logger.error(f"[DBMANAGER TOOL] {e}")
        result = f"ERROR: {str(e)}"
    except Exception as e:
        logger.error(f"[DBMANAGER TOOL] Unexpected error during search: {e}")
        result = f"ERROR: {str(e)}"

    if VERBOSE:
        args_str = json.dumps({k: v for k, v in {"country": country, "region": region, "province": province, "city": city, "semantic_query": semantic_query}.items() if v is not None}, ensure_ascii=False)
        print(f"\n[TOOLS] semantic_search_suppliers\n  Arguments: {args_str}\n  Result: {result}")
    return result