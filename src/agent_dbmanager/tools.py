from typing import Any, Dict, List, Union

from langchain_core.tools import tool
import json
from logger import logger

# Import the updated db functions
from agent_dbmanager.db.db_manager import save_supplier_to_db, execute_search_query

@tool
def save_suppliers(suppliers_data: Union[List[Dict[str, Any]], Dict[str, Any]]) -> str:
    """
    Save a list of suppliers to the database.
    
    Args:
        suppliers_data: List of supplier objects or a single supplier object to save.
    """
    try:
        data = suppliers_data
        
        # Handle both array and single object for robustness
        if not isinstance(data, list):
            data = [data]

        saved_ids = []
        errors = []
        
        for item in data:
            try:
                sup_id = save_supplier_to_db(item)
                saved_ids.append(sup_id)
            except Exception as e:
                errors.append(str(e))
        
        logger.info(f"[DBMANAGER TOOL] Saved {len(saved_ids)} suppliers, {len(errors)} errors.")

        return json.dumps({
            "status": "success" if not errors else "partial_success",
            "saved_count": len(saved_ids),
            "error_count": len(errors)
        })
    
    except json.JSONDecodeError:
        return json.dumps({"status": "error", "message": "The data format is not a valid JSON."})
    except Exception as e:
        logger.error(f"[DBMANAGER TOOL] Save error: {e}")
        return json.dumps({"status": "error", "message": f"System error: {str(e)}"})

@tool
def semantic_search_suppliers(location: str) -> str:
    """
    Search for suppliers filtering by location.
    
    Args:
        location: Place or city to search in (e.g., 'Pisa'). Leave empty if not specified.
    """
    try:
        # The tool delegates SQL query construction and execution to the DB layer
        results = execute_search_query(location)
        return json.dumps(results, ensure_ascii=False, indent=4)
    except Exception as e:
        logger.error(f"[DBMANAGER TOOL] Semantic search error: {e}")
        return json.dumps({"status": "error", "message": f"Error during database search: {str(e)}"})