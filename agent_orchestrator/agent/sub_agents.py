import json
from logger import logger   
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import tool

@tool("run_search_agent", description="Searches the web for external data and new suppliers.")
def run_search_agent(query: str) -> str:
    """
    Execute the agent_tool to search for suppliers on the web.
    Use this tool when you need to search for suppliers.
    
    Args:
        query: A detailed search query specifying the criteria for finding suppliers.

    Returns:
        JSON string containing the found suppliers or data.
    """
    try:
        # Dynamic import to avoid circular dependencies if agents share modules
        from agent_tool.agent.agent import run_agent
        
        logger.info(f"[ORCHESTRATOR] Delegating to agent_tool with query: {query}")
        result, error = run_agent(query)
        logger.info(f"[ORCHESTRATOR] Result from agent_tool: {result}, Error: {error}")

        if error:
            logger.error(f"[ORCHESTRATOR] agent_tool returned an error: {error}")
            return f"Error in agent_tool execution: {error}"

        return result
    except Exception as e:
        logger.error(f"[ORCHESTRATOR] Exception in agent_tool: {e}")
        return f"Tool exception: {str(e)}"


@tool("run_dbmanager_agent", description="Performs database operations like retrieving existing suppliers or any database operations.")
def run_dbmanager_agent(query: str) -> str:
    """
    Execute the dbmanager agent for database operations.
    Use this tool to save newly found data, retrieve existing suppliers and perform any database operations.
    
    Args:
        query: Clear instructions for the dbmanager (e.g., "Retrieve all suppliers located in Pisa" or "Save these suppliers: <json>").

    Returns:
        String confirming the database operation result or the fetched data.
    """
    try:
        from agent_dbmanager.agent.agent import run_dbmanager
        
        logger.info(f"[ORCHESTRATOR] Delegating to agent_dbmanager with query: {query}")
        result, error = run_dbmanager(query)

        logger.info(f"[ORCHESTRATOR] Result from agent_dbmanager: {result}, Error: {error}")

        if error:
            logger.error(f"[ORCHESTRATOR] agent_dbmanager returned an error: {error}")
            return f"Error in agent_dbmanager execution: {error}"

        return result
    except Exception as e:
        logger.error(f"[ORCHESTRATOR] Exception in agent_dbmanager: {e}")
        return f"Tool exception: {str(e)}"