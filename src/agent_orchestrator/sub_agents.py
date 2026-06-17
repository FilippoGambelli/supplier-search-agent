from logger import logger
from langchain_core.tools import tool
from artifact_store import artifact_store
from agent_orchestrator.exceptions import (
    OrchestratorError, ArtifactError, SearchAgentError, DbManagerAgentError
)

VERBOSE = False


@tool("run_search_agent")
def run_search_agent(query: str) -> str:
    """
    Execute the web-search subagent to discover suppliers and collect structured company data.

    This tool acts as an orchestration layer and does not perform web searches directly.
    Instead, it forwards the provided search query to the dedicated `agent_websearch`
    subagent, which is responsible for:

    - Executing web searches using the supplied criteria.
    - Identifying relevant supplier companies.
    - Extracting structured information for each company.
    - Normalizing results into a machine-readable JSON format.

    The extracted supplier information may include (when available):
    - Company name
    - Business description
    - Contact information
    - Headquarters and office locations
    - Website and external references
    - Supplier metadata and other discovered attributes

    Result handling:
    The structured JSON is NOT returned directly by this tool.
    Instead, the output is persisted into the artifact store and the tool returns
    only the generated `artifact_id`.

    This design is intentional and provides several benefits:
    - Avoids transferring potentially large JSON payloads between agents.
    - Keeps orchestrator responses lightweight.
    - Enables downstream agents to retrieve, reuse, and process results independently.
    - Preserves search outputs for traceability and auditing.
    - Decouples execution from data consumption.

    Args:
        query:
            A supplier search query describing what type of suppliers should be
            searched and, optionally, the target location or business constraints.

            Example:
            - "fornitori materiali edili Milano"
            - "fornitori piastrelle Torino"
            - "fornitori acciaio Emilia-Romagna"

    Returns:
        str:
            The identifier (`artifact_id`) of the stored search result artifact.
            The artifact contains the full structured JSON produced by the web-search subagent.

            Example:
            "vYtM84r9uP2E9zXyB6Hn9c"

    Raises:
        Does not raise exceptions directly.
        Execution failures are captured and returned as error strings:
        - "Error in agent_tool execution: ..."
        - "Tool exception: ..."
    """
    try:
        # Dynamic import to avoid circular dependencies if agents share modules
        from agent_websearch.agent_tool import run_agent

        logger.info(f"[ORCHESTRATOR] Delegating to agent_tool with query: {query}")

        if VERBOSE:
            print(f"\n[TOOLS] run_search_agent\n  Arguments: query=\"{query}\"")

        result, error = run_agent(query, verbose=VERBOSE)
        logger.info(f"[ORCHESTRATOR] Result from agent_tool: {result}, Error: {error}")

        if error:
            raise SearchAgentError(f"agent_tool execution failed: {error}")

        artifact_id = artifact_store.save(
            data=result
        )

        if VERBOSE:
            print(f"  Result: {artifact_id}")

        return artifact_id

    except SearchAgentError as e:
        logger.error(f"[ORCHESTRATOR] {e}")
        result_str = f"Error in agent_tool execution: {e}"
        if VERBOSE:
            print(f"  Result: {result_str}")
        return result_str
    except ArtifactError as e:
        logger.error(f"[ORCHESTRATOR] Artifact error in agent_tool: {e}")
        result_str = f"Error in agent_tool execution: {e}"
        if VERBOSE:
            print(f"  Result: {result_str}")
        return result_str
    except OrchestratorError as e:
        logger.error(f"[ORCHESTRATOR] {e}")
        result_str = str(e)
        if VERBOSE:
            print(f"  Result: {result_str}")
        return result_str
    except Exception as e:
        logger.error(f"[ORCHESTRATOR] Unexpected exception in agent_tool: {e}")
        result_str = f"Tool exception: {str(e)}"
        if VERBOSE:
            print(f"  Result: {result_str}")
        return result_str


@tool("run_dbmanager_agent")
def run_dbmanager_agent(query: str) -> str:
    """
    Execute the database manager subagent to perform persistent database operations.

    This tool acts as an orchestration layer and does not access the database directly.
    Instead, it forwards instructions to the dedicated `agent_dbmanager` subagent,
    which is responsible for interpreting the request and executing the required
    database actions.

    The dbmanager subagent accepts ONLY a single string as input.

    The input must contain one or more instructions written in natural language
    describing the operation to perform. Instructions should clearly express the
    intended action and any filtering or business criteria.

    Typical supported operations include:
    - Retrieve existing suppliers
    - Save newly discovered suppliers
    - Insert structured data into persistent storage
    - Update existing records
    - Execute database lookups and filtering
    - Perform generic database management tasks

    Examples:
    - "Find all building material suppliers in Pisa"
    - "Retrieve suppliers of ceramic tiles located in Turin"
    - "Store suppliers contained in artifact vYtM84r9uP2E9zXyB6Hn9c"
    - "Update supplier records using data from artifact vYtZ84r9uP5E9zXyC6Hn7c"

    IMPORTANT — Data exchange through artifacts:
    Large or structured datasets MUST NOT be embedded directly inside the query string.

    When data needs to be provided to the dbmanager:
    - Store the data in the artifact store first.
    - Pass only the corresponding `artifact_id` inside the instruction.

    Correct:
    - "Save suppliers from artifact vYtM84r9uP2E9zXyB6Hn9c"

    Incorrect:
    - "Save these suppliers: { ...large json payload... }"

    Similarly, when retrieving data:
    - The dbmanager may persist the result into an artifact.
    - In such cases, the returned value will contain an `artifact_id`
      instead of the full dataset.

    This artifact-based communication model is intentional and provides:
    - Reduced message size between agents
    - Separation between execution and data transport
    - Improved scalability for large datasets
    - Reusability of generated results across workflow steps
    - Persistent storage and traceability

    Args:
        query:
            One or more natural language instructions describing the database
            operation to execute.

            If input data is required, reference it through an `artifact_id`
            instead of embedding raw content.

    Returns:
        str:
            A textual execution result.

            Depending on the operation, this may contain:
            - A confirmation message
            - A database operation status
            - An `artifact_id` pointing to persisted retrieved data
            - An error message if execution fails

    Raises:
        Does not raise exceptions directly.
        Execution failures are captured and returned as strings:
        - "Error in agent_dbmanager execution: ..."
        - "Tool exception: ..."
    """
    try:
        from agent_dbmanager.agent import run_dbmanager

        logger.info(f"[ORCHESTRATOR] Delegating to agent_dbmanager with query: {query}")

        if VERBOSE:
            print(f"\n[TOOLS] run_dbmanager_agent\n  Arguments: query=\"{query}\"")

        result, error = run_dbmanager(query, verbose=VERBOSE)

        logger.info(f"[ORCHESTRATOR] Result from agent_dbmanager: {result}, Error: {error}")

        if error:
            raise DbManagerAgentError(f"agent_dbmanager execution failed: {error}")

        if VERBOSE:
            print(f"  Result: {result}")

        return result
    except DbManagerAgentError as e:
        logger.error(f"[ORCHESTRATOR] {e}")
        result_str = f"Error in agent_dbmanager execution: {e}"
        if VERBOSE:
            print(f"  Result: {result_str}")
        return result_str
    except ArtifactError as e:
        logger.error(f"[ORCHESTRATOR] Artifact error in agent_dbmanager: {e}")
        result_str = f"Error in agent_dbmanager execution: {e}"
        if VERBOSE:
            print(f"  Result: {result_str}")
        return result_str
    except OrchestratorError as e:
        logger.error(f"[ORCHESTRATOR] {e}")
        result_str = str(e)
        if VERBOSE:
            print(f"  Result: {result_str}")
        return result_str
    except Exception as e:
        logger.error(f"[ORCHESTRATOR] Unexpected exception in agent_dbmanager: {e}")
        result_str = f"Tool exception: {str(e)}"
        if VERBOSE:
            print(f"  Result: {result_str}")
        return result_str