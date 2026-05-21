import json
import sys
import re
from stats import get_stats
from logger import logger


def _run_cli(mode: str, run_agent, output_path: str) -> None:
    print(f"\nLocal AI Search CLI ({mode} Mode)")
    print("Type 'exit' to quit\n")

    while True:
        try:
            query = input("AI Search > ").strip()
        except KeyboardInterrupt:
            print("\nExiting...")
            break

        if not query:
            continue

        if query.lower() in {"exit", "quit"}:
            break

        try:
            answer, error = run_agent(query)

            if error or answer is None:
                logger.error(f"[CLI ERROR] Query: '{query}' - Agent error: {error}")
                print(f"\nERROR: {error}\n")
                continue

            stats = get_stats().to_dict()
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"query": query, "answer": answer, "stats": stats, "error": None},
                    f, indent=4, ensure_ascii=False
                )

            logger.info(f"[STATS] {stats}")

            print(f"\nANSWER:\n\n{answer}\n")
            print(f"STATS: {stats}\n")
            print("-" * 50 + "\n")

        except Exception as e:
            logger.error(f"[CLI ERROR] Exception: {e}")
            print(f"Unexpected exception: {e}")


def main_tool():
    """Run the Tool agent mode."""
    from agent_websearch.agent_tool import run_agent as run_agent_tool

    def run_agent(query):
        raw_answer, error = run_agent_tool(query)
        if error or raw_answer is None:
            return None, error

        cleaned_text = raw_answer.strip()
        match = re.search(r'```(?:json)?(.*?)```', cleaned_text, re.DOTALL | re.IGNORECASE)
        if match:
            cleaned_text = match.group(1).strip()

        try:
            parsed_json = json.loads(cleaned_text)
            return parsed_json, None
        except json.JSONDecodeError as e:
            logger.error(f"[AGENT PARSE ERROR] {str(e)}\nReceived string: {raw_answer}")
            return None, f"JSON Decode Error: {str(e)}"

    _run_cli("Tool", run_agent, "src/outputs/output_agent_tool.json")


def main_pipeline():
    """Run the Pipeline agent mode."""
    from agent_websearch.agent_pipeline import run_agent as run_agent_pipeline

    def run_agent(query):
        result = run_agent_pipeline(query)
        return result.get("final_answer"), result.get("error")

    _run_cli("Pipeline", run_agent, "src/outputs/output_agent_pipeline.json")


def main_orchestrator():
    """Run the Orchestrator agent mode."""
    from agent_orchestrator.agent import run_orchestrator

    def run_agent(query):
        raw_answer, error = run_orchestrator(query)

        if error or raw_answer is None:
            return None, error

        cleaned_text = raw_answer.strip()
        match = re.search(r'```(?:json)?(.*?)```', cleaned_text, re.DOTALL | re.IGNORECASE)
        if match:
            cleaned_text = match.group(1).strip()

        try:
            parsed_json = json.loads(cleaned_text)
            return parsed_json, None
        except json.JSONDecodeError as e:
            logger.error(f"[AGENT PARSE ERROR] {str(e)}\nReceived string: {raw_answer}")
            return None, f"JSON Decode Error: {str(e)}"

    _run_cli("Orchestrator", run_agent, "src/outputs/output_orchestrator.json")


if __name__ == "__main__":
    print("\n--- Setup CLI ---")
    print("Choose which version of the program you want to run:")
    print("1. Tool - Autonomous LLM-driven supplier search")
    print("2. Pipeline - Structured multi-step search")
    print("3. Orchestrator - Coordinates Tool and DB agents")

    try:
        choice = ""
        while choice not in {"1", "2", "3"}:
            choice = input("Enter 1, 2 or 3: ").strip()

        if choice == "1":
            main_tool()
        elif choice == "2":
            main_pipeline()
        elif choice == "3":
            main_orchestrator()

    except KeyboardInterrupt:
        print("\nProgram interrupted. Goodbye!")
        sys.exit(0)