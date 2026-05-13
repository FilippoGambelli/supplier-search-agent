import json
import sys
import re
from stats import get_stats


def _run_cli(mode: str, run_agent, logger, output_path: str) -> None:
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

            logger.info(f"[CLI SUCCESS] Query processed successfully.")
            logger.info(f"[STATS] {stats}")
            
            print(f"\nANSWER:\n\n{answer}\n")
            print(f"STATS: {stats}\n")
            print("-" * 50 + "\n")

        except Exception as e:
            logger.error(f"[CLI ERROR] Exception: {e}")
            print(f"Unexpected exception: {e}")


def main_tool():
    from agent_tool.logger import logger as logger_tool
    from agent_tool.agent.agent import run_agent as run_agent_tool

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
            logger_tool.error(f"[AGENT PARSE ERROR] {str(e)}\nReceived string: {raw_answer}")
            return None, f"JSON Decode Error: {str(e)}"

    _run_cli("Tool", run_agent, logger_tool, "agent_tool/output.json")


def main_pipeline():
    from agent_pipeline.agent.agent import run_agent as run_agent_pipeline
    from agent_pipeline.logger import logger as logger_pipeline

    def run_agent(query):
        result = run_agent_pipeline(query)
        return result.get("final_answer"), result.get("error")

    _run_cli("Pipeline", run_agent, logger_pipeline, "agent_pipeline/output.json")


if __name__ == "__main__":
    print("\n--- Setup CLI ---")
    print("Choose which version of the program you want to run:")
    print("1. Tool")
    print("2. Pipeline")

    try:
        choice = ""
        while choice not in {"1", "2"}:
            choice = input("Enter 1 or 2: ").strip()

        if choice == "1":
            main_tool()
        elif choice == "2":
            main_pipeline()

    except KeyboardInterrupt:
        print("\nProgram interrupted. Goodbye!")
        sys.exit(0)