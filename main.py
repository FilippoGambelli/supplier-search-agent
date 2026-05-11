import json
import sys


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

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"query": query, "answer": answer, "error": None},
                    f, indent=4, ensure_ascii=False
                )

            logger.info(f"[CLI SUCCESS] Query: '{query}' processed successfully.")
            print(f"\nANSWER:\n\n{answer}\n")
            print("-" * 50 + "\n")

        except Exception as e:
            logger.error(f"Exception: {e}")
            print(f"Unexpected exception: {e}")


def main_tool():
    from agent_tool.logger import logger as logger_tool
    from agent_tool.agent.agent import run_agent as run_agent_tool

    _run_cli("Tool", run_agent_tool, logger_tool, "agent_tool/output.json")


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