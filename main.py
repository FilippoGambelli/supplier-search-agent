import json
import sys
from agent_tool.agent.agent import run_agent as run_tool_agent
from agent_pipeline.agent.agent import run_agent as run_pipeline_agent
from agent_tool.logger import logger as tool_logger
from agent_pipeline.logger import logger as pipeline_logger

def main_tool():
    print("\nLocal AI Search CLI (Tool Mode)")
    print("Type 'exit' to quit\n")

    while True:
        try:
            query = input("AI Search > ")
        except KeyboardInterrupt:
            print("\nExiting...")
            break

        if query.lower() in ["exit", "quit"]:
            break

        try:
            answer, error = run_tool_agent(query)

            if error:
                tool_logger.error(f"[CLI ERROR] Query: '{query}' - Agent error: {error}")
                print(f"\nERROR: {error}\n")
                continue

            with open("agent_tool/output.json", "w", encoding="utf-8") as f:
                json.dump({
                    "query": query,
                    "answer": answer,
                    "error": error
                }, f, indent=4, ensure_ascii=False)
            
            tool_logger.info(f"[CLI SUCCESS] Query: '{query}' processed successfully.")

            print("\nANSWER:\n")
            print(answer)
            print("\n" + "-" * 50 + "\n")

        except Exception as e:
            print("Unexpected exception:", e)
            tool_logger.error(f"Exception: {str(e)}")


def main_pipeline():
    print("\nLocal AI Search CLI (Pipeline Mode)")
    print("Type 'exit' to quit\n")

    while True:
        try:
            query = input("AI Search > ")
        except KeyboardInterrupt:
            print("\nExiting...")
            break

        if query.lower() in ["exit", "quit"]:
            break

        try:
            # Call the agent directly without going through the web server
            result = run_pipeline_agent(query)
            
            answer = result.get("answer")
            error = result.get("error")

            if error is not None or answer is None:
                pipeline_logger.error(f"[CLI ERROR] Query: '{query}' - Agent error: {error}")
                print(f"\nERROR: {error}\n")
                continue

            # Save to the correct JSON file for the pipeline
            with open("agent_pipeline/output.json", "w", encoding="utf-8") as f:
                json.dump({
                    "query": query,
                    "answer": answer,
                    "error": error
                }, f, indent=4, ensure_ascii=False)
            
            pipeline_logger.info(f"[CLI SUCCESS] Query: '{query}' processed successfully.")

            print("\nANSWER:\n")
            print(answer)
            print("\n" + "-" * 50 + "\n")

        except Exception as e:
            print("Unexpected exception:", e)
            pipeline_logger.error(f"Exception: {str(e)}")


if __name__ == "__main__":
    print("\n--- Setup CLI ---")
    print("Choose which version of the program you want to run:")
    print("1. Tool")
    print("2. Pipeline")
    
    try:
        choice = ""
        while choice not in ["1", "2"]:
            choice = input("Enter 1 or 2: ")

        if choice == "1":
            main_tool()
        elif choice == "2":
            main_pipeline()
    except KeyboardInterrupt:
        print("\nProgram interrupted. Goodbye!")
        sys.exit(0)