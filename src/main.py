from pathlib import Path
import json
from colorama import Fore, Style, init
from stats import get_stats, reset_stats
from config import *

init(autoreset=True)

MODES = {"tool", "pipeline", "orchestrator"}
CURRENT_MODE = "orchestrator"

RED = Fore.RED + Style.BRIGHT
YELLOW = Fore.YELLOW + Style.BRIGHT

MODE_MAP = {"1": "tool", "2": "pipeline", "3": "orchestrator"}

OUTPUT_DIR = Path("src/outputs")
MODE_FILE_MAP = {
    "tool": "agentic_websearch",
    "pipeline": "pipeline_websearch",
    "orchestrator": "orchestrator_websearch_store",
}


def save_output(content: str, mode: str):
    OUTPUT_DIR.mkdir(exist_ok=True)
    name = MODE_FILE_MAP.get(mode, mode)
    path = OUTPUT_DIR / f"{name}.json"
    try:
        data = json.loads(content)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except json.JSONDecodeError:
        path.write_text(content, encoding="utf-8")


def print_mode_help():
    print("\nModes: [1] Agentic WebSearch | [2] Pipeline WebSearch | [3] WebSearch + Store")
    print("Use: /mode <number>")


def handle_mode(cmd: str):
    global CURRENT_MODE

    parts = cmd.split()

    if len(parts) == 1:
        return

    raw = parts[1].lower()
    mode = MODE_MAP.get(raw, raw)

    if mode in MODES:
        CURRENT_MODE = mode
    else:
        print("\nInvalid mode. Use /mode <number>")


def get_runner(mode: str):
    if mode == "tool":
        from agent_websearch.agent_tool import run_agent as run_websearch_agent
        return run_websearch_agent
    
    elif mode == "pipeline":
        from agent_websearch.agent_pipeline import run_agent as run_websearch_agent
        return run_websearch_agent
    
    elif mode == "orchestrator":
        from agent_orchestrator.agent import run_orchestrator
        return run_orchestrator


def run_cli():
    global CURRENT_MODE

    runner = get_runner(CURRENT_MODE)

    while True:
        print_mode_help()
        query = input(f"{CURRENT_MODE} > ").strip()

        if query in {"exit", "quit"}:
            print("Bye!\n")
            break

        if query.startswith("/mode"):
            handle_mode(query)
            runner = get_runner(CURRENT_MODE)
            continue

        try:
            reset_stats()
            stats = get_stats()
            stats.start()
            msg, error = runner(query)
            stats.stop()

            if error:
                print(RED + f"ERROR: {error}")
            else:
                save_output(msg, CURRENT_MODE)
                print(f"\nFinal answer:\n{msg}\n")

            stats_dict = stats.to_dict()
            input_tokens = stats_dict.get("input_tokens", 0)
            output_tokens = stats_dict.get("output_tokens", 0)
            total_tokens = stats_dict.get("total_tokens", 0)
            exec_time = stats_dict.get("total_execution_time_seconds", 0)

            input_pct = (input_tokens / total_tokens * 100) if total_tokens else 0
            output_pct = (output_tokens / total_tokens * 100) if total_tokens else 0

            print(YELLOW + (
                    f"STATS: "
                    f"input tokens: {input_tokens} ({input_pct:.1f}%) | "
                    f"output tokens: {output_tokens} ({output_pct:.1f}%) | "
                    f"total tokens: {total_tokens} | "
                    f"total execution time: {exec_time:.2f}s\n\n"
                )
            )
        except Exception as e:
            print(RED + f"\nEXCEPTION ({type(e).__name__}): {e}\n")

if __name__ == "__main__":
    run_cli()
