import json
import re
from colorama import Fore, Style, init
from stats import get_stats, reset_stats
from config import *

init(autoreset=True)

# MODES
MODES = {"tool", "pipeline", "orchestrator"}
CURRENT_MODE = "orchestrator"

# COLORS
CYAN = Fore.CYAN + Style.BRIGHT
MAGENTA = Fore.MAGENTA + Style.BRIGHT
RED = Fore.RED + Style.BRIGHT
DIM = Style.DIM
GREEN = Fore.GREEN + Style.BRIGHT
YELLOW = Fore.YELLOW + Style.BRIGHT

def clean(text):
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text).replace("\n\n", "\n")).strip()

def extract_json(raw: str) -> str:
    raw = raw.strip()
    match = re.search(r'```(?:json)?(.*?)```', raw, re.DOTALL)
    return match.group(1).strip() if match else raw


def print_header():
    print(CYAN + Style.BRIGHT + r"""
███████╗ ███████╗  █████╗ 
██╔════╝ ██╔════╝ ██╔══██╗
███████╗ ███████╗ ███████║
╚════██║ ╚════██║ ██╔══██║
███████║ ███████║ ██║  ██║
╚══════╝ ╚══════╝ ╚═╝  ╚═╝
""" + Style.RESET_ALL)

    print(DIM + "CLI Multi-Agent System")
    print(DIM + "Modes: tool | pipeline | orchestrator")
    print(DIM + "Use: /mode <name>\n")


# MODE HANDLER
def handle_mode(cmd: str):
    global CURRENT_MODE

    parts = cmd.split()

    if len(parts) == 1:
        print("\nAvailable modes:", ", ".join(MODES), "\n")
        return

    mode = parts[1].lower()

    if mode in MODES:
        CURRENT_MODE = mode
        print(f"\n→ mode: {CURRENT_MODE}\n")
    else:
        print("\nInvalid mode\n")


def print_node_pipeline(event: dict):
    for key in event:
        print(MAGENTA + f"\nNODE: {key.upper()}")


def print_event(agent: str, event: dict):

    if "agent" in event:
        messages = event["agent"].get("messages", [])

        for msg in messages:
            print(CYAN + f"\n[AGENT - {agent}]")

            # reasoning vs content
            content = getattr(msg, "content", None)
            reasoning = getattr(msg, "additional_kwargs", {}).get("reasoning_content", None)
            if content:
                print(DIM + clean(f"Response:\n{content}"))
            elif reasoning:
                print(DIM + clean(f"Reasoning:\n{reasoning}"))

            # tool calls
            tool_calls = getattr(msg, "tool_calls", []) or []
            if tool_calls:
                print(DIM + "Tools called:")
                for tc in tool_calls:
                    name = tc.get("name")
                    args = tc.get("args")
                    print(DIM + f"- {name} | {json.dumps(args, ensure_ascii=False)}")

    elif "tools" in event:
        tool_messages = event["tools"].get("messages", [])
        
        print(MAGENTA + f"\n[TOOLS - {agent}]")

        for msg in tool_messages:
            content = getattr(msg, "content", None)
            name = getattr(msg, "name", None)
            if name:
                print(DIM + f"Tool name: {name}")
            if content:
                print(DIM + clean(f"Result: {content}"))


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

    print_header()
    runner = get_runner(CURRENT_MODE)

    while True:
        query = input(f"{CURRENT_MODE} > ").strip()

        if query in {"exit", "quit"}:
            print("\nBye!\n")
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
                print(GREEN + f"\nFinal answer:\n{msg}\n\n")
            
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