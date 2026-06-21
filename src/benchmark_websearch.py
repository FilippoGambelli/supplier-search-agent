import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stats import get_stats, reset_stats
from tqdm import tqdm


QUERIES = [
    "fornitori piastrelle Bologna",
    "fornitori materiali edili Pisa",
    "fornitori legname Lucca",
    "fornitori infissi Firenze",
    "fornitori cemento Prato",
    "fornitori cartongesso Arezzo",
    "fornitori vernici Siena",
    "fornitori pavimenti Grosseto",
    "fornitori sanitari Massa",
    "fornitori ferramenta Pisa",
    "fornitori ponteggi Lucca",
    "fornitori coperture tetti Siena",
    "fornitori laterizi Prato",
    "fornitori ferro per edilizia Arezzo",
    "fornitori serramenti Massa",
]

SYSTEMS = {
    "tool": {
        "module": "agent_websearch.agent_tool",
        "label": "Agentic workflow",
    },
    "pipeline": {
        "module": "agent_websearch.agent_pipeline",
        "label": "Deterministic workflow",
    },
}

# Colors used consistently across charts for each system
SYSTEM_COLORS = {
    "tool": "#2196F3",       # blue  – agentic
    "pipeline": "#FF5722",   # deep-orange – deterministic
}

# Mean-line colors (distinct from the bar colors above)
MEAN_LINE_COLORS = {
    "tool": "#0D47A1",       # dark blue
    "pipeline": "#BF360C",   # dark red-orange
}

CSV_COLUMNS = [
    "run_id",
    "query_index",
    "query",
    "system",
    "result_count",
    "total_llm_requests",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "execution_time_seconds",
]


def strip_markdown_fences(value: str) -> str:
    """Remove leading/trailing markdown code fences (e.g. ```json ... ```)."""
    stripped = value.strip()
    # Remove opening fence: ```json or ``` (with optional language tag)
    if stripped.startswith("```"):
        # Drop the first line (the fence + optional language tag)
        first_newline = stripped.find("\n")
        if first_newline != -1:
            stripped = stripped[first_newline + 1:]
    # Remove closing fence
    if stripped.endswith("```"):
        stripped = stripped[: stripped.rfind("```")].rstrip()
    return stripped


def load_json_if_possible(value):
    """Try to decode a JSON string and return the parsed value when possible.

    Also handles strings wrapped in markdown code fences (```json ... ```),
    which the agentic workflow sometimes produces.
    """
    if not isinstance(value, str):
        return value

    # First attempt: parse as-is
    try:
        return json.loads(value)
    except Exception:
        pass

    # Second attempt: strip markdown fences and retry
    cleaned = strip_markdown_fences(value)
    if cleaned != value:
        try:
            return json.loads(cleaned)
        except Exception:
            pass

    return value


def normalize_output(result) -> list:
    """Convert different response shapes into a plain list of results."""
    if not result:
        return []

    if isinstance(result, dict):
        if "final_answer" in result:
            return result["final_answer"] or []
        if "answer" in result:
            result = result["answer"]

    result = load_json_if_possible(result)

    if isinstance(result, list):
        return result

    return []


def import_runner(module_path: str) -> Callable[[str], Tuple[object, Optional[str]]]:
    """Load the runner function from a module and wrap it in a simple callable."""
    module = import_module(module_path)
    runner = getattr(module, "run_agent")
    return lambda query: runner(query, False)


def run_system(query: str, runner: Callable):
    """Run one query through one system and collect stats for the execution."""
    reset_stats()
    stats = get_stats()
    stats.start()

    message = None
    error = None

    try:
        message, error = runner(query)
    except Exception as exc:
        error = str(exc)

    stats.stop()
    return message, error, stats.to_dict()


def count_results(message: object) -> int:
    """Count how many results were returned by a system."""
    return len(normalize_output(message))


def to_float(row: Dict[str, str], key: str) -> float:
    """Read a numeric field from a CSV row and return a safe float value."""
    try:
        return float(row.get(key, 0) or 0)
    except Exception:
        return 0.0


def extract_domains(result) -> set:
    """Extract unique domains from result items that contain a website field."""
    domains = set()

    for item in normalize_output(result):
        if not isinstance(item, dict):
            continue

        website = item.get("website")
        if not website:
            continue

        try:
            parsed = urlparse(website)
            domain = parsed.netloc.lower().replace("www.", "")
            if domain:
                domains.add(domain)
        except Exception:
            continue

    return domains


def save_row(path: Path, row: Dict[str, str]):
    """Append one benchmark row to the CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()

    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def group_rows_by_query(rows: List[Dict[str, str]]) -> Dict[int, Dict[str, Dict[str, str]]]:
    """Group rows by query index and system name."""
    grouped: Dict[int, Dict[str, Dict[str, str]]] = defaultdict(dict)

    for row in rows:
        query_index = int(row["query_index"])
        system_name = row["system"]
        grouped[query_index][system_name] = row

    return grouped


def benchmark(csv_path: Path, queries: List[str], systems: Dict[str, str]):
    """Run every query against every system and store the benchmark results."""
    # Clear the CSV file so each benchmark run starts fresh
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        pass

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    runners = {
        name: import_runner(config["module"])
        for name, config in systems.items()
    }

    outputs_cache = defaultdict(dict)
    current_rows: List[Dict[str, str]] = []
    total_runs = len(queries) * len(runners)

    progress = tqdm(total=total_runs, desc="Benchmark runs", unit="run")

    for query_index, query in enumerate(queries, 1):
        for system_name, runner in runners.items():
            message, error, stats = run_system(query, runner)

            outputs_cache[query_index][system_name] = message

            row = {
                "run_id": run_id,
                "query_index": str(query_index),
                "query": query,
                "system": system_name,
                "result_count": str(count_results(message)),
                "total_llm_requests": str(stats.get("total_LLM_requests", 0)),
                "input_tokens": str(stats.get("input_tokens", 0)),
                "output_tokens": str(stats.get("output_tokens", 0)),
                "total_tokens": str(stats.get("total_tokens", 0)),
                "execution_time_seconds": str(stats.get("total_execution_time_seconds", 0)),
            }

            save_row(csv_path, row)
            current_rows.append(row)
            progress.update(1)

    progress.close()
    return outputs_cache, current_rows


def compare_outputs(outputs_cache: Dict, out: Path):
    """Compare the domains returned by each system for every query."""
    rows = []

    for query_index in sorted(outputs_cache.keys()):
        systems = outputs_cache[query_index]

        tool_domains = extract_domains(systems.get("tool"))
        pipeline_domains = extract_domains(systems.get("pipeline"))

        shared = tool_domains & pipeline_domains
        only_tool = tool_domains - pipeline_domains
        only_pipeline = pipeline_domains - tool_domains

        rows.append(
            {
                "query_index": query_index,
                "tool_count": len(tool_domains),
                "pipeline_count": len(pipeline_domains),
                "intersection": len(shared),
                "only_tool": len(only_tool),
                "only_pipeline": len(only_pipeline),
            }
        )

    return pd.DataFrame(rows)


def _compute_means(
    grouped: Dict[int, Dict[str, Dict[str, str]]],
    system_names: List[str],
    metric: str,
) -> Dict[str, float]:
    """Return the mean value of *metric* across all queries for each system."""
    means: Dict[str, float] = {}
    query_indexes = sorted(grouped.keys())

    for system_name in system_names:
        values = [
            to_float(grouped[q][system_name], metric)
            for q in query_indexes
            if system_name in grouped[q]
        ]
        means[system_name] = float(np.mean(values)) if values else 0.0

    return means


def plot_execution_time_per_query(rows: List[Dict[str, str]], out: Path):
    """
    Grouped bar chart of execution time per query with a horizontal mean line
    for each system drawn in a clearly distinguishable colour.
    """
    grouped = group_rows_by_query(rows)
    query_indexes = sorted(grouped.keys())
    system_names = list(SYSTEMS.keys())
    labels = [SYSTEMS[name]["label"] for name in system_names]

    means = _compute_means(grouped, system_names, "execution_time_seconds")

    x = np.arange(len(query_indexes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(12, len(query_indexes) * 1.2), 6))

    for i, system_name in enumerate(system_names):
        values = [
            to_float(grouped[q][system_name], "execution_time_seconds")
            for q in query_indexes
        ]
        offset = (i - (len(system_names) - 1) / 2) * width
        ax.bar(
            x + offset,
            values,
            width=width,
            label=labels[i],
            color=SYSTEM_COLORS[system_name],
            alpha=0.85,
        )

    # Horizontal mean lines
    for system_name in system_names:
        mean_val = means[system_name]
        color = MEAN_LINE_COLORS[system_name]
        label = f"Mean – {SYSTEMS[system_name]['label']} ({mean_val:.1f}s)"
        ax.axhline(
            mean_val,
            color=color,
            linewidth=2.5,
            linestyle="--",
            label=label,
            zorder=5,
        )

    ax.set_title("Execution time per query")
    ax.set_xlabel("Query")
    ax.set_ylabel("Seconds")
    ax.set_xticks(x)
    ax.set_xticklabels([str(q) for q in query_indexes])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "execution_time_per_query.png", dpi=200)
    plt.close(fig)


def plot_total_tokens_per_query(rows: List[Dict[str, str]], out: Path):
    """
    Grouped bar chart of *total* tokens per query with a horizontal mean line
    for each system drawn in a clearly distinguishable colour.
    """
    grouped = group_rows_by_query(rows)
    query_indexes = sorted(grouped.keys())
    system_names = list(SYSTEMS.keys())
    labels = [SYSTEMS[name]["label"] for name in system_names]

    means = _compute_means(grouped, system_names, "total_tokens")

    x = np.arange(len(query_indexes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(12, len(query_indexes) * 1.2), 6))

    for i, system_name in enumerate(system_names):
        values = [
            to_float(grouped[q][system_name], "total_tokens")
            for q in query_indexes
        ]
        offset = (i - (len(system_names) - 1) / 2) * width
        ax.bar(
            x + offset,
            values,
            width=width,
            label=labels[i],
            color=SYSTEM_COLORS[system_name],
            alpha=0.85,
        )

    # Horizontal mean lines
    for system_name in system_names:
        mean_val = means[system_name]
        color = MEAN_LINE_COLORS[system_name]
        label = f"Mean – {SYSTEMS[system_name]['label']} ({mean_val:,.0f} tok)"
        ax.axhline(
            mean_val,
            color=color,
            linewidth=2.5,
            linestyle="--",
            label=label,
            zorder=5,
        )

    ax.set_title("Total tokens per query")
    ax.set_xlabel("Query")
    ax.set_ylabel("Tokens")
    ax.set_xticks(x)
    ax.set_xticklabels([str(q) for q in query_indexes])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "total_tokens_per_query.png", dpi=200)
    plt.close(fig)


def plot_result_overlap(df: pd.DataFrame, out: Path):
    """Create a stacked bar chart showing result overlap between systems."""
    if df.empty:
        return

    x = df["query_index"].astype(int)

    shared = df["intersection"].values
    only_tool = df["only_tool"].values
    only_pipeline = df["only_pipeline"].values

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x, shared, label="Shared results")
    ax.bar(x, only_tool, bottom=shared, label="Only agentic workflow")
    ax.bar(x, only_pipeline, bottom=shared + only_tool, label="Only deterministic workflow")

    ax.set_title("Result overlap per query")
    ax.set_xlabel("Query")
    ax.set_ylabel("Number of results")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "result_overlap.png", dpi=200)
    plt.close(fig)


def create_charts(rows: List[Dict[str, str]], out: Path, outputs_cache=None):
    """Read the benchmark data and generate the charts."""
    if not rows:
        return

    out.mkdir(parents=True, exist_ok=True)

    plot_execution_time_per_query(rows, out)
    plot_total_tokens_per_query(rows, out)

    if outputs_cache:
        df = compare_outputs(outputs_cache, out)
        plot_result_overlap(df, out)


def parse_args():
    """Read the optional command-line limit for the number of queries."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main():
    """Run the benchmark and generate the output charts."""
    args = parse_args()

    base_dir = Path(__file__).resolve().parent
    benchmark_dir = base_dir / "benchmark_results"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    csv_path = benchmark_dir / "websearch.csv"
    charts_dir = benchmark_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    queries = QUERIES[: args.limit] if args.limit else QUERIES

    outputs_cache, current_rows = benchmark(csv_path, queries, SYSTEMS)
    create_charts(current_rows, charts_dir, outputs_cache)


if __name__ == "__main__":
    main()