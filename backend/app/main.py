from fastapi import FastAPI, Query
import json

from .agent import run_agent
from .logger import logger

app = FastAPI(title="AI Search Pipeline with LangGraph")

@app.get("/")
def root():
    return {"status": "ok", "agent": "LangGraph"}


@app.get("/ask")
def ask(q: str = Query(...)):
    """
    Full AI pipeline using LangGraph:
    search → scrape → extract → final_answer
    """

    # Run the LangGraph agent
    result = run_agent(q)

    # Prepare response
    answer = result.get("final_answer", [])
    error = result.get("error")

    # Save to output file
    with open("output.json", "w", encoding="utf-8") as f:
        json.dump({
            "query": q,
            "answer": answer,
            "error": error
        }, f, indent=4, ensure_ascii=False)

    logger.info(f"[API RESPONSE] Query: '{q}' - Results: {len(answer)}")

    return {
        "query": q,
        "answer": answer,
        "error": error
    }