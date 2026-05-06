from fastapi import FastAPI, Query
import json

from .search import search_web
from .processor import process_results
from .llm import generate_answer
from .logger import logger

app = FastAPI(title="AI Search Pipeline")

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/ask")
def ask(q: str = Query(...)):
    """
    Full AI pipeline:
    search → process → LLM → answer
    """

    # 1. Search web
    results = search_web(q)

    # 2. Process results (intelligence layer)
    context = process_results(results)
    logger.info(context)

    # 3. LLM final answer
    answer = generate_answer(context)

    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(answer, f, indent=4, ensure_ascii=False)

    return {
        "query": q,
        "answer": answer
    }