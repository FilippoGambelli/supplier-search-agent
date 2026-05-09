"""FastAPI application and API routes for AI Search Pipeline."""

import json
from fastapi import FastAPI, APIRouter, HTTPException, Query

from ..agent.agent import run_agent
from ..logger import logger

app = FastAPI(title="AI Search Pipeline with LangGraph")

router = APIRouter()

@router.get("/")
def root():
    """Health check endpoint."""
    return {"status": "ok", "agent": "LangGraph"}

@router.get("/ask")
def ask(q: str):
    result = run_agent(q)
    
    answer = result.get("answer")
    error = result.get("error")
    
    if error is not None or answer is None:
        logger.error(f"[API ERROR] Query: '{q}' - Errore dell'agente: {error}")
        
        raise HTTPException(
            status_code=500, 
            detail={
                "message": "Errore durante l'elaborazione della richiesta.",
                "error": error,
                "raw_answer": result.get("raw_answer")
            }
        )
        
    try:
        results_count = len(answer)
    except TypeError:
        results_count = 1 

    with open("agent_tool/output.json", "w", encoding="utf-8") as f:
        json.dump({
            "query": q,
            "answer": answer,
            "error": error
        }, f, indent=4, ensure_ascii=False)
    
    logger.info(f"[API RESPONSE] Query: '{q}' - Results: {results_count}")
    
    return {
        "status": "success",
        "data": answer
    }

app.include_router(router)