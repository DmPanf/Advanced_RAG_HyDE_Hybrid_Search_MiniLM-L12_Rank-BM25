"""FastAPI: Hybrid RAG чат для отеля — embedding + BM25 keyword search."""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from llm import generate_answer
from rag import HybridIndex

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("hotel-rag")

KB_FILE = os.getenv("KB_FILE", "hotel_kb.md")
HALLUCINATION_THRESHOLD = float(os.getenv("HALLUCINATION_THRESHOLD", "0.35"))

WELCOME_MESSAGE = (
    "Добрый день! Я AI-консультант премиального отеля для сотрудников. "
    "Подскажу регламенты по любым ситуациям: заселение, выезд, работа с гостями, "
    "бронирование, дресс-код, ЧП и т.д. Что вас интересует?"
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: List[ChatMessage] = Field(default_factory=list)


index: HybridIndex | None = None
index_error: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global index, index_error
    try:
        idx = HybridIndex(KB_FILE, threshold=HALLUCINATION_THRESHOLD)
        idx.build()
        index = idx
        logger.info("Hybrid RAG готов. Чанков: %d, порог: %.2f",
                    len(idx.chunks), HALLUCINATION_THRESHOLD)
    except FileNotFoundError as e:
        index_error = str(e)
        logger.error("База знаний не найдена: %s", e)
    except Exception as e:
        index_error = f"Ошибка инициализации RAG: {type(e).__name__}: {e}"
        logger.exception("Ошибка инициализации RAG")
    yield


app = FastAPI(title="Advanced-RAG-Hotel", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "welcome_message": WELCOME_MESSAGE,
            "index_error": index_error,
            "threshold": HALLUCINATION_THRESHOLD,
        },
    )


@app.get("/api/health")
async def health():
    return {
        "ok": index is not None,
        "chunks": len(index.chunks) if index else 0,
        "threshold": HALLUCINATION_THRESHOLD,
        "error": index_error,
    }


@app.post("/chat")
async def chat(req: ChatRequest):
    if index is None:
        return JSONResponse(
            status_code=503,
            content={
                "error": index_error or "RAG-индекс не готов.",
            },
        )

    try:
        sr = index.search(req.message, top_k=5)
    except Exception as e:
        logger.exception("Ошибка поиска")
        return JSONResponse(
            status_code=500,
            content={"error": f"Ошибка поиска: {e}"},
        )

    history_dicts = [m.model_dump() for m in req.history]
    result = await generate_answer(
        question=req.message,
        chunks=sr.final_hits,
        history=history_dicts,
        below_threshold=sr.below_threshold,
    )

    return {
        "keywords": sr.keywords,
        "embedding_hits": sr.embedding_hits,
        "bm25_hits": sr.bm25_hits,
        "final_hits": sr.final_hits,
        "max_similarity": sr.max_similarity,
        "below_threshold": sr.below_threshold,
        "threshold": HALLUCINATION_THRESHOLD,
        "answer": result.answer,
        "llm_model_used": result.model_used,
        "llm_error": result.error,
        "skipped_llm": result.skipped_llm,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("APP_PORT", "8000")),
        reload=False,
    )
