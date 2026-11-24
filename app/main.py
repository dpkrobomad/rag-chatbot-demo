"""FastAPI app: /api/ask answers questions from the documents in ./docs with citations."""
from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .rag import generate, get_index, reload_index

app = FastAPI(title="RAG Chatbot Demo", version="1.0.0")
STATIC = Path(__file__).parent.parent / "static"


class Question(BaseModel):
    question: str = Field(..., min_length=2, max_length=500)


class Source(BaseModel):
    n: int
    source: str
    heading: str
    score: float
    excerpt: str


class Answer(BaseModel):
    answer: str
    mode: str
    sources: List[Source]


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "chunks": len(get_index().chunks)}


@app.post("/api/ask", response_model=Answer)
def ask(q: Question) -> Answer:
    hits = get_index().search(q.question)
    answer, mode = generate(q.question, hits)
    return Answer(
        answer=answer,
        mode=mode,
        sources=[
            Source(n=i, source=h.chunk.source, heading=h.chunk.heading, score=round(h.score, 3),
                   excerpt=h.chunk.text[:240])
            for i, h in enumerate(hits, 1)
        ],
    )


@app.post("/api/reload")
def reload() -> dict:
    try:
        return {"chunks": reload_index()}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
