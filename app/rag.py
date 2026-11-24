"""Document loading, chunking, retrieval and answer generation."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import httpx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SUPPORTED = {".md", ".txt"}


@dataclass
class Chunk:
    source: str
    heading: str
    text: str


@dataclass
class Hit:
    chunk: Chunk
    score: float


def load_chunks(folder: Path, max_chars: int = 800) -> List[Chunk]:
    """Split each document by Markdown headings, then by paragraph so no chunk exceeds max_chars."""
    chunks: List[Chunk] = []
    for path in sorted(folder.rglob("*")):
        if path.suffix.lower() not in SUPPORTED:
            continue
        heading = path.stem
        buf: List[str] = []

        def flush() -> None:
            text = "\n".join(buf).strip()
            if text:
                chunks.append(Chunk(path.name, heading, text))
            buf.clear()

        for line in path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^#{1,6}\s+(.*)", line)
            if m:
                flush()
                heading = m.group(1).strip()
                continue
            if sum(len(b) for b in buf) + len(line) > max_chars and not line.strip():
                flush()
            buf.append(line)
        flush()
    return chunks


class Index:
    def __init__(self, chunks: List[Chunk]):
        if not chunks:
            raise ValueError("No documents found to index.")
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([f"{c.heading}\n{c.text}" for c in chunks])

    def search(self, query: str, k: int = 4, min_score: float = 0.05) -> List[Hit]:
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        ranked = scores.argsort()[::-1][:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in ranked if scores[i] >= min_score]


SYSTEM_PROMPT = (
    "You answer customer questions using ONLY the numbered sources provided. "
    "Cite sources inline like [1]. If the sources don't contain the answer, say you don't know "
    "and suggest contacting the team. Be concise and friendly."
)


def build_prompt(question: str, hits: List[Hit]) -> str:
    sources = "\n\n".join(
        f"[{i}] ({h.chunk.source} — {h.chunk.heading})\n{h.chunk.text}" for i, h in enumerate(hits, 1)
    )
    return f"Sources:\n{sources}\n\nQuestion: {question}"


def generate(question: str, hits: List[Hit]) -> tuple:
    """Return (answer, mode). Uses any OpenAI-compatible endpoint (Ollama, OpenAI, Groq...).
    Falls back to an extractive answer if the model isn't reachable."""
    if not hits:
        return ("I couldn't find that in the documents. Please contact the team directly.", "no-match")

    base = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1").rstrip("/")
    model = os.getenv("LLM_MODEL", "qwen2.5:3b")
    key = os.getenv("LLM_API_KEY", "ollama")
    try:
        r = httpx.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": model,
                "temperature": 0.1,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_prompt(question, hits)},
                ],
            },
            timeout=float(os.getenv("LLM_TIMEOUT", "60")),
        )
        r.raise_for_status()
        return (r.json()["choices"][0]["message"]["content"].strip(), "llm")
    except (httpx.HTTPError, KeyError, ValueError):
        top = hits[0].chunk
        return (f"Here's the most relevant section ({top.heading}) [1]:\n\n{top.text}", "extractive")


_index: Optional[Index] = None


def get_index() -> Index:
    global _index
    if _index is None:
        _index = Index(load_chunks(Path(os.getenv("DOCS_DIR", Path(__file__).parent.parent / "docs"))))
    return _index


def reload_index() -> int:
    global _index
    _index = None
    return len(get_index().chunks)
