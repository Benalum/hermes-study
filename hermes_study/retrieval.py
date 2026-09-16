from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .db import StudyDB
from .llm import OllamaClient


class Retriever:
    def __init__(self, db: StudyDB, llm: OllamaClient, top_k: int = 6):
        self.db = db
        self.llm = llm
        self.top_k = top_k

    async def search(self, course_id: int, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        rows = self.db.chunks_for_course(course_id)
        if not rows:
            return []
        k = top_k or self.top_k
        embedded_rows = [r for r in rows if r.get("embedding")]
        scores: dict[int, float] = {}
        if embedded_rows:
            try:
                qvec = np.asarray((await self.llm.embed([query]))[0], dtype=float)
                qnorm = np.linalg.norm(qvec) or 1.0
                for r in embedded_rows:
                    vec = np.asarray(r["embedding"], dtype=float)
                    denom = (np.linalg.norm(vec) or 1.0) * qnorm
                    scores[r["id"]] = float(np.dot(qvec, vec) / denom)
            except Exception:
                scores = {}
        if not scores:
            corpus = [r["text"] for r in rows]
            try:
                matrix = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform(corpus + [query])
                sims = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
            except ValueError:
                sims = np.zeros(len(rows))
            scores = {r["id"]: float(s) for r, s in zip(rows, sims, strict=True)}
        ranked = []
        for r in rows:
            score = scores.get(r["id"], 0.0)
            authority_factor = 0.85 + (min(max(int(r["authority"]), 0), 100) / 100.0) * 0.30
            item = dict(r)
            item["score"] = score * authority_factor
            ranked.append(item)
        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked[:k]


def context_block(rows: list[dict[str, Any]]) -> str:
    blocks = []
    for i, r in enumerate(rows, 1):
        page = f", page {r['page']}" if r.get("page") else ""
        blocks.append(
            f"[SOURCE {i}: {r['filename']}{page}; type={r['source_type']}; authority={r['authority']}]\n{r['text']}"
        )
    return "\n\n".join(blocks)
