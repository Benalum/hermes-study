from __future__ import annotations

import re
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .db import StudyDB


def _normalize_structure(text: str) -> str:
    """Expand common course labels so CH1C, Ch 1, and Chapter 1 match."""
    out = str(text or "")
    out = re.sub(
        r"(?i)(?<![a-z0-9])ch(?:apter)?[\s_-]*(\d+)([a-z]?)(?![a-z0-9])",
        lambda m: f" chapter {m.group(1)} {m.group(2)} ",
        out,
    )
    out = re.sub(
        r"(?i)(?<![a-z0-9])week[\s_-]*(\d+)(?![a-z0-9])",
        lambda m: f" week {m.group(1)} ",
        out,
    )
    return out


class Retriever:
    """Small-course retrieval that needs no second model server.

    Hermes owns the LLM. Retrieval stays local and deterministic with TF-IDF so a
    fresh Hermes-only Mac does not also need Ollama or an embedding service.
    """

    def __init__(self, db: StudyDB, llm_or_top_k=None, top_k: int = 6):
        self.db = db
        # v0.1 accepted (db, llm, top_k). Keep that call shape working while
        # retrieval itself is now intentionally model-free.
        if isinstance(llm_or_top_k, int):
            self.top_k = llm_or_top_k
        else:
            self.top_k = top_k

    async def search(self, course_id: int, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        rows = self.db.chunks_for_course(course_id)
        if not rows:
            return []
        k = top_k or self.top_k

        # Include source metadata in the searchable text. This matters for course
        # folders where the strongest structure is in names such as "Chapter 1",
        # "Week 3", "Syllabus", "Homework", or "Professor Slides".
        corpus = [
            _normalize_structure(
                "\n".join(
                    part for part in (
                        str(r.get("filename") or ""),
                        str(r.get("heading") or ""),
                        str(r.get("source_type") or ""),
                        str(r.get("text") or ""),
                    ) if part
                )
            )
            for r in rows
        ]
        normalized_query = _normalize_structure(query)
        try:
            matrix = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                sublinear_tf=True,
            ).fit_transform(corpus + [normalized_query])
            sims = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
        except ValueError:
            sims = np.zeros(len(rows))

        ranked = []
        for r, sim in zip(rows, sims, strict=True):
            # Keep class authority important, but never enough to beat a completely unrelated source.
            authority_factor = 0.85 + (min(max(int(r["authority"]), 0), 100) / 100.0) * 0.30
            item = dict(r)
            item["score"] = float(sim) * authority_factor
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
