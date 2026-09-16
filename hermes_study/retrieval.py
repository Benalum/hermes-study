from __future__ import annotations

import re
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .db import StudyDB


NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20",
}


def _normalize_structure(text: str) -> str:
    """Normalize CH2, Ch 2, Chapter Two, and similar structural labels."""
    out = str(text or "")
    word_pattern = "|".join(NUMBER_WORDS)
    out = re.sub(
        rf"(?i)(?<![a-z0-9])ch(?:apter)?[\s_-]*({word_pattern})(?![a-z0-9])",
        lambda m: f" chapter {NUMBER_WORDS[m.group(1).lower()]} ",
        out,
    )
    out = re.sub(
        r"(?i)(?<![a-z0-9])ch(?:apter)?[\s_-]*(\d+)([a-z]?)(?![a-z0-9])",
        lambda m: f" chapter {m.group(1)} {m.group(2)} ",
        out,
    )
    out = re.sub(
        rf"(?i)(?<![a-z0-9])week[\s_-]*({word_pattern})(?![a-z0-9])",
        lambda m: f" week {NUMBER_WORDS[m.group(1).lower()]} ",
        out,
    )
    out = re.sub(
        r"(?i)(?<![a-z0-9])week[\s_-]*(\d+)(?![a-z0-9])",
        lambda m: f" week {m.group(1)} ",
        out,
    )
    return out


def _structural_scope_pattern(scope: str | None) -> re.Pattern[str] | None:
    normalized = _normalize_structure(scope or "").lower()
    chapter = re.search(r"\bchapter\s+(\d+)\b", normalized)
    if chapter:
        return re.compile(rf"\bchapter\s+{re.escape(chapter.group(1))}\b", re.I)
    week = re.search(r"\bweek\s+(\d+)\b", normalized)
    if week:
        return re.compile(rf"\bweek\s+{re.escape(week.group(1))}\b", re.I)
    return None


def _row_structure_text(row: dict[str, Any]) -> str:
    return _normalize_structure(
        "\n".join(
            part for part in (
                str(row.get("structural_scope") or ""),
                str(row.get("relative_path") or ""),
                str(row.get("filename") or ""),
                str(row.get("heading") or ""),
            ) if part
        )
    )


class Retriever:
    """Local deterministic retrieval with explicit course-structure scoping."""

    def __init__(self, db: StudyDB, llm_or_top_k=None, top_k: int = 6):
        self.db = db
        if isinstance(llm_or_top_k, int):
            self.top_k = llm_or_top_k
        else:
            self.top_k = top_k

    async def search(
        self,
        course_id: int,
        query: str,
        top_k: int | None = None,
        *,
        scope: str | None = None,
        strict_scope: bool = False,
    ) -> list[dict[str, Any]]:
        rows = self.db.chunks_for_course(course_id)
        if not rows:
            return []

        scope_pattern = _structural_scope_pattern(scope)
        if scope_pattern is not None:
            # Prefer explicit metadata learned from the user's folder tree. Only
            # fall back to filename/heading inference for old, not-yet-organized
            # records so existing installations remain usable during migration.
            explicitly_scoped = [
                r for r in rows
                if r.get("structural_scope")
                and scope_pattern.search(_normalize_structure(str(r.get("structural_scope"))))
            ]
            if explicitly_scoped:
                rows = explicitly_scoped
            else:
                inferred = [r for r in rows if scope_pattern.search(_row_structure_text(r))]
                if inferred:
                    rows = inferred
                elif strict_scope:
                    return []

        k = top_k or self.top_k
        corpus = [
            _normalize_structure(
                "\n".join(
                    part for part in (
                        str(r.get("structural_scope") or ""),
                        str(r.get("relative_path") or ""),
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
        scope = f"; scope={r['structural_scope']}" if r.get("structural_scope") else ""
        path = f"; path={r['relative_path']}" if r.get("relative_path") else ""
        blocks.append(
            f"[SOURCE {i}: {r['filename']}{page}; type={r['source_type']}; authority={r['authority']}{scope}{path}]\n{r['text']}"
        )
    return "\n\n".join(blocks)
