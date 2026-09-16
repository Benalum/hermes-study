from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

from docx import Document
from pypdf import PdfReader

from .db import StudyDB
from .llm import OllamaClient

AUTHORITY = {
    "professor": 100,
    "syllabus": 95,
    "homework": 90,
    "textbook": 85,
    "notes": 75,
    "reference": 65,
    "other": 60,
}


def chunk_text(text: str, *, chunk_size: int = 1800, overlap: int = 250) -> list[str]:
    clean = "\n".join(line.rstrip() for line in text.replace("\r", "").split("\n"))
    clean = "\n".join(line for line in clean.split("\n") if line.strip())
    if not clean.strip():
        return []
    chunks: list[str] = []
    start = 0
    while start < len(clean):
        end = min(len(clean), start + chunk_size)
        if end < len(clean):
            boundary = max(clean.rfind("\n", start, end), clean.rfind(". ", start, end))
            if boundary > start + chunk_size // 2:
                end = boundary + 1
        piece = clean[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(clean):
            break
        start = max(start + 1, end - overlap)
    return chunks


def extract_sections(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        sections = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                sections.append({"page": i, "text": text})
        return sections
    if suffix == ".docx":
        doc = Document(str(path))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return [{"page": None, "text": text}]
    if suffix in {".txt", ".md", ".csv", ".py", ".json"}:
        return [{"page": None, "text": path.read_text(encoding="utf-8", errors="replace")}]
    raise ValueError(f"Unsupported file type: {suffix}. Supported: PDF, DOCX, TXT, MD, CSV, JSON.")


class DocumentIngestor:
    def __init__(self, db: StudyDB, llm: OllamaClient, upload_dir: Path):
        self.db = db
        self.llm = llm
        self.upload_dir = Path(upload_dir)

    async def ingest(self, course_id: int, source_path: Path, source_type: str = "other") -> dict[str, Any]:
        source_path = Path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        target_dir = self.upload_dir / str(course_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        safe_name = source_path.name.replace("/", "_")
        stored = target_dir / f"{digest[:12]}-{safe_name}"
        if source_path.resolve() != stored.resolve():
            shutil.copy2(source_path, stored)
        doc = self.db.add_document(
            course_id=course_id,
            filename=source_path.name,
            stored_path=str(stored),
            source_type=source_type,
            authority=AUTHORITY.get(source_type, AUTHORITY["other"]),
            sha256=digest,
        )
        existing = self.db.list_documents(course_id)
        same = next((d for d in existing if d["id"] == doc["id"]), None)
        if same and int(same["chunk_count"]) > 0:
            return same
        raw_chunks: list[dict[str, Any]] = []
        for section in extract_sections(stored):
            for text in chunk_text(section["text"]):
                raw_chunks.append({"page": section.get("page"), "text": text, "heading": ""})
        if not raw_chunks:
            raise ValueError("No readable text was extracted from the document.")
        try:
            embeddings = await self.llm.embed([c["text"] for c in raw_chunks])
            for ch, emb in zip(raw_chunks, embeddings, strict=False):
                ch["embedding"] = emb
        except Exception:
            pass
        self.db.replace_chunks(doc["id"], course_id, raw_chunks)
        return next(d for d in self.db.list_documents(course_id) if d["id"] == doc["id"])
