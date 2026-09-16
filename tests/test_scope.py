from pathlib import Path

import pytest

from hermes_study.db import StudyDB
from hermes_study.ingest import DocumentIngestor
from hermes_study.retrieval import Retriever


@pytest.mark.asyncio
async def test_chapter_one_scope_excludes_chapter_nine(tmp_path: Path):
    db = StudyDB(tmp_path / "study.sqlite3")
    course = db.create_course("General Chemistry I", "CHEM 1215")
    upload_dir = tmp_path / "uploads"
    ingestor = DocumentIngestor(db, upload_dir)

    ch1 = tmp_path / "CHEM1215_CH1C_OpenStax.txt"
    ch1.write_text(
        "Chapter one covers measurements, units, density, dimensional analysis, and significant figures.",
        encoding="utf-8",
    )
    ch9 = tmp_path / "CHEM 1215 Ch9A Slides.txt"
    ch9.write_text(
        "Chapter nine covers electronegativity, electron affinity, periodic trends, and gas-phase atoms.",
        encoding="utf-8",
    )

    await ingestor.ingest(course["id"], ch1, "textbook")
    await ingestor.ingest(course["id"], ch9, "professor")

    retriever = Retriever(db, top_k=10)
    rows = await retriever.search(
        course["id"],
        "chapter 1 important concepts",
        top_k=10,
        scope="chapter 1",
        strict_scope=True,
    )

    assert rows
    assert all("CH1" in row["filename"].upper() for row in rows)
    assert all("CH9" not in row["filename"].upper().replace(" ", "") for row in rows)
