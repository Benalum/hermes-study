from pathlib import Path

import pytest

from hermes_study.db import StudyDB
from hermes_study.ingest import DocumentIngestor
from hermes_study.retrieval import Retriever
from hermes_study.tutor import _explicit_scope


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


@pytest.mark.asyncio
async def test_folder_scope_is_authoritative_even_with_vague_filename(tmp_path: Path):
    db = StudyDB(tmp_path / "study.sqlite3")
    course = db.create_course("General Chemistry I", "CHEM 1215")
    ingestor = DocumentIngestor(db, tmp_path / "uploads")

    chapter_two = tmp_path / "Naming Compounds Handout.txt"
    chapter_two.write_text(
        "Ionic compounds, molecular compounds, nomenclature, ions, and chemical formulas.",
        encoding="utf-8",
    )
    later = tmp_path / "CHEM1215_CH6A.txt"
    later.write_text(
        "Gas laws, pressure, temperature, volume, and ideal gas relationships.",
        encoding="utf-8",
    )

    await ingestor.ingest(
        course["id"],
        chapter_two,
        "professor",
        relative_path="Chapter 2 Material/Naming Compounds Handout.txt",
        structural_scope="chapter 2",
    )
    await ingestor.ingest(
        course["id"],
        later,
        "other",
        relative_path="Chapter 6 Material/CHEM1215_CH6A.txt",
        structural_scope="chapter 6",
    )

    rows = await Retriever(db, top_k=10).search(
        course["id"],
        "Teach me chapter two from the beginning",
        top_k=10,
        scope="chapter 2",
        strict_scope=True,
    )

    assert rows
    assert {row["structural_scope"] for row in rows} == {"chapter 2"}
    assert all("CH6" not in row["filename"].upper() for row in rows)


def test_spoken_chapter_words_are_canonicalized():
    assert _explicit_scope("Teach me chapter two from the beginning") == "chapter 2"
    assert _explicit_scope("Study ch two") == "chapter 2"
    assert _explicit_scope("Walk me through week three") == "week 3"
