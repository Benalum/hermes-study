from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_study.db import StudyDB
from hermes_study.ingest import DocumentIngestor, chunk_text
from hermes_study.retrieval import Retriever
from hermes_study.tutor import Tutor


class FakeLLM:
    def __init__(self):
        self.chat_calls = 0

    async def embed(self, texts):
        raise RuntimeError("embedding model intentionally unavailable in test")

    async def chat(self, messages, *, json_mode=False):
        self.chat_calls += 1
        if self.chat_calls == 1:
            return json.dumps({
                "intro": "Starting Chemistry 1.",
                "topic": "density",
                "question": "A sample has mass 10 g and volume 2 mL. What is its density?",
            })
        return json.dumps({
            "score": 1.0,
            "feedback": "Correct: density is mass divided by volume, so 5 g/mL.",
            "next_topic": "significant figures",
            "next_question": "How many significant figures are in 0.0500?",
        })


@pytest.fixture()
def db(tmp_path: Path):
    return StudyDB(tmp_path / "study.sqlite3")


def test_chunk_text_overlaps_and_preserves_content():
    text = "Sentence one. " * 400
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) > 2
    assert "Sentence one" in chunks[0]
    assert all(len(c) <= 520 for c in chunks)


@pytest.mark.asyncio
async def test_ingest_and_lexical_retrieval(db: StudyDB, tmp_path: Path):
    course = db.create_course("Chemistry 1", "CHEM 1215")
    doc = tmp_path / "notes.txt"
    doc.write_text(
        "Density is mass divided by volume. Units include grams per milliliter.\n\n"
        "Significant figures communicate measurement precision. Leading zeros are not significant.",
        encoding="utf-8",
    )
    llm = FakeLLM()
    ingestor = DocumentIngestor(db, llm, tmp_path / "uploads")
    result = await ingestor.ingest(course["id"], doc, "professor")
    assert result["chunk_count"] >= 1
    retriever = Retriever(db, llm, top_k=3)
    rows = await retriever.search(course["id"], "How do I calculate density?")
    assert rows
    assert "Density" in rows[0]["text"]
    assert rows[0]["source_type"] == "professor"


@pytest.mark.asyncio
async def test_study_command_grades_and_tracks_mastery(db: StudyDB, tmp_path: Path):
    course = db.create_course("Chemistry 1", "CHEM 1215")
    notes = tmp_path / "chem.txt"
    notes.write_text(
        "Density equals mass divided by volume. Significant figures represent measurement precision. "
        "For 0.0500, leading zeros are not significant and trailing zeros after the decimal are significant.",
        encoding="utf-8",
    )
    llm = FakeLLM()
    ingestor = DocumentIngestor(db, llm, tmp_path / "uploads")
    await ingestor.ingest(course["id"], notes, "professor")
    tutor = Tutor(db, llm, Retriever(db, llm, top_k=4))

    started = await tutor.command("Hermes, study Chemistry 1")
    assert started["handled"] is True
    assert started["course"]["id"] == course["id"]
    assert started["topic"] == "density"

    turn = await tutor.study_turn(started["session"]["id"], "5 grams per milliliter")
    assert turn["score"] == 1.0
    assert turn["topic"] == "significant figures"
    mastery = db.mastery_for_course(course["id"])
    assert mastery[0]["topic"] == "density"
    assert mastery[0]["score"] == 1.0


def test_course_lookup_is_flexible(db: StudyDB):
    course = db.create_course("General Chemistry I", "CHEM 1215")
    assert db.find_course("CHEM 1215")["id"] == course["id"]
    assert db.find_course("General Chemistry")["id"] == course["id"]
