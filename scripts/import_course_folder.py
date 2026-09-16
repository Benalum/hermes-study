#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import re
from pathlib import Path

from hermes_study.config import Settings
from hermes_study.db import StudyDB
from hermes_study.ingest import DocumentIngestor

SUPPORTED = {".pdf", ".docx", ".txt", ".md", ".csv", ".json"}

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
}

SOURCE_LABELS = {
    "professor": "Professor",
    "syllabus": "Syllabus",
    "homework": "Homework",
    "textbook": "Textbook",
    "notes": "Notes",
    "reference": "Reference",
    "other": "Course Material",
}


def _number_token(value: str) -> int | None:
    value = value.strip().lower()
    if value.isdigit():
        return int(value)
    return NUMBER_WORDS.get(value)


def infer_structural_scope(path: Path, root: Path) -> str:
    """Infer chapter/week from the folder tree first, then the filename.

    The user's directory organization is authoritative course structure. A file
    under `Chapter 2 Material/` belongs to Chapter 2 even when its filename is
    vague, e.g. `Naming Compounds Handout.pdf`.
    """
    rel = path.relative_to(root)
    candidates = list(rel.parent.parts) + [path.stem]
    word_pattern = "|".join(NUMBER_WORDS)
    for value in candidates:
        text = value.lower().replace("_", " ").replace("-", " ")
        chapter = re.search(rf"\b(?:chapter|ch)\s*({word_pattern}|\d+)\b", text, flags=re.I)
        if chapter:
            number = _number_token(chapter.group(1))
            if number is not None:
                return f"chapter {number}"
        week = re.search(rf"\bweek\s*({word_pattern}|\d+)\b", text, flags=re.I)
        if week:
            number = _number_token(week.group(1))
            if number is not None:
                return f"week {number}"
    return ""


def classify_source(path: Path, root: Path) -> str:
    """Classify what a file *is* independently of which chapter folder contains it."""
    name = path.name.lower()
    parent = " ".join(path.relative_to(root).parent.parts).lower()

    if "syllabus" in name:
        return "syllabus"
    if any(k in name for k in ("slides", "lecture", "professor", "instructor", "class material", "handout")):
        return "professor"
    if any(k in name for k in (
        "homework", "assignment", "worksheet", "problem set", "problem_set",
        "practice problems", "practice problem", "answer key", "answers",
    )):
        return "homework"
    if any(k in name for k in ("openstax", "chemistry-2e", "textbook", "ebook")):
        return "textbook"
    if any(k in name for k in ("my notes", "notes", "notebook")):
        return "notes"
    if any(k in name for k in (
        "formula", "reference", "study guide", "study_guide", "cheat sheet",
        "periodic table", "conversion", "constants",
    )):
        return "reference"

    # Explicit source folders can still provide type context, but generic
    # `Chapter N Material` folders never determine source type.
    if any(k in parent for k in ("professor slides", "lecture slides", "instructor material")):
        return "professor"
    if any(k in parent for k in ("homework", "worksheets", "assignments")):
        return "homework"
    if any(k in parent for k in ("textbook", "openstax")):
        return "textbook"
    if "notes" in parent:
        return "notes"
    return "other"


def display_name(path: Path, root: Path, source_type: str, scope: str) -> str:
    """Create a readable UI label while preserving the real filename separately."""
    stem = path.stem
    cleaned = re.sub(r"(?i)\bchem\s*1215\b", "", stem)
    cleaned = re.sub(r"(?i)\bch(?:apter)?\s*\d+[a-z]?\b", "", cleaned)
    cleaned = re.sub(r"(?i)\bopenstax\b", "", cleaned)
    cleaned = re.sub(r"[_-]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .-_") or stem
    prefix = scope.title() if scope else "Course"
    return f"{prefix} · {SOURCE_LABELS.get(source_type, 'Course Material')} · {cleaned}"


async def run(args: argparse.Namespace) -> int:
    root = Path(args.folder).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Folder not found: {root}")

    settings = Settings()
    settings.ensure_dirs()
    db = StudyDB(settings.db_path)

    course = db.find_course(args.course)
    if course is None:
        course = db.create_course(args.course, args.code or "", args.description or "")
        print(f"Created course: {course['name']} ({course.get('code') or 'no code'})")
    else:
        print(f"Using existing course: {course['name']} ({course.get('code') or 'no code'})")

    ingestor = DocumentIngestor(db, settings.upload_dir)
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)
    if not files:
        print(f"No supported files found under {root}")
        return 0

    imported = 0
    updated = 0
    failed = 0
    print(f"Found {len(files)} supported files under {root}")

    for path in files:
        source_type = args.source_type or classify_source(path, root)
        scope = infer_structural_scope(path, root)
        rel = path.relative_to(root).as_posix()
        label = display_name(path, root, source_type, scope)
        try:
            before = {d['sha256'] for d in db.list_documents(course['id'])}
            result = await ingestor.ingest(
                course['id'],
                path,
                source_type,
                relative_path=rel,
                structural_scope=scope,
            )
            is_new = result.get('sha256') not in before
            if is_new:
                imported += 1
                status = "IMPORTED"
            else:
                updated += 1
                status = "ORGANIZED"
            scope_text = scope or "unscoped"
            print(f"[{status}] {rel} -> {scope_text} / {source_type} / {label} ({result.get('chunk_count', 0)} chunks)")
        except Exception as exc:
            failed += 1
            print(f"[FAILED] {rel}: {exc}")

    print()
    print(f"Done. Imported={imported}, organized={updated}, failed={failed}")
    print(f"Course now has {len(db.list_documents(course['id']))} indexed documents.")
    print("Existing readable documents are metadata-updated without re-chunking.")
    return 1 if failed else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recursively import and organize a class-material folder into Hermes Study."
    )
    parser.add_argument("folder", help="Folder containing class material")
    parser.add_argument("--course", required=True, help="Course name to create/use, e.g. 'General Chemistry I for STEM Majors'")
    parser.add_argument("--code", default="", help="Optional course code, e.g. CHEM 1215")
    parser.add_argument("--description", default="", help="Optional course description")
    parser.add_argument(
        "--source-type",
        choices=["professor", "syllabus", "homework", "textbook", "notes", "reference", "other"],
        default=None,
        help="Force one source type for every file instead of auto-classifying by filename/folder",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
