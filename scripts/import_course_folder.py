#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from hermes_study.config import Settings
from hermes_study.db import StudyDB
from hermes_study.ingest import DocumentIngestor

SUPPORTED = {".pdf", ".docx", ".txt", ".md", ".csv", ".json"}


def classify_source(path: Path, root: Path) -> str:
    rel = str(path.relative_to(root)).lower()
    name = path.name.lower()
    text = f"{rel} {name}"
    if "syllabus" in text:
        return "syllabus"
    if any(k in text for k in ("homework", "assignment", "worksheet", "problem set", "problem_set", "/hw", " hw")):
        return "homework"
    if any(k in text for k in ("lecture", "professor", "instructor", "handout", "slides", "class material")):
        return "professor"
    if any(k in text for k in ("textbook", "chapter", "ebook")):
        return "textbook"
    if any(k in text for k in ("my notes", "notes", "notebook")):
        return "notes"
    if any(k in text for k in ("formula", "reference", "study guide", "study_guide", "cheat sheet", "periodic table")):
        return "reference"
    return "other"


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
    skipped = 0
    failed = 0
    print(f"Found {len(files)} supported files under {root}")

    for path in files:
        source_type = args.source_type or classify_source(path, root)
        try:
            before = {d['sha256'] for d in db.list_documents(course['id'])}
            result = await ingestor.ingest(course['id'], path, source_type)
            after_new = result.get('sha256') not in before
            if after_new:
                imported += 1
                status = "IMPORTED"
            else:
                skipped += 1
                status = "SKIPPED duplicate"
            rel = path.relative_to(root)
            print(f"[{status}] {rel}  -> {source_type} ({result.get('chunk_count', 0)} chunks)")
        except Exception as exc:
            failed += 1
            print(f"[FAILED] {path.relative_to(root)}: {exc}")

    print()
    print(f"Done. Imported={imported}, duplicates={skipped}, failed={failed}")
    print(f"Course now has {len(db.list_documents(course['id']))} indexed documents.")
    return 1 if failed else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Recursively import a class-material folder into Hermes Study.")
    parser.add_argument("folder", help="Folder containing class material")
    parser.add_argument("--course", required=True, help="Course name to create/use, e.g. 'General Chemistry 1'")
    parser.add_argument("--code", default="", help="Optional course code, e.g. CHEM 1215")
    parser.add_argument("--description", default="", help="Optional course description")
    parser.add_argument(
        "--source-type",
        choices=["professor", "syllabus", "homework", "textbook", "notes", "reference", "other"],
        default=None,
        help="Force one source type for every file instead of auto-classifying by path/name",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
