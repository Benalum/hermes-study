from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS courses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    slug TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    code TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS course_settings (
    course_id INTEGER NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(course_id, key)
);
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    course_id INTEGER NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    source_type TEXT NOT NULL,
    authority INTEGER NOT NULL DEFAULT 60,
    sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(course_id, sha256)
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    course_id INTEGER NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    page INTEGER,
    heading TEXT NOT NULL DEFAULT '',
    text TEXT NOT NULL,
    embedding TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_course ON chunks(course_id);
CREATE TABLE IF NOT EXISTS study_sessions (
    id TEXT PRIMARY KEY,
    course_id INTEGER NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    mode TEXT NOT NULL,
    status TEXT NOT NULL,
    current_topic TEXT NOT NULL DEFAULT '',
    current_question TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES study_sessions(id) ON DELETE CASCADE,
    course_id INTEGER NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    topic TEXT NOT NULL,
    question TEXT NOT NULL,
    learner_answer TEXT NOT NULL,
    score REAL NOT NULL,
    feedback TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS mastery (
    course_id INTEGER NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    topic TEXT NOT NULL,
    score REAL NOT NULL,
    attempts INTEGER NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(course_id, topic)
);
"""


class StudyDB:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as con:
            con.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.path)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys=ON")
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def create_course(self, name: str, code: str = "", description: str = "") -> dict[str, Any]:
        slug = self._slugify(code or name)
        with self.connect() as con:
            cur = con.execute(
                "INSERT INTO courses(slug,name,code,description,created_at) VALUES(?,?,?,?,?)",
                (slug, name.strip(), code.strip(), description.strip(), utcnow()),
            )
            row = con.execute("SELECT * FROM courses WHERE id=?", (cur.lastrowid,)).fetchone()
        return dict(row)

    def list_courses(self) -> list[dict[str, Any]]:
        with self.connect() as con:
            rows = con.execute("SELECT * FROM courses ORDER BY name COLLATE NOCASE").fetchall()
        return [dict(r) for r in rows]

    def get_course(self, course_id: int) -> dict[str, Any] | None:
        with self.connect() as con:
            row = con.execute("SELECT * FROM courses WHERE id=?", (course_id,)).fetchone()
        return dict(row) if row else None

    def find_course(self, query: str) -> dict[str, Any] | None:
        q = query.strip().lower()
        courses = self.list_courses()
        for c in courses:
            if q in {c["name"].lower(), c["code"].lower(), c["slug"].lower()}:
                return c
        for c in courses:
            hay = f"{c['name']} {c['code']} {c['slug']}".lower()
            if q in hay or hay in q:
                return c
        return None

    def set_course_setting(self, course_id: int, key: str, value: str) -> None:
        value = value.strip()
        with self.connect() as con:
            if value:
                con.execute(
                    "INSERT INTO course_settings(course_id,key,value,updated_at) VALUES(?,?,?,?) "
                    "ON CONFLICT(course_id,key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                    (course_id, key, value, utcnow()),
                )
            else:
                con.execute(
                    "DELETE FROM course_settings WHERE course_id=? AND key=?",
                    (course_id, key),
                )

    def get_course_setting(self, course_id: int, key: str, default: str = "") -> str:
        with self.connect() as con:
            row = con.execute(
                "SELECT value FROM course_settings WHERE course_id=? AND key=?",
                (course_id, key),
            ).fetchone()
        return str(row["value"]) if row else default

    def add_document(self, *, course_id: int, filename: str, stored_path: str, source_type: str,
                     authority: int, sha256: str) -> dict[str, Any]:
        with self.connect() as con:
            existing = con.execute(
                "SELECT * FROM documents WHERE course_id=? AND sha256=?", (course_id, sha256)
            ).fetchone()
            if existing:
                return dict(existing)
            cur = con.execute(
                "INSERT INTO documents(course_id,filename,stored_path,source_type,authority,sha256,created_at) VALUES(?,?,?,?,?,?,?)",
                (course_id, filename, stored_path, source_type, authority, sha256, utcnow()),
            )
            row = con.execute("SELECT * FROM documents WHERE id=?", (cur.lastrowid,)).fetchone()
        return dict(row)

    def get_document(self, document_id: int) -> dict[str, Any] | None:
        with self.connect() as con:
            row = con.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
        return dict(row) if row else None

    def list_documents(self, course_id: int) -> list[dict[str, Any]]:
        with self.connect() as con:
            rows = con.execute(
                "SELECT d.*, COUNT(c.id) AS chunk_count FROM documents d LEFT JOIN chunks c ON c.document_id=d.id "
                "WHERE d.course_id=? GROUP BY d.id ORDER BY d.created_at DESC", (course_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def replace_chunks(self, document_id: int, course_id: int, chunks: list[dict[str, Any]]) -> None:
        with self.connect() as con:
            con.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
            con.executemany(
                "INSERT INTO chunks(document_id,course_id,ordinal,page,heading,text,embedding,created_at) VALUES(?,?,?,?,?,?,?,?)",
                [(
                    document_id, course_id, idx, ch.get("page"), ch.get("heading", ""), ch["text"],
                    json.dumps(ch["embedding"]) if ch.get("embedding") is not None else None, utcnow()
                ) for idx, ch in enumerate(chunks)]
            )

    def chunks_for_course(self, course_id: int) -> list[dict[str, Any]]:
        with self.connect() as con:
            rows = con.execute(
                "SELECT c.*, d.filename, d.source_type, d.authority FROM chunks c JOIN documents d ON d.id=c.document_id "
                "WHERE c.course_id=?", (course_id,)
            ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["embedding"] = json.loads(item["embedding"]) if item["embedding"] else None
            out.append(item)
        return out

    def start_session(self, course_id: int, mode: str, topic: str, question: str) -> dict[str, Any]:
        sid = str(uuid.uuid4())
        now = utcnow()
        with self.connect() as con:
            con.execute(
                "INSERT INTO study_sessions(id,course_id,mode,status,current_topic,current_question,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?,?)", (sid, course_id, mode, "active", topic, question, now, now)
            )
        return self.get_session(sid)  # type: ignore[return-value]

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self.connect() as con:
            row = con.execute("SELECT * FROM study_sessions WHERE id=?", (session_id,)).fetchone()
        return dict(row) if row else None

    def update_session_question(self, session_id: str, topic: str, question: str) -> None:
        with self.connect() as con:
            con.execute(
                "UPDATE study_sessions SET current_topic=?,current_question=?,updated_at=? WHERE id=?",
                (topic, question, utcnow(), session_id),
            )

    def record_attempt(self, session_id: str, course_id: int, topic: str, question: str,
                       learner_answer: str, score: float, feedback: str) -> None:
        score = max(0.0, min(1.0, float(score)))
        with self.connect() as con:
            con.execute(
                "INSERT INTO attempts(session_id,course_id,topic,question,learner_answer,score,feedback,created_at) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (session_id, course_id, topic, question, learner_answer, score, feedback, utcnow()),
            )
            current = con.execute(
                "SELECT score,attempts FROM mastery WHERE course_id=? AND topic=?", (course_id, topic)
            ).fetchone()
            if current:
                attempts = int(current["attempts"]) + 1
                new_score = (float(current["score"]) * (attempts - 1) + score) / attempts
                con.execute(
                    "UPDATE mastery SET score=?,attempts=?,updated_at=? WHERE course_id=? AND topic=?",
                    (new_score, attempts, utcnow(), course_id, topic),
                )
            else:
                con.execute(
                    "INSERT INTO mastery(course_id,topic,score,attempts,updated_at) VALUES(?,?,?,?,?)",
                    (course_id, topic, score, 1, utcnow()),
                )

    def mastery_for_course(self, course_id: int) -> list[dict[str, Any]]:
        with self.connect() as con:
            rows = con.execute(
                "SELECT * FROM mastery WHERE course_id=? ORDER BY score ASC, attempts DESC", (course_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    @staticmethod
    def _slugify(text: str) -> str:
        import re
        slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "course"
        return slug
