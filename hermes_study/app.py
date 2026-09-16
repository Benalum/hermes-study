from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from .config import Settings
from .db import StudyDB
from .ingest import AUTHORITY, DocumentIngestor
from .llm import HermesClient
from .retrieval import Retriever
from .tutor import Tutor
from .voice import SpeakerGate, VoiceUnavailable, WhisperSTT, save_upload_bytes


class CourseCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    code: str = Field(default="", max_length=80)
    description: str = Field(default="", max_length=2000)


class AskBody(BaseModel):
    text: str = Field(min_length=1, max_length=20000)


class StartBody(BaseModel):
    course_id: int
    mode: str = "adaptive"


class CommandBody(BaseModel):
    text: str = Field(min_length=1, max_length=20000)


class TurnBody(BaseModel):
    text: str = Field(min_length=1, max_length=20000)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    settings.ensure_dirs()
    db = StudyDB(settings.db_path)
    llm = HermesClient(
        settings.hermes_base_url,
        settings.hermes_api_key,
        settings.hermes_model,
        settings.hermes_session_key,
    )
    retriever = Retriever(db, settings.top_k)
    tutor = Tutor(db, llm, retriever)
    ingestor = DocumentIngestor(db, settings.upload_dir)
    stt = WhisperSTT(settings.whisper_model)
    gate = SpeakerGate(settings.voice_dir, settings.speaker_threshold)

    app = FastAPI(title="Hermes Study", version="0.2.0")
    app.state.settings = settings
    app.state.db = db
    app.state.llm = llm
    app.state.tutor = tutor
    app.state.ingestor = ingestor
    app.state.stt = stt
    app.state.speaker_gate = gate

    templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        return templates.TemplateResponse(request=request, name="index.html", context={"title": "Hermes Study"})

    @app.get("/api/health")
    async def health():
        llm_status = await llm.health()
        return {
            "ok": True,
            "hermes": llm_status,
            "ollama": llm_status,
            "speaker_gate_enabled": settings.speaker_gate,
            "speaker_enrolled": gate.enrolled,
            "research_via_hermes": True,
            "data_dir": str(settings.data_dir),
        }

    @app.get("/api/courses")
    async def courses():
        return db.list_courses()

    @app.post("/api/courses")
    async def create_course(body: CourseCreate):
        try:
            return db.create_course(body.name, body.code, body.description)
        except Exception as exc:
            if "UNIQUE constraint" in str(exc):
                raise HTTPException(409, "A course with that name/code already exists.") from exc
            raise

    @app.get("/api/courses/{course_id}")
    async def course_detail(course_id: int):
        course = db.get_course(course_id)
        if not course:
            raise HTTPException(404, "Course not found")
        return {
            **course,
            "documents": db.list_documents(course_id),
            "mastery": db.mastery_for_course(course_id),
        }

    @app.post("/api/courses/{course_id}/documents")
    async def upload_document(
        course_id: int,
        file: Annotated[UploadFile, File(...)],
        source_type: Annotated[str, Form()] = "other",
    ):
        if not db.get_course(course_id):
            raise HTTPException(404, "Course not found")
        if source_type not in AUTHORITY:
            raise HTTPException(400, f"source_type must be one of: {', '.join(AUTHORITY)}")
        suffix = Path(file.filename or "upload.txt").suffix or ".txt"
        temp = save_upload_bytes(await file.read(), suffix=suffix)
        temp_named = temp.with_name((file.filename or temp.name).replace("/", "_"))
        try:
            os.replace(temp, temp_named)
            result = await ingestor.ingest(course_id, temp_named, source_type)
            return result
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(400, str(exc)) from exc
        finally:
            for p in (temp, temp_named):
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass

    @app.post("/api/courses/{course_id}/ask")
    async def ask_course(course_id: int, body: AskBody):
        if not db.get_course(course_id):
            raise HTTPException(404, "Course not found")
        return await tutor.ask(course_id, body.text)

    @app.post("/api/study/start")
    async def start_study(body: StartBody):
        try:
            return await tutor.start_session(body.course_id, body.mode)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/study/{session_id}/turn")
    async def study_turn(session_id: str, body: TurnBody):
        try:
            return await tutor.study_turn(session_id, body.text)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/command")
    async def command(body: CommandBody):
        return await tutor.command(body.text)

    @app.post("/api/voice/enroll")
    async def enroll_voice(file: Annotated[UploadFile, File(...)]):
        path = save_upload_bytes(await file.read(), suffix=Path(file.filename or "voice.webm").suffix or ".webm")
        try:
            gate.enroll(path)
            return {"ok": True, "enrolled": True}
        except VoiceUnavailable as exc:
            raise HTTPException(503, str(exc)) from exc
        finally:
            path.unlink(missing_ok=True)

    @app.post("/api/voice/transcribe")
    async def transcribe(file: Annotated[UploadFile, File(...)]):
        path = save_upload_bytes(await file.read(), suffix=Path(file.filename or "voice.webm").suffix or ".webm")
        try:
            speaker_score = None
            if settings.speaker_gate:
                if not gate.enrolled:
                    raise HTTPException(409, "Speaker gate is enabled but no voice is enrolled yet.")
                passed, speaker_score = gate.verify(path)
                if not passed:
                    raise HTTPException(403, detail={"message": "Speaker rejected", "score": speaker_score})
            text = stt.transcribe(path)
            return {"text": text, "speaker_score": speaker_score}
        except VoiceUnavailable as exc:
            raise HTTPException(503, str(exc)) from exc
        finally:
            path.unlink(missing_ok=True)

    @app.get("/api/research")
    async def research(q: str):
        if not q.strip():
            raise HTTPException(400, "q is required")
        try:
            return {"answer": await llm.research(q.strip())}
        except Exception as exc:
            raise HTTPException(503, f"Hermes research is unavailable: {exc}") from exc

    return app


app = create_app()
