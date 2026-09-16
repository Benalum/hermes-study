# Hermes Study

Hermes Study is a voice-first study assistant built around a **local Ollama LLM**, your own class material, and a private browser UI that can be shared to your devices with **Tailscale Serve**.

## What v0.1 does

- Create multiple courses and say/type **“Study Chemistry 1”** to start the matching course.
- Upload PDF, DOCX, TXT, Markdown, CSV, or JSON material.
- Mark sources as professor material, syllabus, homework, textbook, notes, reference, or other.
- Prefer higher-authority class sources when relevance is similar.
- Use Ollama `/api/embed` for semantic RAG retrieval, with local TF-IDF fallback if the embedding model is unavailable.
- Run an adaptive oral study loop: ask a question, grade the answer, explain mistakes, ask the next question, and track mastery by topic.
- Record microphone audio in the browser and transcribe it on the Hermes Study host with Faster-Whisper.
- Speak Hermes replies with the browser/device speech engine.
- Optionally verify the speaker with SpeechBrain ECAPA before transcription.
- Use an Ollama local chat model through `/api/chat`.
- Optionally use Ollama's hosted web-search API when `OLLAMA_API_KEY` is configured.
- Bind only to `127.0.0.1` by default and expose the UI to authorized tailnet devices with Tailscale Serve.

## Architecture

```text
Phone / Mac / tablet browser
        │
        │ tailnet-only HTTPS (Tailscale Serve)
        ▼
Hermes Study FastAPI @ 127.0.0.1:8787
        ├── Course/document store (SQLite + files)
        ├── PDF/DOCX/text ingestion
        ├── Retriever
        │     ├── Ollama embeddings
        │     └── TF-IDF fallback
        ├── Adaptive tutor / mastery tracking
        ├── Voice
        │     ├── optional speaker gate
        │     └── Faster-Whisper STT
        └── Ollama local LLM
```

The backend deliberately binds to loopback. Tailscale Serve reverse-proxies that local service over HTTPS to devices authorized by your tailnet, so Hermes Study does not need an open LAN port.

## MacBook quick start

```bash
git clone https://github.com/Benalum/hermes-study.git
cd hermes-study
bash scripts/install_mac.sh
```

The installer creates `.venv`, installs Hermes Study plus Faster-Whisper, installs `ffmpeg` with Homebrew when available, creates `.env`, and pulls the embedding model when Ollama is installed.

Edit `.env` if your local model name differs. Defaults:

```bash
OLLAMA_CHAT_MODEL=gemma4:31b-mlx
OLLAMA_EMBED_MODEL=nomic-embed-text
```

Make sure Ollama is running, then:

```bash
./scripts/run_mac.sh
```

Open locally:

```text
http://127.0.0.1:8787
```

If Tailscale is installed, `run_mac.sh` also tries:

```bash
tailscale serve --bg localhost:8787
```

Use this to see the tailnet HTTPS URL:

```bash
tailscale serve status
```

Then open that `https://...ts.net` URL from your phone, tablet, or another computer logged into your tailnet.

## First Chemistry setup

1. Open Hermes Study.
2. Add a course such as **Chemistry 1** / **CHEM 1215**.
3. Upload the syllabus and mark it **Syllabus**.
4. Upload professor handouts and mark them **Professor material**.
5. Upload homework, textbook pages, notes, and reference sheets with the appropriate source type.
6. Click **Study selected course**, or use the microphone and say **“Study Chemistry 1.”**

During a study session, spoken/typed answers are graded for conceptual correctness rather than exact wording. Hermes records mastery by topic and includes weaker topics in future planning.

## Source authority

When relevance is similar, Hermes prefers:

1. Professor material
2. Syllabus
3. Homework / worksheets
4. Assigned textbook
5. Your notes
6. Reference material
7. Other material

This is meant to keep the tutor aligned with what your actual class expects rather than letting generic outside material override course-specific instructions.

## Voice and speaker filtering

The standard install includes Faster-Whisper STT. Browser microphone audio is sent to the Mac running Hermes Study and transcribed there.

To add the heavier SpeechBrain/PyTorch speaker-verification dependency:

```bash
INSTALL_SPEAKER_GATE=1 bash scripts/install_mac.sh
```

Then set:

```bash
HERMES_STUDY_SPEAKER_GATE=true
HERMES_STUDY_SPEAKER_THRESHOLD=0.72
```

Start Hermes Study, click **Enroll my voice sample**, speak naturally for several seconds, and stop recording. With the gate enabled, later microphone turns are checked against the enrolled reference before STT.

The threshold is configurable because microphones, rooms, and speaking distance vary. Higher values are stricter.

## Optional external research

Set `OLLAMA_API_KEY` to enable `/api/research?q=...`, which calls Ollama's hosted web-search API. External results are supplemental; class material remains the authority for assignments, deadlines, professor instructions, and class-specific conventions.

## Tests

```bash
python -m pip install -e '.[test]'
pytest -q
```

The core tests do not need a running Ollama instance or a downloaded speech model.

Current local validation for v0.1: **5 tests passing** covering chunking, ingestion, fallback retrieval, course lookup, `Study Chemistry 1` command routing, adaptive grading/mastery updates, and API smoke behavior.

## Start automatically at login

After the normal install works:

```bash
./scripts/install_launch_agent.sh
```

Logs are written under `data/logs/`.

## Hermes Voice compatibility

This repository is intentionally separate from `hermes-voice`, but its boundaries mirror the existing Hermes design:

- voice/browser input becomes text turns;
- `Tutor.command()` handles `study <course>` intent;
- `Tutor.study_turn()` owns active study-state transitions;
- Ollama access is isolated behind `OllamaClient`;
- STT/speaker verification are isolated under `voice.py`.

A later Hermes Voice adapter can delegate study commands to this service without copying the course RAG/database into the voice repo.
