from __future__ import annotations

import argparse

import uvicorn

from .config import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Hermes Study")
    parser.add_argument("--host", default=None, help="Bind host; defaults to HERMES_STUDY_HOST or 127.0.0.1")
    parser.add_argument("--port", type=int, default=None, help="Bind port; defaults to HERMES_STUDY_PORT or 8787")
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    settings = Settings()
    uvicorn.run("hermes_study.app:app", host=args.host or settings.host, port=args.port or settings.port, reload=args.reload)


if __name__ == "__main__":
    main()
