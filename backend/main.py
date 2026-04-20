"""FastAPI app — HTTP-only so it can be deployed to Vercel serverless
(Python runtime) OR run locally via `uvicorn backend.main:app`.

Endpoints:
  GET  /api/health       - service status
  POST /api/transcribe   - multipart WAV upload -> {transcript, suggestions}
  POST /api/suggest      - ?q=<text>             -> {suggestions}

Anything else is served from frontend/ as static files (single index.html).
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import sarvam_client, sarvam_llm, suggester
from .romanize import romanize

_FRONTEND_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "frontend")
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if sarvam_client.is_available():
        print(f"✅ Sarvam STT enabled (mode={os.environ.get('SARVAM_MODE','transcribe')}, "
              f"model={os.environ.get('SARVAM_MODEL','saarika:v2.5')})")
    else:
        print(f"⚠️  Sarvam STT disabled: {sarvam_client.disabled_reason()}")
    if sarvam_llm.is_available():
        print(f"✅ Sarvam LLM enabled (model={os.environ.get('SARVAM_LLM_MODEL','sarvam-m')})")
    else:
        print(f"⚠️  Sarvam LLM disabled: {sarvam_llm.disabled_reason()}")
    yield


app = FastAPI(title="Tamil Voice → English Name", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _final_transcript(raw: str) -> str:
    if not raw:
        return ""
    return raw if sarvam_client.returns_latin() else romanize(raw)


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "sarvam_stt_available": sarvam_client.is_available(),
        "sarvam_llm_available": sarvam_llm.is_available(),
        "sarvam_stt_model": os.environ.get("SARVAM_MODEL", "saarika:v2.5"),
        "sarvam_stt_mode": os.environ.get("SARVAM_MODE", "transcribe"),
        "sarvam_llm_model": os.environ.get("SARVAM_LLM_MODEL", "sarvam-m"),
    }


@app.post("/api/suggest")
async def http_suggest(q: str = "", k: int = 4) -> JSONResponse:
    result = await asyncio.to_thread(suggester.suggest, q, k)
    return JSONResponse(result)


@app.post("/api/transcribe")
async def http_transcribe(audio: UploadFile = File(...), k: int = 4) -> JSONResponse:
    wav_bytes = await audio.read()
    if not wav_bytes:
        return JSONResponse({"error": "empty_audio"}, status_code=400)
    if not sarvam_client.is_available():
        return JSONResponse(
            {"error": "sarvam_unavailable",
             "detail": sarvam_client.disabled_reason()},
            status_code=503,
        )
    raw = await asyncio.to_thread(
        sarvam_client.transcribe_wav, wav_bytes, audio.content_type or "audio/wav"
    )
    transcript = _final_transcript(raw)
    if not transcript:
        return JSONResponse({"transcript": "", "suggestions": [],
                             "error": "no_speech_detected"})
    result = await asyncio.to_thread(suggester.suggest, transcript, k)
    return JSONResponse({
        "transcript": transcript,
        "suggestions": result.get("suggestions", []),
        "sources": result.get("sources", {}),
    })


# Legacy routes kept so existing frontend JS doesn't break during migration.
@app.get("/health")
async def _legacy_health() -> dict[str, Any]:
    return await health()


@app.post("/suggest")
async def _legacy_suggest(q: str = "", k: int = 4) -> JSONResponse:
    return await http_suggest(q=q, k=k)


if os.path.isdir(_FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")
