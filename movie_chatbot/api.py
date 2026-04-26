"""FastAPI server for the movie chatbot."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from movie_chatbot.agent import run
from movie_chatbot.services import data_fetcher as tmdb
from movie_chatbot.services import vector_engine as vdb

ROOT = Path(__file__).resolve().parents[1]

# In-memory session store mapping session IDs to conversation history.
_SESSIONS: dict[str, list[dict]] = {}


class ChatRequest(BaseModel):
    query: str = Field(min_length=1)
    session_id: str | None = None
    history: list[dict] | None = None


class ChatResponse(BaseModel):
    session_id: str
    response: str
    history: list[dict]


def ensure_db_ready() -> None:
    """Ensure the movie dataset and vector database are available before serving requests."""
    collection = vdb.get_collection()
    if collection and collection.count() > 0:
        return

    data_path = ROOT / "movies_data.json"
    if not data_path.exists():
        tmdb.fetch_and_save_diverse_movies(str(data_path))
    vdb.setup_vector_db(str(data_path))


app = FastAPI(title="Movie Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event() -> None:
    # Prepare data and vector store before accepting any requests.
    ensure_db_ready()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    # Use an existing session ID if provided; otherwise create a new one.
    session_id = payload.session_id or uuid.uuid4().hex
    if session_id in _SESSIONS:
        history = _SESSIONS[session_id]
    else:
        history = payload.history or []

    response, updated_history = run(payload.query.strip(), history)
    _SESSIONS[session_id] = updated_history
    return ChatResponse(session_id=session_id, response=response, history=updated_history)


@app.post("/reset")
def reset_session(payload: dict) -> dict:
    session_id = str(payload.get("session_id") or "").strip()
    if session_id and session_id in _SESSIONS:
        # Remove stored conversation history for this session.
        _SESSIONS.pop(session_id, None)
        return {"ok": True}
    return {"ok": False}

