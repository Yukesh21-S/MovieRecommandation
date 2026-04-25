"""Intent classification node."""

from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage

from ..constants import LATEST_KEYWORDS, LITE_MOOD_KEYWORDS, MOOD_TO_GENRE
from ..helpers import extract_title, is_followup
from ..runtime import json_llm
from ..state import State
from ...services import data_fetcher as tmdb


def classify(state: State) -> State:
    query, history = state["query"], state["history"]
    lower_query = query.lower()

    is_latest = bool(LATEST_KEYWORDS & set(lower_query.split()))
    is_lite = bool(LITE_MOOD_KEYWORDS & set(lower_query.split()))

    if is_followup(query, history):
        return {
            **state,
            "intent": "followup",
            "search_title": None,
            "language_code": None,
            "genre_id": None,
            "person": None,
            "is_latest": False,
            "is_lite_mood": False,
            "disambiguation_options": None,
        }

    if lower_query.strip() in {"hi", "hello", "hey", "hola", "namaste"} or (
        "how are you" in lower_query and "movie" not in lower_query
    ):
        return {
            **state,
            "intent": "chat",
            "search_title": None,
            "language_code": None,
            "genre_id": None,
            "person": None,
            "is_latest": is_latest,
            "is_lite_mood": False,
            "disambiguation_options": None,
        }

    lang_list = ", ".join(tmdb.LANGUAGE_MAP)
    genre_list = ", ".join(tmdb.GENRE_MAP)
    prompt = f"""Classify this movie query. Respond ONLY with valid JSON — no markdown, no explanation.

Query: "{query}"

Rules:
- "search"   → user wants info about ONE specific movie by title (e.g. "Find Inception", "Tell me about Parasite")
- "discover" → user wants recommendations by mood, genre, language, or person
- "chat"     → greeting or small talk only

JSON:
{{
  "intent": "chat" | "search" | "discover",
  "title": exact movie title if intent=search else null,
  "language": one of [{lang_list}] or null,
  "genre": one of [{genre_list}] or null,
  "person": actor/director name or null
}}"""

    try:
        resp = json_llm.invoke([HumanMessage(content=prompt)])
        match = re.search(r"\{.*?\}", resp.content, re.DOTALL)
        parsed = json.loads(match.group()) if match else {}
    except Exception:
        parsed = {}

    lang_code = tmdb.LANGUAGE_MAP.get((parsed.get("language") or "").lower())
    genre_id = tmdb.GENRE_MAP.get((parsed.get("genre") or "").lower())

    for keyword, genre_name in MOOD_TO_GENRE.items():
        if keyword in lower_query and not genre_id:
            genre_id = tmdb.GENRE_MAP.get(genre_name)
            break

    if not genre_id:
        genre_id = tmdb.GENRE_MAP.get("drama")

    intent = parsed.get("intent", "discover")
    search_title = None
    if intent == "search":
        search_title = parsed.get("title") or extract_title(query)

    return {
        **state,
        "intent": intent,
        "search_title": search_title,
        "language_code": lang_code,
        "genre_id": genre_id,
        "person": parsed.get("person"),
        "is_latest": is_latest,
        "is_lite_mood": is_lite,
        "disambiguation_options": None,
    }

