"""State types for the agent graph."""

from typing import Literal, TypedDict


class State(TypedDict):
    query: str
    history: list[dict]
    intent: Literal["chat", "search", "discover", "followup"]
    search_title: str | None
    language_code: str | None
    genre_id: int | None
    person: str | None
    is_latest: bool
    is_lite_mood: bool
    movies: list[dict]
    disambiguation_options: list[dict] | None
    response: str

