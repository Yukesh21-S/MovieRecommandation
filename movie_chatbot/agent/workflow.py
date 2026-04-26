"""LangGraph workflow composition and public run API."""

from typing import Literal

from langgraph.graph import END, StateGraph

from .nodes import (
    chat,
    classify,
    fetch_discover,
    followup,
    respond_discover,
    respond_search,
    retrieve,
    search_movie,
)
from .state import State


def _route_intent(state: State) -> Literal["chat", "followup", "search_movie", "retrieve"]:
    # Route the graph based on the classified user intent.
    intent = state["intent"]
    if intent == "chat":
        return "chat"
    if intent == "followup":
        return "followup"
    if intent == "search":
        return "search_movie"
    return "retrieve"


def _needs_discover_fallback(state: State) -> Literal["fetch_discover", "respond_discover"]:
    # If there are too few candidate movies, fetch additional discover results.
    return "fetch_discover" if len(state["movies"]) < 3 else "respond_discover"


def build_graph() -> StateGraph:
    graph = StateGraph(State)

    # Define the workflow nodes and how they connect.
    graph.add_node("classify", classify)
    graph.add_node("search_movie", search_movie)
    graph.add_node("retrieve", retrieve)
    graph.add_node("fetch_discover", fetch_discover)
    graph.add_node("respond_search", respond_search)
    graph.add_node("respond_discover", respond_discover)
    graph.add_node("followup", followup)
    graph.add_node("chat", chat)

    graph.set_entry_point("classify")
    graph.add_conditional_edges("classify", _route_intent)
    graph.add_edge("search_movie", "respond_search")
    graph.add_conditional_edges("retrieve", _needs_discover_fallback)
    graph.add_edge("fetch_discover", "respond_discover")
    graph.add_edge("respond_search", END)
    graph.add_edge("respond_discover", END)
    graph.add_edge("followup", END)
    graph.add_edge("chat", END)

    return graph.compile()


_graph = build_graph()


def run(query: str, history: list[dict] | None = None) -> tuple[str, list[dict]]:
    """Execute the graph for a user query and return the assistant response plus updated history."""
    history = history or []
    final_state = _graph.invoke(
        {
            "query": query,
            "history": history,
            "intent": "discover",
            "search_title": None,
            "language_code": None,
            "genre_id": None,
            "person": None,
            "is_latest": False,
            "is_lite_mood": False,
            "movies": [],
            "disambiguation_options": None,
            "response": "",
        }
    )
    response = final_state["response"]
    updated_history = history + [
        {"role": "user", "content": query},
        {"role": "assistant", "content": response},
    ]
    return response, updated_history

