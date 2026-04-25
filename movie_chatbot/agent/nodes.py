"""Public node API (thin re-exports).

LangGraph workflow imports node functions from this module.
Implementations live in `movie_chatbot.agent.node_logic.*` to keep files small.
"""

from .node_logic.classify import classify
from .node_logic.chat import chat
from .node_logic.discover import fetch_discover, respond_discover, retrieve
from .node_logic.followup import followup
from .node_logic.search import respond_search, search_movie

__all__ = [
    "classify",
    "search_movie",
    "retrieve",
    "fetch_discover",
    "respond_search",
    "respond_discover",
    "followup",
    "chat",
]

