"""Small-talk node."""

from __future__ import annotations

from ..helpers import build_messages
from ..runtime import llm
from ..state import State


def chat(state: State) -> State:
    system = "You are a friendly movie chatbot. Keep answers brief and warm."
    response = llm.invoke(build_messages(system, state["history"], state["query"])).content.strip()
    return {**state, "response": response}

