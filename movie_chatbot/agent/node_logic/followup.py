"""Follow-up node: answer questions about the last recommendations list."""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from ..helpers import (
    extract_last_movie_title,
    extract_recent_movie_block,
    extract_requested_index,
    extract_title,
    fuzzy_match_title,
    parse_movie_block,
    sanitize_commentary,
)
from ..prompts import prompt_followup
from ..runtime import llm
from ..services.movie_lookup import get_movie_details_by_title
from ..state import State


def followup(state: State) -> State:
    q = state["query"].strip()
    q_lower = q.lower()

    wants_detail = any(k in q_lower for k in ["detail", "detailed", "explain", "explanation", "elaborate", "more"])
    refers_to_previous = any(k in q_lower for k in ["this movie", "that movie", "this film", "that film", "it"])
    if wants_detail and refers_to_previous:
        last_title = extract_last_movie_title(state["history"])
        detailed = get_movie_details_by_title(last_title) if last_title else None
        if detailed:
            year = str(detailed.get("release_date", ""))[:4] or "?"
            return {
                **state,
                "response": (
                    f"**{detailed.get('title', last_title)} ({year})** — {detailed.get('genres', '')}, "
                    f"⭐ {detailed.get('vote_average', 0.0)}/10.\n"
                    f"{(detailed.get('overview') or '').strip()}"
                ).strip(),
            }

    movie_block = extract_recent_movie_block(state["history"])
    if not movie_block:
        return {
            **state,
            "response": "I don't have a recent movie list to reference. Ask for recommendations first.",
        }

    items = parse_movie_block(movie_block)

    idx = extract_requested_index(q)
    if idx is not None and 1 <= idx <= len(items):
        picked = items[idx - 1]
        title = picked.get("title", "")
        year = picked.get("year", "?")
        detailed = get_movie_details_by_title(title) or picked
        overview = detailed.get("overview") or ""
        return {
            **state,
            "response": (f"**{picked['title']} ({year})** — {picked['genres']}, ⭐ {picked['rating']}/10.\n{overview}").strip(),
        }

    if any(phrase in q_lower for phrase in ["most lighthearted", "lightest", "easy", "comfort", "funniest", "most fun"]):

        def score(movie: dict) -> float:
            g = str(movie.get("genres", "")).lower()
            s = 0.0
            if "comedy" in g:
                s += 3.0
            if "family" in g or "animation" in g:
                s += 2.0
            if "horror" in g or "thriller" in g:
                s -= 2.5
            s += float(movie.get("rating", 0.0)) / 10.0
            return s

        best = max(items, key=score, default=None)
        if best:
            year = best.get("year", "?")
            detailed = get_movie_details_by_title(best.get("title", "")) or best
            overview = (detailed.get("overview") or "").strip()
            return {
                **state,
                "response": (f"The most lighthearted pick from your list is **{best['title']} ({year})** — {best['genres']}.\n{overview}").strip(),
            }

    titles = [it["title"] for it in items if it.get("title")]
    matched = fuzzy_match_title(extract_title(q), titles)
    if matched:
        picked = next((it for it in items if it["title"] == matched), None)
        if picked:
            year = picked.get("year", "?")
            detailed = get_movie_details_by_title(picked.get("title", "")) or picked
            overview = detailed.get("overview") or ""
            return {
                **state,
                "response": (f"**{picked['title']} ({year})** — {picked['genres']}, ⭐ {picked['rating']}/10.\n{overview}").strip(),
            }

    system = prompt_followup()
    user_msg = f'Follow-up question: "{state["query"]}"\n\nAvailable movie data:\n{movie_block}'
    response = sanitize_commentary(llm.invoke([HumanMessage(content=f"{system}\n\n{user_msg}")]).content.strip())
    return {**state, "response": response}

