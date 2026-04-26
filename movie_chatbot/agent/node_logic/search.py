"""Search nodes: resolve a specific title and respond."""

from __future__ import annotations

from ..helpers import (
    build_messages,
    extract_title,
    format_movies,
    resolve_search_candidates,
    results_to_movies,
    sanitize_commentary,
)
from ..prompts import prompt_movie_search
from ..runtime import llm
from ..state import State
from ..services.movie_lookup import get_movie_details_by_title
from ...services import vector_engine as vdb


def search_movie(state: State) -> State:
    # Extract the intended movie title and search local vector matches first.
    raw_title = state["search_title"] or state["query"]
    title = extract_title(raw_title)
    print(f"[Search] Looking for title: '{title}'")

    results = vdb.query_movies(title, n_results=10)
    if results:
        candidates = results_to_movies(results)
        close, ambiguous = resolve_search_candidates(title, candidates)
        if close:
            print(f"[Search] Found in DB: {close[0]['title']}")
            options = close[:3] if ambiguous else None
            return {**state, "movies": [close[0]], "disambiguation_options": options}

    detailed = get_movie_details_by_title(title)
    if detailed:
        return {**state, "movies": [detailed], "disambiguation_options": None}

    return {**state, "movies": [], "disambiguation_options": None}


def respond_search(state: State) -> State:
    movies, query, history = state["movies"], state["query"], state["history"]
    title = state["search_title"] or query
    disambiguation_options = state.get("disambiguation_options") or []

    if not movies:
        return {
            **state,
            "response": (
                f"Sorry, I couldn't find **{title}** anywhere. "
                "Please check the spelling or try a different title."
            ),
        }

    if disambiguation_options and len(disambiguation_options) > 1:
        options = []
        for i, movie in enumerate(disambiguation_options[:3], 1):
            year = str(movie.get("release_date", ""))[:4] or "?"
            options.append(f"{i}. {movie.get('title', 'Unknown')} ({year})")
        options_text = "\n".join(options)
        return {
            **state,
            "response": ("I found multiple close matches. Did you mean:\n" f"{options_text}\n\nReply with the exact title you want."),
        }

    movie_block = format_movies([movies[0]], overview_max_len=None)
    system = prompt_movie_search()
    user_msg = f'User asked about: "{title}"\n\nMovie data:\n{movie_block}'
    commentary = sanitize_commentary(llm.invoke(build_messages(system, history, user_msg)).content.strip())
    return {**state, "response": f"{commentary}\n\n{movie_block}"}

