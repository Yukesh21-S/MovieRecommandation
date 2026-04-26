"""Discover nodes: retrieve candidates and respond with recommendations."""

from __future__ import annotations

from ..helpers import build_messages, format_movies, rank_discover, results_to_movies, sanitize_commentary
from ..prompts import prompt_discover
from ..runtime import llm
from ..state import State
from ...services import data_fetcher as tmdb
from ...services import vector_engine as vdb


def retrieve(state: State) -> State:
    # If the user asked about a person, skip local similarity retrieval and use fallback data.
    if state.get("person"):
        return {**state, "movies": []}

    results = vdb.query_movies(state["query"], n_results=40, language_code=state["language_code"])
    movies = (
        rank_discover(
            results_to_movies(results),
            state["query"],
            lite_mood=state["is_lite_mood"],
            genre_id=state.get("genre_id"),
        )
        if results
        else []
    )
    return {**state, "movies": movies}


def fetch_discover(state: State) -> State:
    print("[Fallback] Fetching discover results from TMDB...")
    sort = "release_date.desc" if state["is_latest"] else "popularity.desc"
    person_id = tmdb.get_person_id(state["person"]) if state.get("person") else None
    new_movies = tmdb.discover_movies(
        language_code=state["language_code"],
        genre_id=state["genre_id"],
        sort_by=sort,
        year_gte=2022 if state["is_latest"] else None,
        person_id=person_id,
        pages=3,
    )
    vdb.upsert_movies(new_movies)

    if person_id and new_movies:
        seen_titles: set[str] = set()
        filtered: list[dict] = []
        for movie in new_movies:
            title = str(movie.get("title", "")).strip()
            if not title or title in seen_titles:
                continue
            if movie.get("vote_average", 0) < 5.5:
                continue
            if "adult" in str(movie.get("overview", "")).lower():
                continue
            seen_titles.add(title)
            filtered.append(movie)
            if len(filtered) >= 10:
                break
        return {**state, "movies": filtered}

    results = vdb.query_movies(state["query"], language_code=state["language_code"])
    movies = (
        rank_discover(
            results_to_movies(results),
            state["query"],
            lite_mood=state["is_lite_mood"],
            genre_id=state.get("genre_id"),
        )
        if results
        else []
    )
    if len(movies) < 3:
        seen_titles: set[str] = set()
        curated: list[dict] = []
        required_genre_aliases = {
            key.lower() for key, value in tmdb.GENRE_MAP.items() if value == state.get("genre_id")
        }
        for movie in new_movies:
            title = str(movie.get("title", "")).strip()
            if not title or title in seen_titles:
                continue
            genres = str(movie.get("genres", "")).lower()
            if required_genre_aliases and not any(alias in genres for alias in required_genre_aliases):
                continue
            if movie.get("vote_average", 0) < 5.5:
                continue
            if "adult" in str(movie.get("overview", "")).lower():
                continue
            seen_titles.add(title)
            curated.append(movie)
            if len(curated) >= 10:
                break
        if curated:
            movies = curated
        elif new_movies:
            seen_titles.clear()
            fallback: list[dict] = []
            for movie in new_movies:
                title = str(movie.get("title", "")).strip()
                if not title or title in seen_titles:
                    continue
                if movie.get("vote_average", 0) < 5.5:
                    continue
                if "adult" in str(movie.get("overview", "")).lower():
                    continue
                seen_titles.add(title)
                fallback.append(movie)
                if len(fallback) >= 10:
                    break
            if fallback:
                movies = fallback
    return {**state, "movies": movies}


def respond_discover(state: State) -> State:
    movies, query, history = state["movies"], state["query"], state["history"]
    if not movies:
        return {**state, "response": "Sorry, I couldn't find matching movies. Try rephrasing!"}

    movie_block = format_movies(movies, overview_max_len=240)
    system = prompt_discover()
    user_msg = f'User request: "{query}"\n\nMatched movies:\n{movie_block}'
    commentary = sanitize_commentary(llm.invoke(build_messages(system, history, user_msg)).content.strip())
    return {**state, "response": f"{commentary}\n\n{movie_block}"}

