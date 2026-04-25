"""
Shared movie lookup helpers.

Purpose: keep node functions small by centralizing "get best movie by title"
using vector DB first, then TMDB API fallback + upsert.
"""

from __future__ import annotations

from ..helpers import find_exact_title, results_to_movies
from ...services import data_fetcher as tmdb
from ...services import vector_engine as vdb


def get_movie_details_by_title(title: str) -> dict | None:
    """
    Return best matching movie dict for a title, or None if nothing found.
    Preference order:
    1) vector DB candidates (exact-ish match)
    2) TMDB search results (exact-ish match), also upserted to vector DB
    """
    title = (title or "").strip()
    if not title:
        return None

    results = vdb.query_movies(title, n_results=10)
    if results:
        candidates = results_to_movies(results)
        detailed = find_exact_title(title, candidates) or (candidates[0] if candidates else None)
        if detailed:
            return detailed

    new_movies = tmdb.search_movies(title)
    if new_movies:
        vdb.upsert_movies(new_movies)
        detailed = find_exact_title(title, new_movies) or new_movies[0]
        return detailed

    return None

