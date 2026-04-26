"""
Fetches movie data from the TMDB API.
"""

import datetime
import json
import os

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv()

API_KEY = os.getenv("TMDB_API_KEY")
BASE_URL = "https://api.tmdb.org/3"

# HTTP session with retry behavior for unstable TMDB responses.
session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)

LANGUAGE_MAP: dict[str, str] = {
    "hindi": "hi",
    "french": "fr",
    "korean": "ko",
    "japanese": "ja",
    "spanish": "es",
    "tamil": "ta",
    "telugu": "te",
    "malayalam": "ml",
    "english": "en",
    "chinese": "zh",
    "german": "de",
    "italian": "it",
    "portuguese": "pt",
    "arabic": "ar",
    "russian": "ru",
    "turkish": "tr",
}

GENRE_MAP: dict[str, int] = {
    "action": 28,
    "adventure": 12,
    "animation": 16,
    "comedy": 35,
    "crime": 80,
    "documentary": 99,
    "drama": 18,
    "family": 10751,
    "fantasy": 14,
    "history": 36,
    "horror": 27,
    "romance": 10749,
    "sci-fi": 878,
    "science fiction": 878,
    "thriller": 53,
    "war": 10752,
}


def _get_genre_mapping() -> dict[int, str]:
    resp = session.get(
        f"{BASE_URL}/genre/movie/list",
        params={"api_key": API_KEY, "language": "en-US"},
        timeout=10,
    )
    if resp.status_code == 200:
        return {g["id"]: g["name"] for g in resp.json().get("genres", [])}
    return {}


def _clean_movie(movie: dict, genre_mapping: dict[int, str]) -> dict | None:
    # Normalize TMDB response fields and skip movies without an overview.
    if not movie.get("overview"):
        return None
    genre_names = [genre_mapping.get(gid, "Unknown") for gid in movie.get("genre_ids", [])]
    return {
        "id": movie["id"],
        "title": movie["title"],
        "overview": movie["overview"],
        "genres": ", ".join(genre_names),
        "release_date": movie.get("release_date", "Unknown"),
        "vote_average": movie.get("vote_average", 0.0),
        "language": movie.get("original_language", "en"),
    }


def _save(movies: list[dict], path: str = "movies_data.json") -> None:
    # Save the curated movie dataset to disk for later vector DB setup.
    with open(path, "w", encoding="utf-8") as file:
        json.dump(movies, file, indent=4, ensure_ascii=False)


def search_movies(query: str) -> list[dict]:
    # Search TMDB for a specific movie title.
    print(f"[TMDB] Searching for '{query}'...")
    genre_mapping = _get_genre_mapping()
    results: list[dict] = []

    for page in range(1, 3):
        resp = session.get(
            f"{BASE_URL}/search/movie",
            params={"api_key": API_KEY, "language": "en-US", "query": query, "page": page},
            timeout=10,
        )
        if resp.status_code != 200:
            break
        data = resp.json()
        for movie in data.get("results", []):
            cleaned = _clean_movie(movie, genre_mapping)
            if cleaned:
                results.append(cleaned)
        if page >= data.get("total_pages", 1):
            break

    print(f"[TMDB] Found {len(results)} movies.")
    return results


def discover_movies(
    language_code: str | None = None,
    genre_id: int | None = None,
    sort_by: str = "popularity.desc",
    year_gte: int | None = None,
    person_id: int | None = None,
    pages: int = 3,
) -> list[dict]:
    # Query TMDB discover API for recommendation candidates.
    print(f"[TMDB] Discovering movies (lang={language_code}, genre={genre_id}, sort={sort_by})...")
    genre_mapping = _get_genre_mapping()
    today = datetime.date.today().isoformat()
    results: list[dict] = []

    for page in range(1, pages + 1):
        params: dict = {
            "api_key": API_KEY,
            "language": "en-US",
            "sort_by": sort_by,
            "include_adult": "false",
            "vote_count.gte": 10,
            "primary_release_date.lte": today,
            "page": page,
        }
        if language_code:
            params["with_original_language"] = language_code
        if genre_id:
            params["with_genres"] = genre_id
        if year_gte:
            params["primary_release_date.gte"] = f"{year_gte}-01-01"
        if person_id:
            params["with_people"] = person_id

        resp = session.get(f"{BASE_URL}/discover/movie", params=params, timeout=10)
        if resp.status_code != 200:
            break
        data = resp.json()
        for movie in data.get("results", []):
            cleaned = _clean_movie(movie, genre_mapping)
            if cleaned:
                results.append(cleaned)
        if page >= data.get("total_pages", 1):
            break

    print(f"[TMDB] Discovered {len(results)} movies.")
    return results


def get_person_id(name: str) -> int | None:
    # Resolve an actor or director name to a TMDB person ID.
    resp = session.get(
        f"{BASE_URL}/search/person",
        params={"api_key": API_KEY, "query": name, "language": "en-US"},
        timeout=10,
    )
    if resp.status_code == 200:
        results = resp.json().get("results", [])
        if results:
            return results[0]["id"]
    return None


def fetch_and_save_diverse_movies(path: str = "movies_data.json") -> list[dict]:
    # Build and persist a diverse local dataset for the vector store bootstrap.
    print("[Setup] Building diversified movie dataset...")
    all_movies: list[dict] = []
    all_movies.extend(discover_movies(sort_by="popularity.desc", pages=2))
    all_movies.extend(discover_movies(language_code="ta", pages=2))
    all_movies.extend(discover_movies(genre_id=GENRE_MAP["drama"], pages=1))
    all_movies.extend(discover_movies(genre_id=GENRE_MAP["comedy"], pages=1))

    seen: set[int] = set()
    unique = [movie for movie in all_movies if not (movie["id"] in seen or seen.add(movie["id"]))]  # type: ignore[func-returns-value]
    _save(unique, path)
    print(f"[Setup] Saved {len(unique)} unique movies.")
    return unique

