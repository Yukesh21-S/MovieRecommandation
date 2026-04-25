"""Helper functions used by graph nodes."""

import difflib
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .constants import FOLLOWUP_HINTS, LITE_GENRES
from ..services import data_fetcher as tmdb


def extract_title(query: str) -> str:
    q = query.strip()
    for prefix in [
        "find ",
        "search for ",
        "look up ",
        "tell me about ",
        "what is ",
        "what's ",
        "show me ",
        "about the movie ",
        "the movie ",
        "movie called ",
        "movie named ",
        "film called ",
        "film named ",
    ]:
        if q.lower().startswith(prefix):
            q = q[len(prefix) :].strip()
            break

    q_lower = q.lower()
    for suffix in [
        " tell me about this movie",
        " tell me about this film",
        " tell me about this",
        " movie",
        " film",
    ]:
        if q_lower.endswith(suffix):
            q = q[: -len(suffix)].strip()
            break

    return q.rstrip("?").strip()


def _truncate(text: str, max_len: int | None) -> str:
    if max_len is None:
        return text
    text = str(text or "").strip()
    if len(text) <= max_len:
        return text
    # Keep it readable in lists, but signal it's truncated.
    return text[: max(0, max_len - 1)].rstrip() + "…"


def format_movies(movies: list[dict], overview_max_len: int | None = 200) -> str:
    lines = []
    for i, movie in enumerate(movies, 1):
        year = (movie.get("release_date") or "")[:4] or "?"
        overview = _truncate(str(movie.get("overview", "")), overview_max_len)
        lines.append(
            f"{i}. {movie['title']} ({year}) | {movie['genres']} | ⭐ {movie['vote_average']}/10\n"
            f"   {overview}"
        )
    return "\n\n".join(lines)


def results_to_movies(results: dict) -> list[dict]:
    movies = []
    for i, meta in enumerate(results["metadatas"][0]):
        doc = results["documents"][0][i]
        overview = doc.split("Overview: ", 1)[-1] if "Overview: " in doc else doc
        movies.append({**meta, "overview": overview, "distance": results["distances"][0][i]})
    return movies


def rank_discover(
    movies: list[dict], query: str, lite_mood: bool = False, genre_id: int | None = None
) -> list[dict]:
    query_words = set(re.findall(r"\w+", query.lower()))
    scored = []
    seen = set()
    required_genre = next((k for k, v in tmdb.GENRE_MAP.items() if v == genre_id), None) if genre_id else None

    for movie in movies:
        title = movie.get("title", "")
        if title in seen:
            continue
        if movie.get("vote_average", 0) < 5.5:
            continue
        if "adult" in movie.get("overview", "").lower():
            continue

        genres = movie.get("genres", "").lower()
        if lite_mood and not any(genre in genres for genre in LITE_GENRES):
            continue
        if required_genre and required_genre not in genres:
            continue

        sim = max(0.0, 1.0 - movie.get("distance", 1.0))
        rating_score = movie.get("vote_average", 0) / 10.0
        title_tokens = set(re.findall(r"\w+", title.lower()))
        keyword_boost = 0.3 if query_words & title_tokens else 0.0
        overview_tokens = set(re.findall(r"\w+", movie.get("overview", "").lower()))
        query_match_boost = 0.2 if query_words & overview_tokens else 0.0
        comedy_boost = 0.2 if "comedy" in genres and (lite_mood or "uplift" in query.lower() or "light" in query.lower()) else 0.0
        seen.add(title)
        scored.append(
            {
                **movie,
                "_score": sim * 0.45 + rating_score * 0.25 + keyword_boost + query_match_boost + comedy_boost,
            }
        )

    scored.sort(key=lambda item: item["_score"], reverse=True)
    return scored[:10]


def title_words(title: str) -> set[str]:
    stop = {"a", "an", "the", "of", "in", "on", "at", "to", "and", "or"}
    return {w for w in re.findall(r"[a-z0-9]+", title.lower()) if w not in stop and len(w) > 1}


def normalize_title(title: str) -> str:
    """Normalize a title for high-precision equality checks."""
    return " ".join(sorted(title_words(title)))


def resolve_search_candidates(title: str, movies: list[dict]) -> tuple[list[dict], bool]:
    """
    Strict title resolver:
    1) exact normalized match
    2) whole-word subset with close token-length ratio
    3) very strict fuzzy match with token overlap
    Returns (matches, is_ambiguous).
    """
    if not movies:
        return [], False

    normalized_query = normalize_title(title)
    query_words = title_words(title)
    if not normalized_query:
        return [], False

    def sort_key(movie: dict) -> tuple[float, str]:
        return (float(movie.get("vote_average", 0.0)), str(movie.get("release_date", "")))

    exact = [movie for movie in movies if normalize_title(movie.get("title", "")) == normalized_query]
    if exact:
        exact.sort(key=sort_key, reverse=True)
        return exact[:3], len(exact) > 1

    whole_word = []
    for movie in movies:
        db_words = title_words(movie.get("title", ""))
        if not query_words or not query_words.issubset(db_words):
            continue
        ratio = len(db_words) / max(len(query_words), 1)
        if ratio <= 1.6:
            whole_word.append(movie)
    if whole_word:
        whole_word.sort(key=sort_key, reverse=True)
        return whole_word[:3], len(whole_word) > 1

    strict_fuzzy = []
    for movie in movies:
        db_title = str(movie.get("title", ""))
        db_words = title_words(db_title)
        token_overlap = len(query_words & db_words) / max(len(query_words | db_words), 1)
        similarity = difflib.SequenceMatcher(None, normalized_query, normalize_title(db_title)).ratio()
        if similarity >= 0.93 and token_overlap >= 0.6:
            strict_fuzzy.append(movie)
    if strict_fuzzy:
        strict_fuzzy.sort(key=sort_key, reverse=True)
        return strict_fuzzy[:3], len(strict_fuzzy) > 1

    return [], False


def find_exact_title(title: str, movies: list[dict]) -> dict | None:
    title_lower = title.lower().strip()
    search_words = title_words(title)
    is_short_single_word = len(search_words) == 1 and len(next(iter(search_words), "")) <= 3

    for movie in movies:
        db_title = movie["title"].lower().strip()
        db_words = title_words(movie["title"])

        if db_title == title_lower:
            return movie

        # For short one-word titles like "Up", avoid accidental partial matches (e.g., "Balls Up").
        if is_short_single_word:
            continue

        if search_words and search_words.issubset(db_words):
            ratio = len(db_words) / max(len(search_words), 1)
            if ratio <= 2.0:
                return movie

        similarity = difflib.SequenceMatcher(None, title_lower, db_title).ratio()
        if similarity > 0.85:
            return movie

    best = min(movies, key=lambda m: m.get("distance", 1.0), default=None)
    if best and best.get("distance", 1.0) < 0.18:
        if is_short_single_word:
            best_title_words = title_words(best.get("title", ""))
            if search_words != best_title_words:
                return None
        return best
    return None


def build_messages(system: str, history: list[dict], user_msg: str) -> list:
    return (
        [SystemMessage(content=system)]
        + [
            HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"])
            for msg in history[-6:]
        ]
        + [HumanMessage(content=user_msg)]
    )


def extract_recent_movie_block(history: list[dict]) -> str:
    """Extract the latest numbered movie list from assistant history."""
    for msg in reversed(history):
        if msg.get("role") != "assistant":
            continue
        lines = msg.get("content", "").splitlines()
        collected: list[str] = []
        in_list = False
        for line in lines:
            if re.match(r"^\s*\d+\.\s+", line):
                in_list = True
                collected.append(line)
                continue
            if in_list and line.startswith("   "):
                collected.append(line)
                continue
            if in_list and not line.strip():
                collected.append(line)
                continue
            if in_list:
                break
        if collected:
            return "\n".join(collected).strip()
    return ""


def extract_last_movie_title(history: list[dict]) -> str | None:
    """
    Extract the last movie title the assistant referenced in a detail response.
    Matches patterns like: **Title (2024)** — Genres, ⭐ x/10.
    """
    pat = re.compile(r"\*\*(?P<title>.+?)\s+\((?:\d{4}|\?)\)\*\*")
    for msg in reversed(history):
        if msg.get("role") != "assistant":
            continue
        m = pat.search(msg.get("content", ""))
        if m:
            title = m.group("title").strip()
            return title or None
    return None


def parse_movie_block(movie_block: str) -> list[dict]:
    """
    Parse numbered movie block lines into structured items:
    1. Title (YEAR) | Genres | ⭐ rating
       overview...
    """
    items: list[dict] = []
    current: dict | None = None
    for line in movie_block.splitlines():
        m = re.match(r"^\s*(\d+)\.\s+(.*?)\s+\((\d{4}|\?)\)\s+\|\s+(.*?)\s+\|\s+⭐\s+([0-9.]+)/10", line)
        if m:
            if current:
                items.append(current)
            idx, title, year, genres, rating = m.groups()
            current = {
                "index": int(idx),
                "title": title.strip(),
                "year": year.strip(),
                "genres": genres.strip(),
                "rating": float(rating),
                "overview": "",
            }
            continue
        if current and line.startswith("   "):
            current["overview"] = (current["overview"] + "\n" + line.strip()).strip()
    if current:
        items.append(current)
    return items


def extract_requested_index(query: str) -> int | None:
    q = query.lower()
    # numeric patterns
    m = re.search(r"\bmovie\s*(\d+)\b|\b(\d+)(st|nd|rd|th)\b|\b(\d+)\b", q)
    if m:
        for g in m.groups():
            if g and g.isdigit():
                return int(g)
    # word ordinals
    mapping = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}
    for k, v in mapping.items():
        if k in q:
            return v
    return None


def fuzzy_match_title(query: str, titles: list[str]) -> str | None:
    """Return best matching title from list based on SequenceMatcher."""
    q = query.lower().strip()
    if not q:
        return None
    best = None
    best_score = 0.0
    for t in titles:
        score = difflib.SequenceMatcher(None, q, t.lower()).ratio()
        if score > best_score:
            best_score = score
            best = t
    return best if best_score >= 0.78 else None


def sanitize_commentary(text: str) -> str:
    """Prevent model from emitting duplicate numbered lists in commentary."""
    lines = text.strip().splitlines()
    cleaned: list[str] = []
    for line in lines:
        if re.match(r"^\s*\d+\.\s+", line):
            break
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def is_followup(query: str, history: list[dict]) -> bool:
    if not history:
        return False
    lower_query = query.lower().strip()

    # If user refers to an item number/ordinal and mentions the list/above,
    # it's almost certainly a follow-up to the previous recommendations.
    idx = extract_requested_index(query)
    if idx is not None and any(token in lower_query for token in ["list", "above", "recommendations", "suggestions"]):
        return True

    if any(hint in lower_query for hint in FOLLOWUP_HINTS) and len(query.split()) < 15:
        return True
    if len(query.split()) <= 4:
        for msg in reversed(history):
            if msg["role"] == "assistant":
                if lower_query in msg["content"].lower():
                    return True
                break
    return False

