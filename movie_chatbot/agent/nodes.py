"""Graph node implementations for movie agent."""

import json
import os
import re

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from .constants import LATEST_KEYWORDS, LITE_MOOD_KEYWORDS, MOOD_TO_GENRE
from .helpers import (
    build_messages,
    extract_title,
    extract_recent_movie_block,
    extract_requested_index,
    fuzzy_match_title,
    format_movies,
    is_followup,
    parse_movie_block,
    rank_discover,
    resolve_search_candidates,
    results_to_movies,
    sanitize_commentary,
    title_words,
)
from .prompts import prompt_discover, prompt_followup, prompt_movie_search
from .state import State
from ..services import data_fetcher as tmdb
from ..services import vector_engine as vdb

load_dotenv()

_llm = ChatGroq(
    model=os.getenv("LLM_MODEL", "llama-3.1-8b-instant"),
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3,
)

_json_llm = ChatGroq(
    model=os.getenv("LLM_MODEL", "llama-3.1-8b-instant"),
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)


def classify(state: State) -> State:
    query, history = state["query"], state["history"]
    lower_query = query.lower()

    is_latest = bool(LATEST_KEYWORDS & set(lower_query.split()))
    is_lite = bool(LITE_MOOD_KEYWORDS & set(lower_query.split()))

    if is_followup(query, history):
        return {
            **state,
            "intent": "followup",
            "search_title": None,
            "language_code": None,
            "genre_id": None,
            "person": None,
            "is_latest": False,
            "is_lite_mood": False,
            "disambiguation_options": None,
        }

    if lower_query.strip() in {"hi", "hello", "hey", "hola", "namaste"} or (
        "how are you" in lower_query and "movie" not in lower_query
    ):
        return {
            **state,
            "intent": "chat",
            "search_title": None,
            "language_code": None,
            "genre_id": None,
            "person": None,
            "is_latest": is_latest,
            "is_lite_mood": False,
            "disambiguation_options": None,
        }

    lang_list = ", ".join(tmdb.LANGUAGE_MAP)
    genre_list = ", ".join(tmdb.GENRE_MAP)
    prompt = f"""Classify this movie query. Respond ONLY with valid JSON — no markdown, no explanation.

Query: "{query}"

Rules:
- "search"   → user wants info about ONE specific movie by title (e.g. "Find Inception", "Tell me about Parasite")
- "discover" → user wants recommendations by mood, genre, language, or person
- "chat"     → greeting or small talk only

JSON:
{{
  "intent": "chat" | "search" | "discover",
  "title": exact movie title if intent=search else null,
  "language": one of [{lang_list}] or null,
  "genre": one of [{genre_list}] or null,
  "person": actor/director name or null
}}"""

    try:
        resp = _json_llm.invoke([HumanMessage(content=prompt)])
        match = re.search(r"\{.*?\}", resp.content, re.DOTALL)
        parsed = json.loads(match.group()) if match else {}
    except Exception:
        parsed = {}

    lang_code = tmdb.LANGUAGE_MAP.get((parsed.get("language") or "").lower())
    genre_id = tmdb.GENRE_MAP.get((parsed.get("genre") or "").lower())

    for keyword, genre_name in MOOD_TO_GENRE.items():
        if keyword in lower_query and not genre_id:
            genre_id = tmdb.GENRE_MAP.get(genre_name)
            break

    if not genre_id:
        genre_id = tmdb.GENRE_MAP.get("drama")

    intent = parsed.get("intent", "discover")
    search_title = None
    if intent == "search":
        search_title = parsed.get("title") or extract_title(query)

    return {
        **state,
        "intent": intent,
        "search_title": search_title,
        "language_code": lang_code,
        "genre_id": genre_id,
        "person": parsed.get("person"),
        "is_latest": is_latest,
        "is_lite_mood": is_lite,
        "disambiguation_options": None,
    }


def search_movie(state: State) -> State:
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

    print("[Search] Not in DB -> fetching from TMDB API...")
    new_movies = tmdb.search_movies(title)
    if new_movies:
        vdb.upsert_movies(new_movies)
        close, ambiguous = resolve_search_candidates(title, new_movies)
        if close:
            print(f"[Search] TMDB returned: {close[0]['title']}")
            options = close[:3] if ambiguous else None
            return {**state, "movies": [close[0]], "disambiguation_options": options}

    return {**state, "movies": [], "disambiguation_options": None}


def retrieve(state: State) -> State:
    # Person queries are much more reliable from TMDB discover(with_people) fallback.
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
            # Final fallback: return best TMDB results even if genre labels don't match aliases.
            # Prevents empty responses when upstream genre naming differs (e.g., "Science Fiction").
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
            "response": (
                "I found multiple close matches. Did you mean:\n"
                f"{options_text}\n\nReply with the exact title you want."
            ),
        }

    # For single-movie detail, never truncate the overview.
    movie_block = format_movies([movies[0]], overview_max_len=None)
    system = prompt_movie_search()
    user_msg = f'User asked about: "{title}"\n\nMovie data:\n{movie_block}'
    commentary = sanitize_commentary(_llm.invoke(build_messages(system, history, user_msg)).content.strip())
    return {**state, "response": f"{commentary}\n\n{movie_block}"}


def respond_discover(state: State) -> State:
    movies, query, history = state["movies"], state["query"], state["history"]
    if not movies:
        return {**state, "response": "Sorry, I couldn't find matching movies. Try rephrasing!"}

    movie_block = format_movies(movies, overview_max_len=240)
    system = prompt_discover()
    user_msg = f'User request: "{query}"\n\nMatched movies:\n{movie_block}'
    commentary = sanitize_commentary(_llm.invoke(build_messages(system, history, user_msg)).content.strip())
    return {**state, "response": f"{commentary}\n\n{movie_block}"}


def followup(state: State) -> State:
    movie_block = extract_recent_movie_block(state["history"])
    if not movie_block:
        return {
            **state,
            "response": "I don't have a recent movie list to reference. Ask for recommendations first.",
        }

    items = parse_movie_block(movie_block)
    q = state["query"].strip()
    q_lower = q.lower()

    # 1) If user refers to a specific number/ordinal, answer from that item directly.
    idx = extract_requested_index(q)
    if idx is not None and 1 <= idx <= len(items):
        picked = items[idx - 1]
        title = picked.get("title", "")
        year = picked.get("year", "?")

        # Re-fetch full details; history list overview may be truncated.
        detailed = None
        results = vdb.query_movies(title, n_results=10)
        if results:
            candidates = results_to_movies(results)
            detailed = find_exact_title(title, candidates) or (candidates[0] if candidates else None)
        if not detailed:
            new_movies = tmdb.search_movies(title)
            if new_movies:
                vdb.upsert_movies(new_movies)
                detailed = find_exact_title(title, new_movies) or new_movies[0]
        overview = (detailed or picked).get("overview") or ""
        return {
            **state,
            "response": (
                f"**{picked['title']} ({year})** — {picked['genres']}, ⭐ {picked['rating']}/10.\n"
                f"{overview}".strip()
            ),
        }

    # 2) If user asks "most lighthearted" / "easy" / "fun" pick best from list by genres.
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
            return {
                **state,
                "response": (
                    f"The most lighthearted pick from your list is **{best['title']} ({year})** "
                    f"— {best['genres']}.\n{(best.get('overview') or '').strip()}"
                ).strip(),
            }

    # 3) If user mentions a title (even with typos), match within the list.
    titles = [it["title"] for it in items if it.get("title")]
    matched = fuzzy_match_title(extract_title(q), titles)
    if matched:
        picked = next((it for it in items if it["title"] == matched), None)
        if picked:
            title = picked.get("title", "")
            year = picked.get("year", "?")

            detailed = None
            results = vdb.query_movies(title, n_results=10)
            if results:
                candidates = results_to_movies(results)
                detailed = find_exact_title(title, candidates) or (candidates[0] if candidates else None)
            if not detailed:
                new_movies = tmdb.search_movies(title)
                if new_movies:
                    vdb.upsert_movies(new_movies)
                    detailed = find_exact_title(title, new_movies) or new_movies[0]
            overview = (detailed or picked).get("overview") or ""
            return {
                **state,
                "response": (
                    f"**{picked['title']} ({year})** — {picked['genres']}, ⭐ {picked['rating']}/10.\n"
                    f"{overview}".strip()
                ),
            }

    system = prompt_followup()
    user_msg = f'Follow-up question: "{state["query"]}"\n\nAvailable movie data:\n{movie_block}'
    response = sanitize_commentary(_llm.invoke([HumanMessage(content=f"{system}\n\n{user_msg}")]).content.strip())
    return {**state, "response": response}


def chat(state: State) -> State:
    system = "You are a friendly movie chatbot. Keep answers brief and warm."
    response = _llm.invoke(build_messages(system, state["history"], state["query"])).content.strip()
    return {**state, "response": response}

