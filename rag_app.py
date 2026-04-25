"""
agent.py
LangGraph-powered movie chatbot agent.

Graph flow:
  classify → chat        (greetings / small talk)
           → followup    (questions about previously shown movies)
           → search_movie → respond_search   (specific title lookup)
           → retrieve    → fetch_discover? → respond_discover
"""

import os
import re
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

import data_fetcher as tmdb
import vector_engine as vdb

load_dotenv()

# ── LLM setup ─────────────────────────────────────────────────────────────────

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

# ── Lookup maps ───────────────────────────────────────────────────────────────

MOOD_TO_GENRE: dict[str, str] = {
    "motivat": "drama", "inspir": "drama",
    "stress": "comedy", "tired": "comedy", "relax": "comedy",
    "fun": "comedy", "funny": "comedy", "happy": "comedy", "feel good": "comedy",
    "sad": "drama", "excited": "action", "scared": "horror", "romantic": "romance",
}

LITE_MOOD_KEYWORDS = {"lite", "light", "stress", "stressed", "tired", "relax",
                      "fun", "happy", "feel good", "chill", "lighthearted"}
LITE_GENRES = {"comedy", "family", "romance", "animation"}

LATEST_KEYWORDS = {"latest", "new", "recent", "newest", "2022", "2023", "2024", "2025"}

FOLLOWUP_HINTS = {
    "first", "second", "third", "fourth", "fifth",
    "that movie", "those movies", "tell me more", "more about",
    "what about", "elaborate", "details", "explain", "describe",
    "movie 1", "movie 2", "movie 3", "number", "plot", "story",
    "who directed", "who acted", "stars in", "rating of", "overview of",
    "which one", "the one", "is it good", "worth watching",
    "how is it", "about it", "see it", "watch it",
}

# Distance threshold: above this = not a real match for a title search
SEARCH_DISTANCE_THRESHOLD = 0.45

# ── State ─────────────────────────────────────────────────────────────────────

class State(TypedDict):
    query: str
    history: list[dict]

    # Populated by classify
    intent: Literal["chat", "search", "discover", "followup"]
    search_title: str | None     # extracted clean title for search queries
    language_code: str | None
    genre_id: int | None
    person: str | None
    is_latest: bool
    is_lite_mood: bool

    # Populated by retrieve nodes
    movies: list[dict]

    # Output
    response: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_title(query: str) -> str:
    """Strip common search phrases to get the bare movie title."""
    q = query.strip()
    for prefix in [
        "find ", "search for ", "look up ", "tell me about ", "what is ",
        "what's ", "show me ", "about the movie ", "the movie ", "movie called ",
        "movie named ", "film called ", "film named ",
    ]:
        if q.lower().startswith(prefix):
            q = q[len(prefix):]
    # Remove trailing question marks
    return q.rstrip("?").strip()


def _format_movies(movies: list[dict]) -> str:
    lines = []
    for i, m in enumerate(movies, 1):
        year = (m.get("release_date") or "")[:4] or "?"
        lines.append(
            f"{i}. {m['title']} ({year}) | {m['genres']} | ⭐ {m['vote_average']}/10\n"
            f"   {m.get('overview', '')[:200]}"
        )
    return "\n\n".join(lines)


def _results_to_movies(results: dict) -> list[dict]:
    movies = []
    for i, meta in enumerate(results["metadatas"][0]):
        doc = results["documents"][0][i]
        overview = doc.split("Overview: ", 1)[-1] if "Overview: " in doc else doc
        movies.append({**meta, "overview": overview, "distance": results["distances"][0][i]})
    return movies


def _rank_discover(movies: list[dict], query: str, lite_mood: bool = False) -> list[dict]:
    """Rank movies for discover queries; optionally enforce lite-genre filter."""
    query_words = set(re.findall(r"\w+", query.lower()))
    scored = []
    for m in movies:
        genres = m.get("genres", "").lower()
        if lite_mood and not any(g in genres for g in LITE_GENRES):
            continue
        sim = max(0.0, 1.0 - m.get("distance", 1.0))
        rating_score = m.get("vote_average", 0) / 10.0
        keyword_boost = 0.3 if query_words & set(re.findall(r"\w+", m["title"].lower())) else 0.0
        comedy_boost = 0.2 if "comedy" in genres else 0.0
        scored.append({**m, "_score": sim * 0.5 + rating_score * 0.3 + keyword_boost + comedy_boost})
    scored.sort(key=lambda x: x["_score"], reverse=True)
    return scored[:10]


def _title_words(title: str) -> set[str]:
    """Meaningful words in a title — strips articles and single chars."""
    stop = {"a", "an", "the", "of", "in", "on", "at", "to", "and", "or"}
    return {w for w in re.findall(r"[a-z0-9]+", title.lower()) if w not in stop and len(w) > 1}


def _find_exact_title(title: str, movies: list[dict]) -> dict | None:
    """
    Return a DB movie ONLY when there is a confident title match.
    Returns None so the caller hits TMDB instead of showing a wrong movie.
    """
    title_lower = title.lower().strip()
    search_words = _title_words(title)

    for m in movies:
        db_title = m["title"].lower().strip()
        db_words = _title_words(m["title"])

        # Rule 1: exact full-title match
        if db_title == title_lower:
            return m

        # Rule 2: all search words present in DB title with similar word count
        # Prevents "I" matching "Inception" (single-char words are stripped)
        if search_words and search_words.issubset(db_words):
            ratio = len(db_words) / max(len(search_words), 1)
            if ratio <= 2.0:
                return m

    # Rule 3: very tight vector distance (nearly identical)
    best = min(movies, key=lambda m: m.get("distance", 1.0), default=None)
    if best and best.get("distance", 1.0) < 0.18:
        return best

    return None  # → caller will fetch from TMDB


def _build_messages(system: str, history: list[dict], user_msg: str) -> list:
    return (
        [SystemMessage(content=system)]
        + [HumanMessage(content=m["content"]) if m["role"] == "user"
           else AIMessage(content=m["content"])
           for m in history[-6:]]
        + [HumanMessage(content=user_msg)]
    )


def _is_followup(query: str, history: list[dict]) -> bool:
    if not history:
        return False
    q = query.lower().strip()
    if any(hint in q for hint in FOLLOWUP_HINTS) and len(query.split()) < 15:
        return True
    if len(query.split()) <= 4:
        for msg in reversed(history):
            if msg["role"] == "assistant":
                if q in msg["content"].lower():
                    return True
                break
    return False


# ── Graph nodes ───────────────────────────────────────────────────────────────

def classify(state: State) -> State:
    query, history = state["query"], state["history"]
    q = query.lower()

    is_latest = bool(LATEST_KEYWORDS & set(q.split()))
    is_lite = bool(LITE_MOOD_KEYWORDS & set(q.split()))

    # Priority 1: follow-up
    if _is_followup(query, history):
        return {**state, "intent": "followup", "search_title": None, "language_code": None,
                "genre_id": None, "person": None, "is_latest": False, "is_lite_mood": False}

    # Priority 2: greeting
    if q.strip() in {"hi", "hello", "hey", "hola", "namaste"} or \
       ("how are you" in q and "movie" not in q):
        return {**state, "intent": "chat", "search_title": None, "language_code": None,
                "genre_id": None, "person": None, "is_latest": is_latest, "is_lite_mood": False}

    # Priority 3: LLM classification
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
        import json as _json
        resp = _json_llm.invoke([HumanMessage(content=prompt)])
        match = re.search(r"\{.*?\}", resp.content, re.DOTALL)
        parsed = _json.loads(match.group()) if match else {}
    except Exception:
        parsed = {}

    lang_code = tmdb.LANGUAGE_MAP.get((parsed.get("language") or "").lower())
    genre_id = tmdb.GENRE_MAP.get((parsed.get("genre") or "").lower())

    for kw, genre_name in MOOD_TO_GENRE.items():
        if kw in q and not genre_id:
            genre_id = tmdb.GENRE_MAP.get(genre_name)
            break

    intent = parsed.get("intent", "discover")
    # Extract title: prefer LLM-parsed title, fall back to stripping the query manually
    search_title = None
    if intent == "search":
        search_title = parsed.get("title") or _extract_title(query)

    return {
        **state,
        "intent": intent,
        "search_title": search_title,
        "language_code": lang_code,
        "genre_id": genre_id,
        "person": parsed.get("person"),
        "is_latest": is_latest,
        "is_lite_mood": is_lite,
    }


def search_movie(state: State) -> State:
    """
    Handle specific title searches.
    1. Always clean the title first (strip "Find", "Tell me about", etc).
    2. Check vector DB — only accept a confident title match.
    3. If not found → call TMDB search API (real fallback).
    4. Return only the matched movie, never unrelated results.
    """
    # Always use the cleaned title for both DB lookup and TMDB search
    raw_title = state["search_title"] or state["query"]
    title = _extract_title(raw_title)          # strips "Find ", "Tell me about ", etc.
    print(f"[Search] Looking for title: '{title}'")

    # Step 1: check vector DB with strict matching
    results = vdb.query_movies(title, n_results=10)
    if results:
        candidates = _results_to_movies(results)
        match = _find_exact_title(title, candidates)
        if match:
            print(f"[Search] Found in DB: {match['title']}")
            return {**state, "movies": [match]}

    # Step 2: TMDB API fallback — movie not confidently found in DB
    print(f"[Search] Not in DB → fetching from TMDB API…")
    new_movies = tmdb.search_movies(title)
    if new_movies:
        vdb.upsert_movies(new_movies)
        # Pick the best match: prefer word-level title match, fallback to top result
        search_words = _title_words(title)
        close = [m for m in new_movies if search_words and search_words.issubset(_title_words(m["title"]))]
        if not close:
            close = new_movies[:1]   # TMDB returns results sorted by relevance; top = best
        print(f"[Search] TMDB returned: {close[0]['title']}")
        return {**state, "movies": close}

    return {**state, "movies": []}


def retrieve(state: State) -> State:
    """Semantic vector search for discover queries."""
    results = vdb.query_movies(state["query"], language_code=state["language_code"])
    movies = _rank_discover(
        _results_to_movies(results), state["query"], lite_mood=state["is_lite_mood"]
    ) if results else []
    return {**state, "movies": movies}


def fetch_discover(state: State) -> State:
    """Fallback for discover: hit TMDB discover API, then re-query."""
    print("[Fallback] Fetching discover results from TMDB…")
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
    results = vdb.query_movies(state["query"], language_code=state["language_code"])
    movies = _rank_discover(
        _results_to_movies(results), state["query"], lite_mood=state["is_lite_mood"]
    ) if results else []
    return {**state, "movies": movies}


def respond_search(state: State) -> State:
    """Respond to a specific movie search — show only the matched movie."""
    movies, query, history = state["movies"], state["query"], state["history"]
    title = state["search_title"] or query

    if not movies:
        return {**state, "response": (
            f"Sorry, I couldn't find **{title}** anywhere. "
            "Please check the spelling or try a different title."
        )}

    movie = movies[0]
    movie_block = _format_movies([movie])

    system = (
        "You are a movie expert. The user asked about a specific movie. "
        "Use ONLY the data provided below to answer. "
        "Give a brief, natural 2–3 sentence description covering genre, plot, and why it's worth watching. "
        "Do NOT mention other movies."
    )
    user_msg = f'User asked about: "{title}"\n\nMovie data:\n{movie_block}'
    commentary = _llm.invoke(_build_messages(system, history, user_msg)).content.strip()
    return {**state, "response": f"{commentary}\n\n{movie_block}"}


def respond_discover(state: State) -> State:
    """Respond to a discover/recommendation query with up to 10 movies."""
    movies, query, history = state["movies"], state["query"], state["history"]

    if not movies:
        return {**state, "response": "Sorry, I couldn't find matching movies. Try rephrasing!"}

    movie_block = _format_movies(movies)
    system = (
        "You are a friendly movie recommender. In 2–3 sentences, explain why "
        "the movies below match the user's mood or request. "
        "Do NOT invent movies — refer only to the list given."
    )
    user_msg = f'User request: "{query}"\n\nMatched movies:\n{movie_block}'
    commentary = _llm.invoke(_build_messages(system, history, user_msg)).content.strip()
    return {**state, "response": f"{commentary}\n\n{movie_block}"}


def followup(state: State) -> State:
    """Answer follow-up questions using only conversation history."""
    system = (
        "You are a movie expert. The conversation history contains recommended movies. "
        "Answer the user's follow-up question using ONLY movies already in the history. "
        "Be concise (2–4 sentences). Do NOT suggest new movies unless explicitly asked."
    )
    response = _llm.invoke(_build_messages(system, state["history"], state["query"])).content.strip()
    return {**state, "response": response}


def chat(state: State) -> State:
    """Handle greetings and small talk."""
    system = "You are a friendly movie chatbot. Keep answers brief and warm."
    response = _llm.invoke(_build_messages(system, state["history"], state["query"])).content.strip()
    return {**state, "response": response}


# ── Routing ───────────────────────────────────────────────────────────────────

def _route_intent(state: State) -> Literal["chat", "followup", "search_movie", "retrieve"]:
    intent = state["intent"]
    if intent == "chat":        return "chat"
    if intent == "followup":    return "followup"
    if intent == "search":      return "search_movie"
    return "retrieve"


def _needs_discover_fallback(state: State) -> Literal["fetch_discover", "respond_discover"]:
    return "fetch_discover" if len(state["movies"]) < 3 else "respond_discover"


# ── Build graph ───────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    g = StateGraph(State)

    g.add_node("classify",        classify)
    g.add_node("search_movie",    search_movie)
    g.add_node("retrieve",        retrieve)
    g.add_node("fetch_discover",  fetch_discover)
    g.add_node("respond_search",  respond_search)
    g.add_node("respond_discover",respond_discover)
    g.add_node("followup",        followup)
    g.add_node("chat",            chat)

    g.set_entry_point("classify")
    g.add_conditional_edges("classify",  _route_intent)
    g.add_edge("search_movie",           "respond_search")   # always go straight to respond
    g.add_conditional_edges("retrieve",  _needs_discover_fallback)
    g.add_edge("fetch_discover",         "respond_discover")
    g.add_edge("respond_search",         END)
    g.add_edge("respond_discover",       END)
    g.add_edge("followup",               END)
    g.add_edge("chat",                   END)

    return g.compile()


_graph = build_graph()


# ── Public API ────────────────────────────────────────────────────────────────

def run(query: str, history: list[dict] | None = None) -> tuple[str, list[dict]]:
    """
    Run one chatbot turn.

You: Find Inception
[Search] Looking for title: 'Inception'
[Search] Not in DB → fetching from TMDB API…
[TMDB] Searching for 'Inception'…
[TMDB] Found 12 movies.
[VectorDB] Upserted 12 movies.
[Search] TMDB returned: Inception

Chatbot: Inception is a mind-bending action film that delves into the world of dream-sharing, where a skilled thief, Cobb, is tasked with planting an idea in someone's mind instead of stealing one. As he navigates the complex layers of his own subconscious, Cobb must confront his past and the blurred lines between reality and dreams. With its thought-provoking premise and stunning visuals, Inception is a thrilling ride that will keep you on the edge of your seat.

1. Inception (2010) | Action, Science Fiction, Adventure | ⭐ 8.372/10
   Cobb, a skilled thief who commits corporate espionage by infiltrating the subconscious of his targets is offered a chance to regain his old life as payment for a task considered to be impossible: "inc

    Args:
        query:   User message.
        history: Previous [{"role": ..., "content": ...}] turns.

    Returns:
        (response_text, updated_history)
    """
    history = history or []
    final_state = _graph.invoke({
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
        "response": "",
    })
    response = final_state["response"]
    updated_history = history + [
        {"role": "user",      "content": query},
        {"role": "assistant", "content": response},
    ]
    return response, updated_history