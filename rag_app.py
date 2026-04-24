import chromadb
from chromadb.utils import embedding_functions
import ollama
import re
import json as _json
import sys
import os
from dotenv import load_dotenv
from groq import Groq

from data_fetcher import search_movies, discover_movies, get_person_id, LANGUAGE_MAP, GENRE_MAP
from vector_engine import add_movies_to_db

load_dotenv()

# Mood to Genre mapping for TMDB discover
MOOD_MAP = {
    "motivation": "drama",
    "motivational": "drama",
    "inspire": "drama",
    "inspiration": "drama",
    "inspirational": "drama",
    "feel good": "comedy",
    "sad": "drama",
    "happy": "comedy",
    "better": "comedy",
    "excited": "action",
    "scared": "horror",
    "romantic": "romance"
}

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# ── Single embedding function used everywhere ──────────────────────────────
st_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
# ── Configuration ────────────────────────────────────────────────────────
DB_PATH = "./chroma_db"
COLLECTION_NAME = "movie_collection"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if LLM_PROVIDER == "groq" and GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
    INTENT_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
    CHAT_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
else:
    INTENT_MODEL = "phi3:mini"
    CHAT_MODEL = "gemma2:2b"

# ── Conversation memory (persists for the whole session) ───────────────────
chat_history = []   # List of {"role": "user"/"assistant", "content": "..."}

def _llm(messages, model=CHAT_MODEL):
    """Call the configured LLM provider (Ollama or Groq)."""
    try:
        if LLM_PROVIDER == "groq" and GROQ_API_KEY:
            completion = groq_client.chat.completions.create(
                model=model,
                messages=messages
            )
            return completion.choices[0].message.content.strip()
        else:
            response = ollama.chat(model=model, messages=messages)
            return response['message']['content'].strip()
    except Exception as e:
        print(f"  LLM Error ({LLM_PROVIDER}): {e}")
        return None

def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        return client.get_collection(name=COLLECTION_NAME, embedding_function=st_ef)
    except Exception:
        return None

def get_recommendations(query, language_code=None, n_results=15):  # fetch more so filtering still leaves 10
    col = get_collection()
    if col is None:
        return None
    
    query_params = {
        "query_texts": [query],
        "n_results": n_results
    }
    if language_code:
        query_params["where"] = {"language": language_code}
        
    results = col.query(**query_params)
    if not results['documents'][0]:
        return None
    return results

def filter_results(results, min_year=None, min_rating=None, max_results=10):
    """Post-filter ChromaDB results by year and rating."""
    if results is None:
        return None
    filtered = {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
    for i, meta in enumerate(results['metadatas'][0]):
        rating = meta.get('vote_average', 0.0)
        release = meta.get('release_date', '') or ''
        year = int(release[:4]) if release and len(release) >= 4 and release[:4].isdigit() else 0

        if min_rating and rating < min_rating:
            continue
        if min_year and year < min_year:
            continue

        filtered['documents'][0].append(results['documents'][0][i])
        filtered['metadatas'][0].append(meta)
        filtered['distances'][0].append(results['distances'][0][i])

        if len(filtered['documents'][0]) >= max_results:
            break

    return filtered if filtered['documents'][0] else None

def check_relevance(query, search_results):
    """Ask the LLM — using conversation history — if results match the query."""
    context = "\n".join(
        search_results['metadatas'][0][i]['title']
        for i in range(min(5, len(search_results['metadatas'][0])))
    )
    messages = [{
        "role": "user",
        "content": (
            f"You are a strict relevance evaluator.\n"
            f"User query: \"{query}\"\n"
            f"Retrieved movies:\n{context}\n\n"
            f"Do these movies clearly and specifically match the user's request?\n"
            f"If the user asked for a specific actor or director, and their name is NOT in the retrieved movies, you MUST answer NO.\n"
            f"Answer exactly YES or NO."
        )
    }]
    answer = _llm(messages, model=INTENT_MODEL) or "YES"
    return "YES" in answer.upper()

def extract_query_intent(query):
    """Detect if user wants discover (language/genre) or title search, and if 'latest'."""
    LATEST_WORDS = ["latest", "new", "recent", "newest", "2024", "2025", "2023", "2022"]
    DISCOVER_KEYWORDS = ["suggest", "recommend", "movies", "feel", "sad", "happy", "motivation", "motivational", "inspire", "inspiration", "best", "top", "good"]
    
    query_lower = query.lower()
    is_latest = any(w in query_lower for w in LATEST_WORDS)
    is_recommendation = any(k in query_lower for k in DISCOVER_KEYWORDS) or len(query.split()) > 4

    lang_list  = ", ".join(LANGUAGE_MAP.keys())
    genre_list = ", ".join(GENRE_MAP.keys())

    prompt = f"""
Classify the user movie query into ONE of these types:
1. "search" -> if they are asking for a SPECIFIC movie title (e.g. "find Inception")
2. "discover" -> if they want recommendations based on mood, genre, language, or person (e.g. "suggest motivational movies")
3. "chat" -> if they are just greeting you (e.g. "hi", "hello"), asking how you are, or making small talk.

Rules:
- If the query is just a greeting (hi, hello, etc.) -> ALWAYS "chat"
- If the query is a full sentence or contains words like "suggest", "recommend", "mood", "feel" -> ALWAYS "discover"
- Only use "search" if it's clearly a specific movie name or very short (1-3 words) title request.

Return a JSON object with:
- "type": "discover", "search", or "chat"
- "language": one of ({lang_list}) or null
- "genre": one of ({genre_list}) or null
- "person": name of director/actor or null
- "keywords": specific movie title keywords if type is "search", else null

User query: "{query}"
JSON:"""

    intent = {
        'type': 'discover' if is_recommendation else 'search', 
        'language_code': None,
        'genre_id': None, 'person': None, 'keywords': None, 'is_latest': is_latest
    }

    # Manual Greeting check
    if query_lower in ["hi", "hello", "hey", "hola", "namaste"]:
        intent['type'] = 'chat'
        return intent

    # Manual Mood Mapping
    for mood, genre in MOOD_MAP.items():
        if mood in query_lower:
            intent['genre_id'] = GENRE_MAP.get(genre)
            intent['type'] = 'discover'

    # Try LLM intent detection (do NOT include chat history, it confuses the JSON output)
    messages = [{"role": "user", "content": prompt}]
    content = _llm(messages, model=INTENT_MODEL)
    if content:
        match = re.search(r'\{.*?\}', content, re.DOTALL)
        if match:
            try:
                parsed    = _json.loads(match.group())
                lang_name  = (parsed.get('language') or '').lower()
                genre_name = (parsed.get('genre')    or '').lower()
                
                # Update intent with LLM results, but respect manual overrides for type
                intent.update({
                    'type':          parsed.get('type', intent['type']),
                    'language_code': LANGUAGE_MAP.get(lang_name) or intent['language_code'],
                    'genre_id':      GENRE_MAP.get(genre_name) or intent['genre_id'],
                    'person':        parsed.get('person'),
                    'keywords':      parsed.get('keywords'),
                })
                
                # Double check for recommendation intent
                if is_recommendation:
                    intent['type'] = 'discover'
                
                return intent
            except Exception:
                pass

    # Keyword fallback
    for lang_name, lang_code in LANGUAGE_MAP.items():
        if lang_name in query_lower:
            intent['language_code'] = lang_code
            intent['type'] = 'discover'
            
    if not intent['keywords'] and intent['type'] == 'search':
        intent['keywords'] = query
        
    return intent

def build_movies_text(search_results):
    """Format all retrieved movies into a clean structured block."""
    movies_text = ""
    context_lines = []
    for i, doc in enumerate(search_results['documents'][0]):
        meta     = search_results['metadatas'][0][i]
        title    = meta['title']
        genres   = meta['genres']
        rating   = meta['vote_average']
        release  = meta.get('release_date', 'Unknown')
        year     = release[:4] if release and release != 'Unknown' else '???'
        overview = doc.split("Overview: ")[1] if "Overview: " in doc else doc

        movies_text += (
            f"{i+1}. {title} ({year})\n"
            f"   Genre   : {genres}\n"
            f"   Rating  : {rating}/10\n"
            f"   Overview: {overview}\n\n"
        )
        context_lines.append(f"{i+1}. {title} ({year}) | {genres} | {rating}/10")

    return movies_text, "\n".join(context_lines)

def generate_response(query, search_results):
    movies_text, context_summary = build_movies_text(search_results)

    # Add conversation history + current context to get a relevant intro
    summary_prompt = (
        f"You are a friendly movie recommendation assistant.\n"
        f"The user asked: \"{query}\"\n"
        f"You found these movies:\n{context_summary}\n\n"
        f"Write a SHORT 2-3 sentence friendly intro. Do NOT list the movies again."
    )
    messages  = chat_history + [{"role": "user", "content": summary_prompt}]
    commentary = _llm(messages, model=CHAT_MODEL) or "Here are the movies I found for you:"

    return f"{commentary}\n\n{movies_text}"

def handle_followup(query):
    """Handle follow-up questions using only the conversation history (no DB search needed)."""
    messages = chat_history + [{"role": "user", "content": query}]
    return _llm(messages, model=CHAT_MODEL) or "I'm not sure — could you rephrase your question?"

def is_followup(query):
    """Detect if the user is asking a follow-up about previously mentioned movies."""
    FOLLOWUP_HINTS = [
        "first", "second", "third", "that", "those", "it", "them", "him", "her",
        "movie", "film", "story", "plot", "elaborate", "details", "more",
        "movie 1", "movie 2", "movie 3", "number", "tell me more",
        "what about", "and the", "overview of", "rating of", "which one",
        "who directed", "who acted", "stars in"
    ]
    q = query.lower().strip()
    
    # 1. Direct keyword hints
    if "the movie" in q or "the story" in q or "the plot" in q:
        return True
    if any(h in q for h in FOLLOWUP_HINTS) and len(query.split()) < 20:
        return True

    # 2. Check if the query is a title of a movie we just mentioned
    # We look for the last 'assistant' message that contains our recommendation list
    for msg in reversed(chat_history[:-1]): # Exclude the current user message
        if msg['role'] == 'assistant' and 'I recommended these movies:' in msg['content']:
            content = msg['content'].lower()
            # If the user typed something that is exactly in our list (like "Leo")
            # We use word boundaries to avoid matching "in" within "inception"
            if re.search(rf"\b{re.escape(q)}\b", content):
                return True
            break
            
    return False

def main():
    print("================================================")
    print(f"  🎬 Movie Chatbot ({LLM_PROVIDER.upper()}) 🎬")
    print("================================================")
    if LLM_PROVIDER == "groq":
        print(f"  Using Cloud Model: {CHAT_MODEL}")
    else:
        print(f"  Using Local Models: {INTENT_MODEL} + {CHAT_MODEL}")
    print("================================================")
    print("  Ask anything! e.g. 'latest hindi movies'")
    print("  Follow-up: 'tell me more about the first one'")
    print("  Type 'quit' to exit.")
    print("================================================\n")

    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye! 🎬")
            break

        # -- Add user message to history --
        chat_history.append({"role": "user", "content": query})

        # -- Check if it's a follow-up question --
        if is_followup(query) and len(chat_history) > 2:
            print("[Chatbot] Answering follow-up from conversation memory...")
            response = handle_followup(query)
            print(f"\nChatbot: {response}\n")
            chat_history.append({"role": "assistant", "content": response})
            continue

        # -- Extract Intent FIRST so we know the language --
        intent = extract_query_intent(query)
        print(f"  Intent: type={intent['type']}, lang={intent['language_code']}, genre={intent['genre_id']}, latest={intent['is_latest']}")

        # -- Handle simple chat/greetings --
        if intent['type'] == 'chat':
            response = handle_followup(query)
            print(f"\nChatbot: {response}\n")
            chat_history.append({"role": "assistant", "content": response})
            continue

        # -- Vector DB search with language filter --
        results = get_recommendations(query, language_code=intent['language_code'])
        results = filter_results(results, min_year=2022 if intent['is_latest'] else None, min_rating=0.1)

        # -- Fallback to TMDB if DB results are not relevant, or if a specific person was requested --
        if not results or intent.get('person') or not check_relevance(query, results):
            print("[Chatbot] Fetching more movies from TMDB...")
            
            if intent['type'] == 'discover':
                sort_by  = "release_date.desc" if intent['is_latest'] else "popularity.desc"
                year_gte = 2022 if intent['is_latest'] else None
                person_id = None
                if intent.get('person'):
                    person_id = get_person_id(intent['person'])
                    print(f"  Resolved person '{intent['person']}' to ID: {person_id}")

                new_movies = discover_movies(
                    language_code=intent['language_code'],
                    genre_id=intent['genre_id'],
                    sort_by=sort_by,
                    year_gte=year_gte,
                    person_id=person_id,
                    pages=3
                )
            else:
                keywords   = intent['keywords'] or query
                print(f"  Searching TMDB by title: '{keywords}'")
                new_movies = search_movies(keywords)

            if new_movies:
                add_movies_to_db(new_movies)
                results = get_recommendations(query, language_code=intent['language_code'])
                results = filter_results(results, min_year=2022 if intent['is_latest'] else None, min_rating=0.1)

        # -- Generate and print response --
        if results:
            response = generate_response(query, results)
            print(f"\nChatbot: {response}")
            # Store a compact version in history so the model knows what was shown
            _, context_summary = build_movies_text(results)
            chat_history.append({
                "role": "assistant",
                "content": f"I recommended these movies:\n{context_summary}"
            })
        else:
            msg = "Sorry, I couldn't find any movies matching your request. Try rephrasing!"
            print(f"\nChatbot: {msg}\n")
            chat_history.append({"role": "assistant", "content": msg})

if __name__ == "__main__":
    main()
