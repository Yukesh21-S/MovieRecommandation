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
    "stress": "comedy",
    "stressed": "comedy",
    "tired": "comedy",
    "relax": "comedy",
    "light": "comedy",
    "lite": "comedy",
    "fun": "comedy",
    "funny": "comedy",
    "happy": "comedy",
    "feel good": "comedy",
    "sad": "drama",
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

# ── Caches and Session Data ────────────────────────────────────────────────
chat_history = []   # List of {"role": "user"/"assistant", "content": "..."}
results_cache = {}  # Cache query -> processed results
last_movie_title = None # Tracks the most recently discussed movie title

def is_specific_movie(query):
    """Detect if the user is asking for a specific movie title."""
    q = query.lower()
    # patterns for movie search
    if "movie" in q and ("called" in q or "named" in q or "about" in q):
        return True
    if len(query.split()) <= 5 and not any(k in q for k in ["suggest", "recommend", "show"]):
        return True
    return False

def extract_movie_name(query):
    """Extract the raw movie name from a natural language query."""
    # Look for the last occurrence of these words to get the actual title
    match = re.search(r".*(called|named|about|for)\s+(.*)", query.lower())
    if match:
        return match.group(2).strip()
    return query

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

def filter_results(results, query="", min_year=None, min_rating=None, max_results=10):
    """Post-filter and RANK results using a weighted score + keyword boost."""
    if results is None:
        return None
        
    query_words = set(re.findall(r'\w+', query.lower()))
    scored_items = []
    for i, meta in enumerate(results['metadatas'][0]):
        rating = meta.get('vote_average', 0.0)
        distance = results['distances'][0][i]
        title = meta['title'].lower()
        release = meta.get('release_date', '') or ''
        year = int(release[:4]) if release and len(release) >= 4 and release[:4].isdigit() else 0

        if min_rating and rating < min_rating:
            continue
        if min_year and year < min_year:
            continue

        # 1. Base Score: 50% Similarity (1-dist), 30% Rating
        similarity = max(0, 1 - distance)
        score = (similarity * 0.5) + ((rating / 10.0) * 0.3)
        
        # 2. Mood Boost: If "comedy" is in genres, give it a 20% boost
        if "comedy" in meta['genres'].lower():
            score += 0.2
            
        # 3. Hybrid Keyword Boost: If query word is in the title, boost score significantly
        title_words = set(re.findall(r'\w+', title))
        if query_words.intersection(title_words):
            score += 0.3 # Large boost for exact keyword matches
        
        scored_items.append({
            'doc': results['documents'][0][i],
            'meta': meta,
            'dist': distance,
            'score': score
        })

    # Sort by the final boosted score
    scored_items.sort(key=lambda x: x['score'], reverse=True)
    top_items = scored_items[:max_results]

    if not top_items:
        return None

    return {
        'documents': [[x['doc'] for x in top_items]],
        'metadatas': [[x['meta'] for x in top_items]],
        'distances': [[x['dist'] for x in top_items]]
    }

def filter_by_mood(results, query):
    """If user wants 'lite' mood, strictly filter for Comedy, Family, or Romance."""
    if not results or not results['metadatas'][0]:
        return results
        
    lite_keywords = ["lite", "light", "stress", "tired", "relax", "fun", "happy"]
    if not any(k in query.lower() for k in lite_keywords):
        return results # Not a lite mood request
        
    print("  [System] Strict Mood Filtering: Keeping only light-hearted genres...")
    filtered = {'documents':[[]], 'metadatas':[[]], 'distances':[[]]}
    for i, meta in enumerate(results['metadatas'][0]):
        genres = meta.get("genres", "").lower()
        if any(g in genres for g in ["comedy", "family", "romance"]):
            filtered['documents'][0].append(results['documents'][0][i])
            filtered['metadatas'][0].append(meta)
            filtered['distances'][0].append(results['distances'][0][i])
            
    return filtered if filtered['documents'][0] else results

def rerank_results(query, results):
    """Use the LLM to rerank the top results based on true semantic meaning."""
    if not results or not results['metadatas'][0]:
        return results
        
    titles = [m['title'] for m in results['metadatas'][0]]
    prompt = (
        f"User query: \"{query}\"\n"
        f"List of movies: {', '.join(titles)}\n\n"
        f"Re-order these movies from most relevant to least relevant for the query.\n"
        f"Return ONLY the titles as a comma-separated list. Do not explain anything."
    )
    
    reranked_text = _llm([{"role": "user", "content": prompt}], model=INTENT_MODEL)
    if not reranked_text:
        return results
        
    # Re-order the results dictionary based on the LLM's preferred order
    order = [t.strip().lower() for t in reranked_text.split(',')]
    
    new_docs, new_metas, new_dists = [], [], []
    for title_name in order:
        for i, m in enumerate(results['metadatas'][0]):
            if m['title'].lower() == title_name:
                new_docs.append(results['documents'][0][i])
                new_metas.append(m)
                new_dists.append(results['distances'][0][i])
                break
                
    # Add any missing ones at the end
    for i, m in enumerate(results['metadatas'][0]):
        if m['title'] not in [x['title'] for x in new_metas]:
            new_docs.append(results['documents'][0][i])
            new_metas.append(m)
            new_dists.append(results['distances'][0][i])

    return {
        'documents': [new_docs],
        'metadatas': [new_metas],
        'distances': [new_dists]
    }

def check_relevance(query, search_results):
    """Hybrid Relevance: Combined distance threshold and LLM check."""
    if not search_results or not search_results['metadatas'][0]:
        return False
        
    # 1. Distance Threshold Check (Tightened)
    avg_dist = sum(search_results['distances'][0][:3]) / 3 if len(search_results['distances'][0]) >= 3 else 2.0
    if avg_dist > 0.7: # High threshold for better precision
        return False

    # 2. LLM Reasoning Check
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
    DISCOVER_KEYWORDS = ["suggest", "recommend", "show", "get", "find", "movies", "film", "recommendation", "best", "top", "good", "latest", "new"]
    
    query_lower = query.lower()
    is_latest = any(w in query_lower for w in LATEST_WORDS)
    # Only treat as recommendation if it contains a keyword AND is long enough, or starts with a verb
    is_recommendation = any(k in query_lower for k in DISCOVER_KEYWORDS) and len(query.split()) > 2

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

    # Manual Greeting and Emotional check
    CHAT_KEYWORDS = ["hi", "hello", "hey", "hola", "namaste", "how are you", "what's up", "good morning", "good evening", "mood", "feeling", "was in", "i am", "i'm"]
    if any(k in query_lower for k in CHAT_KEYWORDS) and not any(k in query_lower for k in ["movie", "show", "suggest"]):
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

def generate_response(query, search_results, is_specific=False):
    movies_text, context_summary = build_movies_text(search_results)

    if is_specific:
        summary_prompt = (
            f"User asked about a specific movie: \"{query}\"\n"
            f"Retrieved movie data:\n{context_summary}\n\n"
            f"Use ONLY the retrieved movie data above.\n"
            f"Do NOT suggest other movies.\n"
            f"Do NOT assume similarities to other films.\n"
            f"Explain the movie briefly and naturally. 2-3 sentences max."
        )
    else:
        # Advanced Generation: Reasoning over the context
        summary_prompt = (
            f"You are a movie expert who understands user moods and preferences.\n"
            f"The user said: \"{query}\"\n"
            f"You chose these movies based on their similarity, rating, and relevance:\n{context_summary}\n\n"
            f"Explain in 3 sentences WHY these specific movies were selected for their request.\n"
            f"Mention specific themes from the movies that match the user's mood."
        )
        
    messages  = chat_history + [{"role": "user", "content": summary_prompt}]
    commentary = _llm(messages, model=CHAT_MODEL) or "Based on your interest, I've selected these highly-rated matches:"

    return f"{commentary}\n\n{movies_text}"

def is_short_query(query):
    return len(query.split()) <= 6

def handle_followup(query):
    """Handle follow-up questions with conditional verbosity."""
    if is_short_query(query):
        prompt = f"""User question: "{query}"
Give a SHORT answer (2–4 lines).
- Be simple and conversational
- Give opinion (good / average / worth watching)
- NO awards, NO trivia, NO long explanation
- Do NOT invent facts
Answer:"""
    else:
        prompt = query

    messages = chat_history + [{"role": "user", "content": prompt}]
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

        # -- Context Tracking & Resolution --
        global last_movie_title
        q_lower = query.lower()
        if any(h in q_lower for h in ["how is it", "this movie", "see this", "about it"]):
            if last_movie_title:
                query = f"Give short opinion about {last_movie_title}"
                print(f"  [Context] Resolved 'it' to: {last_movie_title}")

        # -- Add user message to history --
        chat_history.append({"role": "user", "content": query})

        # -- Check if it's a follow-up question --
        if is_followup(query) and len(chat_history) > 2:
            print("[Chatbot] Answering follow-up from conversation memory...")
            
            # If the user typed just a movie name, update last_movie_title
            if len(query.split()) < 4:
                last_movie_title = query

            response = handle_followup(query)
            print(f"\nChatbot: {response}\n")
            chat_history.append({"role": "assistant", "content": response})
            continue

        # -- Extract Intent FIRST --
        intent = extract_query_intent(query)
        
        # Force genre if discover intent detected but no genre identified
        if intent['type'] == 'discover' and not intent['genre_id']:
            if any(k in query.lower() for k in ["lite", "light", "stress", "tired", "relax", "fun", "happy"]):
                intent['genre_id'] = GENRE_MAP.get("comedy")
                
        # -- Force Intent Override for Specific Titles --
        if is_specific_movie(query) and intent['type'] != 'chat':
            intent['type'] = 'search'
            intent['keywords'] = extract_movie_name(query)
            print(f"  [Intent] Overridden to 'search' for title: {intent['keywords']}")
        
        print(f"  Intent: type={intent['type']}, lang={intent['language_code']}, genre={intent['genre_id']}, latest={intent['is_latest']}")

        # -- Handle simple chat/greetings --
        if intent['type'] == 'chat':
            response = handle_followup(query)
            print(f"\nChatbot: {response}\n")
            chat_history.append({"role": "assistant", "content": response})
            continue

        # -- Check Cache First --
        results = None
        if query in results_cache:
            results = results_cache[query]
        else:
            # -- Search Logic --
            if intent['type'] == 'search':
                # Skip vector DB for exact title search to ensure accuracy
                keywords = intent['keywords'] or query
                print(f"  Searching TMDB directly for exact title: '{keywords}'")
                new_movies = search_movies(keywords)
                if new_movies:
                    add_movies_to_db(new_movies)
                    results = get_recommendations(keywords) # Query specifically for the title
                    if results:
                        results = filter_results(results, query=keywords)
            else:
                # -- Discover Logic (Mood/Genre/etc) --
                results = get_recommendations(query, language_code=intent['language_code'])
                results = filter_results(results, query=query, min_year=2022 if intent['is_latest'] else None, min_rating=0.1)
                results = filter_by_mood(results, query)

                # Fallback if discovery failed
                if not results or intent.get('person') or not check_relevance(query, results):
                    print("[Chatbot] Fetching more discovery results from TMDB...")
                    sort_by  = "release_date.desc" if intent['is_latest'] else "popularity.desc"
                    year_gte = 2022 if intent['is_latest'] else None
                    person_id = get_person_id(intent['person']) if intent.get('person') else None
                    new_movies = discover_movies(language_code=intent['language_code'], genre_id=intent['genre_id'], sort_by=sort_by, year_gte=year_gte, person_id=person_id, pages=3)
                    
                    if new_movies:
                        add_movies_to_db(new_movies)
                        results = get_recommendations(query, language_code=intent['language_code'])
                        results = filter_results(results, query=query, min_year=2022 if intent['is_latest'] else None, min_rating=0.1)
                        results = filter_by_mood(results, query)
            
            # -- Only Rerank if discovery results (> 5 results) --
            if results and intent['type'] == 'discover' and len(results['metadatas'][0]) > 5:
                results = rerank_results(query, results)
                
            if results:
                results_cache[query] = results

        # -- Generate and print response --
        if results:
            response = generate_response(query, results, is_specific=(intent['type'] == 'search'))
            print(f"\nChatbot: {response}")
            # Store a compact version in history so the model knows what was shown
            _, context_summary = build_movies_text(results)
            # Update last_movie_title if we found a specific title or just one result
            if results['metadatas'][0]:
                last_movie_title = results['metadatas'][0][0]['title']

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
