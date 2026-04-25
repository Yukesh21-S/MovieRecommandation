"""
ChromaDB vector store for semantic movie search.
"""

import json

import chromadb
from chromadb.utils import embedding_functions

DB_PATH = "./chroma_db"
COLLECTION_NAME = "movie_collection"

_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


def _get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=DB_PATH)


def get_collection() -> chromadb.Collection | None:
    try:
        return _get_client().get_collection(COLLECTION_NAME, embedding_function=_embedding_fn)
    except Exception:
        return None


def _movie_to_doc(movie: dict) -> tuple[str, dict, str]:
    doc = f"Title: {movie['title']}\nGenres: {movie['genres']}\nOverview: {movie['overview']}"
    meta = {
        "title": movie["title"],
        "genres": movie["genres"],
        "release_date": movie["release_date"],
        "vote_average": movie["vote_average"],
        "language": movie.get("language", "en"),
    }
    return doc, meta, str(movie["id"])


def setup_vector_db(json_path: str = "movies_data.json") -> None:
    print(f"[VectorDB] Loading movies from {json_path}...")
    with open(json_path, encoding="utf-8") as file:
        movies: list[dict] = json.load(file)

    client = _get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME, embedding_function=_embedding_fn)

    docs, metas, ids = zip(*[_movie_to_doc(movie) for movie in movies])

    batch = 50
    for i in range(0, len(docs), batch):
        collection.add(
            documents=list(docs[i : i + batch]),
            metadatas=list(metas[i : i + batch]),
            ids=list(ids[i : i + batch]),
        )
        print(f"[VectorDB] Indexed {min(i + batch, len(docs))}/{len(docs)} movies...")

    print("[VectorDB] Setup complete.")


def upsert_movies(movies: list[dict]) -> None:
    if not movies:
        return
    collection = get_collection()
    if collection is None:
        print("[VectorDB] Collection not found. Run setup_vector_db() first.")
        return

    # TMDB can return duplicate IDs across pages; Chroma upsert requires unique ids per call.
    deduped_by_id: dict[int, dict] = {}
    for movie in movies:
        movie_id = movie.get("id")
        if isinstance(movie_id, int):
            deduped_by_id[movie_id] = movie
    deduped_movies = list(deduped_by_id.values())
    if not deduped_movies:
        return

    docs, metas, ids = zip(*[_movie_to_doc(movie) for movie in deduped_movies])
    collection.upsert(documents=list(docs), metadatas=list(metas), ids=list(ids))
    print(f"[VectorDB] Upserted {len(deduped_movies)} movies.")


def query_movies(
    query_text: str,
    n_results: int = 15,
    language_code: str | None = None,
) -> dict | None:
    collection = get_collection()
    if collection is None:
        return None

    params: dict = {"query_texts": [query_text], "n_results": n_results}
    if language_code:
        params["where"] = {"language": language_code}

    results = collection.query(**params)
    return results if results["documents"][0] else None

