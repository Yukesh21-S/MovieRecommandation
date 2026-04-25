"""
vector_engine.py
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
    """Return the movie collection, or None if it doesn't exist."""
    try:
        return _get_client().get_collection(COLLECTION_NAME, embedding_function=_embedding_fn)
    except Exception:
        return None


def _movie_to_doc(movie: dict) -> tuple[str, dict, str]:
    """Convert a movie dict to (document, metadata, id)."""
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
    """Create and populate the ChromaDB collection from a JSON file."""
    print(f"[VectorDB] Loading movies from {json_path}…")
    with open(json_path, encoding="utf-8") as f:
        movies: list[dict] = json.load(f)

    client = _get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME, embedding_function=_embedding_fn)

    docs, metas, ids = zip(*[_movie_to_doc(m) for m in movies])

    batch = 50
    for i in range(0, len(docs), batch):
        collection.add(
            documents=list(docs[i : i + batch]),
            metadatas=list(metas[i : i + batch]),
            ids=list(ids[i : i + batch]),
        )
        print(f"[VectorDB] Indexed {min(i + batch, len(docs))}/{len(docs)} movies…")

    print("[VectorDB] Setup complete.")


def upsert_movies(movies: list[dict]) -> None:
    """Add or update movies in an existing collection."""
    if not movies:
        return
    col = get_collection()
    if col is None:
        print("[VectorDB] Collection not found. Run setup_vector_db() first.")
        return

    docs, metas, ids = zip(*[_movie_to_doc(m) for m in movies])
    col.upsert(documents=list(docs), metadatas=list(metas), ids=list(ids))
    print(f"[VectorDB] Upserted {len(movies)} movies.")

def query_movies(
    query_text: str,
    n_results: int = 15,
    language_code: str | None = None,
) -> dict | None:
    """Query the vector store and return raw ChromaDB results."""
    col = get_collection()
    if col is None:
        return None

    params: dict = {"query_texts": [query_text], "n_results": n_results}
    if language_code:
        params["where"] = {"language": language_code}

    results = col.query(**params)
    return results if results["documents"][0] else None


if __name__ == "__main__":
    setup_vector_db()