import json
import chromadb
from chromadb.utils import embedding_functions

# Use SentenceTransformers for embeddings
st_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

def setup_vector_db(json_file="movies_data.json", db_path="./chroma_db"):
    print(f"Loading data from {json_file}...")
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            movies = json.load(f)
    except FileNotFoundError:
        print(f"Error: {json_file} not found. Please run data_fetcher.py first.")
        return

    print("Initializing ChromaDB...")
    client = chromadb.PersistentClient(path=db_path)
    
    collection_name = "movie_collection"
    
    # Try to delete the collection if it already exists to start fresh
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass
        
    collection = client.create_collection(
        name=collection_name, 
        embedding_function=st_ef
    )

    documents = []
    metadatas = []
    ids = []

    print(f"Preparing {len(movies)} movies for vectorization...")
    for idx, movie in enumerate(movies):
        # We combine title, genres, and overview for the semantic search
        content = f"Title: {movie['title']}\nGenres: {movie['genres']}\nOverview: {movie['overview']}"
        
        documents.append(content)
        metadatas.append({
            "title": movie["title"],
            "genres": movie["genres"],
            "release_date": movie["release_date"],
            "vote_average": movie["vote_average"],
            "language": movie.get("language", "en")
        })
        ids.append(str(movie["id"]))

    print("Adding movies to ChromaDB (this might take a minute depending on your local machine)...")
    # Batch add to avoid memory issues
    batch_size = 50
    for i in range(0, len(documents), batch_size):
        end_idx = min(i + batch_size, len(documents))
        print(f"Processing batch {i} to {end_idx}...")
        collection.add(
            documents=documents[i:end_idx],
            metadatas=metadatas[i:end_idx],
            ids=ids[i:end_idx]
        )

    print("Vector database built successfully!")

def add_movies_to_db(movies, db_path="./chroma_db"):
    if not movies:
        return
        
    client = chromadb.PersistentClient(path=db_path)
    try:
        collection = client.get_collection(name="movie_collection", embedding_function=st_ef)
    except Exception:
        print("Collection does not exist. Run setup_vector_db first.")
        return

    documents = []
    metadatas = []
    ids = []

    for movie in movies:
        content = f"Title: {movie['title']}\nGenres: {movie['genres']}\nOverview: {movie['overview']}"
        documents.append(content)
        metadatas.append({
            "title": movie["title"],
            "genres": movie["genres"],
            "release_date": movie["release_date"],
            "vote_average": movie["vote_average"],
            "language": movie.get("language", "en")
        })
        ids.append(str(movie["id"]))

    print(f"Adding {len(movies)} new movies to ChromaDB...")
    # Add to DB, Chroma handles ignoring existing IDs if we use upsert or if we just catch the warning
    # We will use upsert to update if exists or insert if new
    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print("New movies added to the database successfully.")

if __name__ == "__main__":
    setup_vector_db()
