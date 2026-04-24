import os
import sys

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def initialize_system():
    """Check if the vector DB exists, build it if not."""
    import chromadb
    from chromadb.utils import embedding_functions
    
    st_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="./chroma_db")
    
    try:
        # Check if collection exists and has data
        col = client.get_collection(name="movie_collection", embedding_function=st_ef)
        count = col.count()
        if count > 0:
            print(f"  Vector DB ready with {count} movies.")
            return
        else:
            raise Exception("Empty collection")
    except Exception:
        print("  First time setup: Building the vector database with sentence_transformers...")
        # Delete old incompatible collection if any
        try:
            client.delete_collection(name="movie_collection")
        except Exception:
            pass

        import data_fetcher
        import vector_engine

        if not os.path.exists("movies_data.json"):
            print("  Fetching initial movie data from TMDB...")
            data_fetcher.fetch_movies(pages=5)

        vector_engine.setup_vector_db()
        print("  Setup complete!")

def main():
    print("Initializing Movie Chatbot...")
    initialize_system()

    import rag_app
    rag_app.main()

if __name__ == "__main__":
    main()
