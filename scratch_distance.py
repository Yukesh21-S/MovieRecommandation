import chromadb
from chromadb.utils import embedding_functions

# Use the same single embedding function as the main application
st_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(path="./chroma_db")

# Ensure we use st_ef to match the data stored in the database
try:
    collection = client.get_collection(name="movie_collection", embedding_function=st_ef)
    
    queries = [
        "A movie about space travel",
        "A movie about a talking dog playing basketball",
        "A french romantic comedy",
    ]

    for q in queries:
        results = collection.query(query_texts=[q], n_results=1)
        if results['distances'] and results['distances'][0]:
            distance = results['distances'][0][0]
            title = results['metadatas'][0][0]['title']
            print(f"Query: '{q}'\nClosest: {title} (Distance: {distance:.4f})\n")
        else:
            print(f"Query: '{q}'\nNo results found.\n")
except Exception as e:
    print(f"Error accessing collection: {e}")
