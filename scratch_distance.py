import chromadb
from chromadb.utils import embedding_functions

ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name="nomic-embed-text"
)

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="movie_collection", embedding_function=ollama_ef)

queries = [
    "A movie about space travel", # Likely in the DB (Interstellar, etc.)
    "A movie about a talking dog playing basketball", # Unlikely in top 100 popular
    "A french romantic comedy",
]

for q in queries:
    results = collection.query(query_texts=[q], n_results=1)
    distance = results['distances'][0][0] if results['distances'] else 'N/A'
    title = results['metadatas'][0][0]['title'] if results['metadatas'] else 'N/A'
    print(f"Query: '{q}'\nClosest: {title} (Distance: {distance})\n")
