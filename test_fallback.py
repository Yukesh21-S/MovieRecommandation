import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from rag_app import get_recommendations, check_relevance, generate_response, search_movies, add_movies_to_db

query = "Space Jam"
print(f"Testing Query: {query}")
results = get_recommendations(query)

if not results or not check_relevance(query, results):
    print("\n[!] Local DB missing necessary data. Falling back to TMDB API...")
    new_movies = search_movies(query)
    
    if new_movies:
        add_movies_to_db(new_movies)
        print("\nRe-querying updated database...")
        results = get_recommendations(query)
    else:
        print("TMDB API also couldn't find matching movies.")

if results:
    response = generate_response(query, results)
    print("\n--- Final Recommendation ---")
    print(response)
