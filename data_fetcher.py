import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("TMDB_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"

def get_genre_mapping():
    url = f"{BASE_URL}/genre/movie/list"
    params = {"api_key": API_KEY, "language": "en-US"}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        genres = response.json().get('genres', [])
        return {genre['id']: genre['name'] for genre in genres}
    return {}

def fetch_movies(pages=5):
    print("Fetching genre mapping...")
    genre_mapping = get_genre_mapping()
    
    all_movies = []
    print(f"Fetching movies from TMDB (Pages: {pages})...")
    
    for page in range(1, pages + 1):
        url = f"{BASE_URL}/movie/popular"
        params = {
            "api_key": API_KEY,
            "language": "en-US",
            "page": page,
            "vote_count.gte": 10,
            "primary_release_date.lte": __import__('datetime').date.today().isoformat()
        }
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            movies = response.json().get('results', [])
            for movie in movies:
                if not movie.get('overview'):
                    continue # Skip if no description
                
                # Convert genre IDs to names
                genre_names = [genre_mapping.get(gid, "Unknown") for gid in movie.get('genre_ids', [])]
                
                cleaned_movie = {
                    "id": movie.get('id'),
                    "title": movie.get('title'),
                    "overview": movie.get('overview'),
                    "genres": ", ".join(genre_names),
                    "release_date": movie.get('release_date', 'Unknown'),
                    "vote_average": movie.get('vote_average', 0.0),
                    "language": movie.get('original_language', 'en')
                }
                all_movies.append(cleaned_movie)
        else:
            print(f"Error fetching page {page}: {response.status_code}")
            
    # Save to JSON
    with open("movies_data.json", "w", encoding="utf-8") as f:
        json.dump(all_movies, f, indent=4, ensure_ascii=False)
        
    print(f"Successfully fetched and saved {len(all_movies)} movies.")
    return all_movies

# Language name → ISO 639-1 code mapping
LANGUAGE_MAP = {
    "hindi": "hi", "french": "fr", "korean": "ko", "japanese": "ja",
    "spanish": "es", "tamil": "ta", "telugu": "te", "malayalam": "ml",
    "english": "en", "chinese": "zh", "german": "de", "italian": "it",
    "portuguese": "pt", "arabic": "ar", "russian": "ru", "turkish": "tr",
}

# Genre name → TMDB genre ID mapping
GENRE_MAP = {
    "action": 28, "adventure": 12, "animation": 16, "comedy": 35,
    "crime": 80, "documentary": 99, "drama": 18, "family": 10751,
    "fantasy": 14, "history": 36, "horror": 27, "romance": 10749,
    "sci-fi": 878, "science fiction": 878, "thriller": 53, "war": 10752,
}

def get_person_id(person_name):
    """Search for a person (actor/director) and return their TMDB ID."""
    url = f"{BASE_URL}/search/person"
    params = {"api_key": API_KEY, "query": person_name, "language": "en-US"}
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            results = response.json().get('results', [])
            if results:
                return results[0]['id']
    except Exception:
        pass
    return None

def discover_movies(language_code=None, genre_id=None, sort_by="popularity.desc", pages=3, year_gte=None, person_id=None):
    """Use TMDB /discover/movie for language/genre/mood based queries."""
    print(f"Discovering movies (language={language_code}, genre={genre_id}, sort={sort_by}, year>={year_gte}, person={person_id})...")
    genre_mapping = get_genre_mapping()
    url = f"{BASE_URL}/discover/movie"
    new_movies = []

    for page in range(1, pages + 1):
        params = {
            "api_key": API_KEY,
            "language": "en-US",
            "sort_by": sort_by,
            "include_adult": "false",
            "vote_count.gte": 10,          # Only movies that have been released
            "primary_release_date.lte": __import__('datetime').date.today().isoformat(), # No future movies
            "page": page
        }
        if language_code:
            params["with_original_language"] = language_code
        if genre_id:
            params["with_genres"] = genre_id
        if year_gte:
            params["primary_release_date.gte"] = f"{year_gte}-01-01"
        if person_id:
            params["with_people"] = person_id

        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Error on discover page {page}: {response.status_code}")
            break

        data = response.json()
        movies = data.get('results', [])
        total_pages = data.get('total_pages', 1)

        for movie in movies:
            if not movie.get('overview'):
                continue
            genre_names = [genre_mapping.get(gid, "Unknown") for gid in movie.get('genre_ids', [])]
            new_movies.append({
                "id": movie.get('id'),
                "title": movie.get('title'),
                "overview": movie.get('overview'),
                "genres": ", ".join(genre_names),
                "release_date": movie.get('release_date', 'Unknown'),
                "vote_average": movie.get('vote_average', 0.0),
                "language": movie.get('original_language', 'en')
            })

        print(f"  Discover page {page}/{min(pages, total_pages)} — {len(new_movies)} movies so far...")
        if page >= total_pages:
            break

    print(f"Discovered {len(new_movies)} movies total.")
    return new_movies

def search_movies(query):
    print(f"Searching TMDB API for '{query}'...")
    genre_mapping = get_genre_mapping()
    url = f"{BASE_URL}/search/movie"
    new_movies = []
    page = 1

    max_pages = 2  # Don't fetch more than 2 pages for a title search
    while page <= max_pages:
        params = {
            "api_key": API_KEY,
            "language": "en-US",
            "query": query,
            "page": page
        }

        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Error searching TMDB on page {page}: {response.status_code}")
            break

        data = response.json()
        movies = data.get('results', [])
        total_pages = data.get('total_pages', 1)

        if not movies:
            break

        for movie in movies:
            if not movie.get('overview'):
                continue

            genre_names = [genre_mapping.get(gid, "Unknown") for gid in movie.get('genre_ids', [])]
            cleaned_movie = {
                "id": movie.get('id'),
                "title": movie.get('title'),
                "overview": movie.get('overview'),
                "genres": ", ".join(genre_names),
                "release_date": movie.get('release_date', 'Unknown'),
                "vote_average": movie.get('vote_average', 0.0),
                "language": movie.get('original_language', 'en')
            }
            new_movies.append(cleaned_movie)

        print(f"  Fetched page {page}/{total_pages} — {len(new_movies)} movies so far...")

        if page >= total_pages:
            break
        page += 1

    print(f"Found {len(new_movies)} total movies via TMDB search.")
    return new_movies

if __name__ == "__main__":
    fetch_movies(pages=5) # 5 pages = ~100 movies
