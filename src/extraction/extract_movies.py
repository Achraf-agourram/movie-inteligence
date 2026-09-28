import json
import os
import time
from pathlib import Path
import requests
from dotenv import load_dotenv


load_dotenv()

BASE_URL = "https://api.themoviedb.org/3"
RAW_FILE = Path("data/raw/movies.json")

TOKEN = os.getenv("TMDB_ACCESS_TOKEN")
TARGET_MOVIES = int(os.getenv("TMDB_TARGET_MOVIES", "1000"))
REQUEST_DELAY = float(os.getenv("TMDB_REQUEST_DELAY", "0.1"))
MAX_RETRIES = int(os.getenv("TMDB_MAX_RETRIES", "5"))


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

def load_json(path):
    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)

def request_data(endpoint, params=None):
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "accept": "application/json"
    }

    for attempt in range(MAX_RETRIES):
        response = requests.get(
            f"{BASE_URL}{endpoint}",
            headers=headers,
            params=params,
            timeout=30
        )

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After", 2)
            time.sleep(float(retry_after))
            continue

        if response.status_code >= 500:
            time.sleep(2 ** attempt)
            continue

        response.raise_for_status()

        if not response.content:
            return None

        data = response.json()

        if not data:
            return None

        time.sleep(REQUEST_DELAY)

        return data

    raise Exception("TMDB request failed")

def extract_movies():
    if not TOKEN:
        raise ValueError("TMDB_ACCESS_TOKEN is missing")

    movies = load_json(RAW_FILE)

    existing_ids = {movie["movie_id"] for movie in movies}
    page = 1

    while len(movies) < TARGET_MOVIES:
        data = request_data(
            "/discover/movie",
            {
                "page": page,
                "sort_by": "popularity.desc",
                "include_adult": "false",
                "include_video": "false"
            }
        )

        if data is None:
            break

        results = data.get("results", [])

        if not results:
            break

        for movie in results:
            movie_id = movie.get("id")

            if movie_id is None or movie_id in existing_ids:
                continue

            details = request_data(
                f"/movie/{movie_id}",
                {
                    "append_to_response": "keywords"
                }
            )

            if details is None:
                continue

            movie_data = {
                "movie_id": details.get("id"),
                "title": details.get("title"),
                "overview": details.get("overview"),
                "release_date": details.get("release_date"),
                "runtime": details.get("runtime"),
                "original_language": details.get("original_language"),
                "genres": details.get("genres", []),
                "keywords": details.get("keywords", {}).get("keywords", []),
                "budget": details.get("budget"),
                "revenue": details.get("revenue"),
                "popularity": details.get("popularity"),
                "vote_average": details.get("vote_average"),
                "vote_count": details.get("vote_count")
            }

            movies.append(movie_data)
            existing_ids.add(movie_id)

            if len(movies) >= TARGET_MOVIES:
                break

        if page >= data.get("total_pages", page):
            break

        page += 1

    save_json(movies, RAW_FILE)

    return movies


if __name__ == "__main__":
    movies = extract_movies()
    print(f"{len(movies)} movies extracted")
