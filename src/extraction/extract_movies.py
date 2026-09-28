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

