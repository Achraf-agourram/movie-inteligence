import json
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient


load_dotenv()

CLEAN_JSON = Path("data/processed/movies_clean.json")
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DATABASE = os.getenv("MONGO_DATABASE")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION")


def load_movies():
    with open(CLEAN_JSON, "r", encoding="utf-8") as file:
        return json.load(file)


def connect_mongodb():
    client = MongoClient(MONGO_URI)
    database = client[MONGO_DATABASE]
    collection = database[MONGO_COLLECTION]

    return client, collection


def convert_dates(movies):
    for movie in movies:
        if movie.get("release_date"):
            movie["release_date"] = datetime.fromisoformat(movie["release_date"])

    return movies


def load_to_mongodb(movies, client, collection):

    movies = convert_dates(movies)
    collection.create_index("movie_id", unique=True)

    for movie in movies:
        collection.replace_one(
            {"movie_id": movie["movie_id"]},
            movie,
            upsert=True
        )

    client.close()

    return len(movies)