import json
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient
import pandas as pd

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DATABASE = os.getenv("MONGO_DATABASE")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION")


def connect_mongodb():
    client = MongoClient(MONGO_URI)
    database = client[MONGO_DATABASE]
    collection = database[MONGO_COLLECTION]

    return client, collection


def convert_dates(movies):

    movies["release_date"] = pd.to_datetime(movies["release_date"], errors="coerce")
    return movies


def load_to_mongodb(movies, client, collection):

    movies = convert_dates(movies)
    collection.create_index("movie_id", unique=True)

    for movie in movies.to_dict("records"):
        collection.replace_one(
            {"movie_id": movie["movie_id"]},
            movie,
            upsert=True
        )

    client.close()

    return len(movies)

if __name__ == "__main__":
    client, collection = connect_mongodb()
    movies = pd.read_json(Path("data/raw/movies.json"))
    load_to_mongodb(movies, client, collection)

    print(f" movies loaded to MongoDB")