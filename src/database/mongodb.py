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

