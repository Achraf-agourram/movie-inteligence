import os
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DATABASE = os.getenv("MONGO_DATABASE", "movies_db")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "movies")


def load_data():

    client = MongoClient(MONGO_URI)
    collection = client[MONGO_DATABASE][MONGO_COLLECTION]
    movies = list(collection.find({}, {"_id": 0}))

    client.close()

    return pd.DataFrame(movies)


def save_data(df, filename):

    filename.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(filename, index=False)