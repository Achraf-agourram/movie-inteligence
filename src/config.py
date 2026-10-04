from pathlib import Path
import numpy as np
import pandas as pd


def load_data(client, collection):

    movies = list(collection.find({}, {"_id": 0}))
    client.close()

    return pd.DataFrame(movies)

def load_overviews(client, collection):
    movies = list(collection.find({}, {"_id": 0, "movie_id": 1, "overview": 1}))
    client.close()

    return pd.DataFrame(movies)


def save_data(df, filename):

    filename.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(filename, index=False)