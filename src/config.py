import os
from pathlib import Path
import numpy as np
import pandas as pd
from database.mongodb import connect_mongodb

client, collection = connect_mongodb()

def load_data(client, collection):

    movies = list(collection.find({}, {"_id": 0}))
    client.close()

    return pd.DataFrame(movies)


def save_data(df, filename):

    filename.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(filename, index=False)