import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score
from src.config import load_overviews
from src.database.mongodb import connect_mongodb
from src.nlp.tfidf import prepare_text

TFIDF_MATRIX_FILE = "data/processed/tfidf/tfidf_matrix.npz"
VECTORIZER_FILE = "data/processed/tfidf/tfidf_vectorizer.pkl"

RESULTS_DIR = "data/results/clustering"

os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data():

    client, collection = connect_mongodb()
    df = load_overviews(client, collection)

    client.close()

    df = prepare_text(df)

    matrix = load_npz(TFIDF_MATRIX_FILE)

    vectorizer = joblib.load(VECTORIZER_FILE)

    if len(df) != matrix.shape[0]:
        raise ValueError(
            "The number of movies does not match "
            "the number of TF-IDF rows."
        )

    return df, matrix, vectorizer


def test_k_values(matrix, k_values):

    results = []

    for k in k_values:

        model = KMeans(n_clusters=k, n_init=10, random_state=12)
        labels = model.fit_predict(matrix)
        score = silhouette_score(matrix, labels)

        results.append({
            "k": k,
            "silhouette_score": score
        })

    return pd.DataFrame(results)


def select_best_k(results):

    best_row = results.loc[results["silhouette_score"].idxmax()]

    return int(best_row["k"])

