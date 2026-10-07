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


def train_kmeans(matrix, k):

    model = KMeans(n_clusters=k, n_init=10, random_state=12)
    labels = model.fit_predict(matrix)

    return model, labels


def get_cluster_terms(model, vectorizer, top_n=10):

    terms = vectorizer.get_feature_names_out()
    cluster_terms = {}

    for cluster_id, center in enumerate(model.cluster_centers_):
        top_indices = center.argsort()[::-1][:top_n]
        cluster_terms[cluster_id] = [terms[index]for index in top_indices]

    return cluster_terms


def create_cluster_profiles(df, labels, cluster_terms):
    
    result = df.copy()
    result["cluster"] = labels

    rows = []

    for cluster_id in sorted(result["cluster"].unique()):

        cluster_data = result[result["cluster"] == cluster_id]

        row = {
            "cluster": cluster_id,
            "size": len(cluster_data),
            "top_terms": ", ".join(cluster_terms[cluster_id])
        }

        if "runtime" in cluster_data.columns:
            row["average_runtime"] = (cluster_data["runtime"].mean())

        if "release_year" in cluster_data.columns:
            row["average_release_year"] = (cluster_data["release_year"].mean())

        rows.append(row)

    return pd.DataFrame(rows)


def save_results(silhouette_results, cluster_profiles, df,labels):

    silhouette_results.to_csv(f"{RESULTS_DIR}/silhouette_scores.csv", index=False)

    cluster_profiles.to_csv(f"{RESULTS_DIR}/cluster_profiles.csv", index=False)

    cluster_data = df.copy()
    cluster_data["cluster"] = labels

    cluster_data.to_csv(f"{RESULTS_DIR}/movies_clusters.csv", index=False)


def plot_silhouette_scores(results):

    plt.figure(figsize=(8, 5))

    plt.plot(results["k"], results["silhouette_score"], marker="o")

    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score by K")

    plt.savefig(f"{RESULTS_DIR}/silhouette_scores.png", bbox_inches="tight")

    plt.close()

