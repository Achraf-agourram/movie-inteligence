import re
from pathlib import Path
import joblib
import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from src.config import load_overviews
from src.database.mongodb import connect_mongodb


OUTPUT_DIR = Path("data/processed/tfidf")
MATRIX_FILE = OUTPUT_DIR / "tfidf_matrix.npz"
VECTORIZER_FILE = OUTPUT_DIR / "tfidf_vectorizer.pkl"
TERMS_FILE = OUTPUT_DIR / "tfidf_terms.csv"


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def prepare_text(df):
    df["overview"] = df["overview"].fillna("")
    df["overview_clean"] = df["overview"].apply(clean_text)

    df = df[df["overview_clean"] != ""]

    return df


def create_tfidf(df, max_features, ngram_range):

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, stop_words="english")
    matrix = vectorizer.fit_transform(df["overview_clean"])

    return vectorizer, matrix


def get_representative_terms(vectorizer, matrix):
    terms = vectorizer.get_feature_names_out()
    scores = matrix.mean(axis=0).A1

    result = pd.DataFrame({
        "term": terms,
        "score": scores
    })

    return result.sort_values("score", ascending=False)


def experiment_tfidf(df):
    configurations = [
        (500, (1, 1)),
        (1000, (1, 1)),
        (1000, (1, 2)),
        (2000, (1, 2))
    ]

    results = []

    for max_features, ngram_range in configurations:

        vectorizer, matrix = create_tfidf(df, max_features, ngram_range)
        representative_terms = get_representative_terms(vectorizer, matrix)

        results.append({
            "max_features": max_features,
            "ngram_range": ngram_range,
            "rows": matrix.shape[0],
            "columns": matrix.shape[1],
            "top_terms": representative_terms.head(10)["term"].tolist()
        })

    return results

