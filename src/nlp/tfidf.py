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

