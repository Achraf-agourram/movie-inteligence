import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from scipy.sparse import load_npz
from sklearn.decomposition import TruncatedSVD

DATA_FILE = "data/features/movies_features.csv"
TFIDF_MATRIX_FILE = ("data/processed/tfidf/tfidf_matrix.npz")
TFIDF_VECTORIZER_FILE = ("data/processed/tfidf/tfidf_vectorizer.pkl")
CLUSTER_DATA_FILE = ("data/results/clustering/movies_clusters.csv")
CLUSTER_PROFILES_FILE = ("data/results/clustering/cluster_profiles.csv")
CLASSIFICATION_MODEL_FILE = ("models/classification/linear_svm_optimized.pkl")
CLUSTER_MODEL_FILE = ("data/results/clustering/kmeans_model.pkl")

st.set_page_config(page_title="Movie Analytics", page_icon="🎬", layout="wide")

@st.cache_data
def load_movies():
    return pd.read_csv(DATA_FILE)

@st.cache_data
def load_cluster_data():
    return pd.read_csv(CLUSTER_DATA_FILE)

@st.cache_data
def load_cluster_profiles():
    return pd.read_csv(CLUSTER_PROFILES_FILE)

@st.cache_resource
def load_classification_model():
    return joblib.load(CLASSIFICATION_MODEL_FILE)

@st.cache_resource
def load_cluster_model():
    return joblib.load(CLUSTER_MODEL_FILE)

@st.cache_resource
def load_tfidf():
    matrix = load_npz(TFIDF_MATRIX_FILE)
    vectorizer = joblib.load(TFIDF_VECTORIZER_FILE)


    return matrix, vectorizer


def create_svd_coordinates(matrix):

    svd = TruncatedSVD(n_components=2, random_state=12)
    coordinates = svd.fit_transform(matrix)

    return coordinates


def create_genre_counts(df):

    genres = (df["genres"].fillna("").str.split(", ").explode())
    genres = genres[genres != ""]

    return genres.value_counts().head(10)

