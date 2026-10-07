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


def get_prediction_score(model, movie):

    if hasattr(model,"predict_proba"):
        return model.predict_proba(movie)[0][1]

    if hasattr(model,"decision_function"):
        return model.decision_function(movie)[0]

    return None


def show_dashboard(df):

    st.header("Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Movies", len(df))

    col2.metric("Average rating", round(df["vote_average"].mean(), 2))

    col3.metric("Average popularity", round(df["popularity"].mean(),2))

    col4.metric("Average runtime",round(df["runtime"].mean(), 1))

    st.subheader("Movies released by year")

    releases = (
        df["release_year"]
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index())

    st.line_chart(
        releases)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top genres")

        genre_counts = create_genre_counts(df)

        st.bar_chart(genre_counts)

    with col2:
        st.subheader("Rating distribution")

        rating_counts = (
            df["vote_average"]
            .round(1)
            .value_counts()
            .sort_index()
        )

        st.bar_chart(rating_counts)

    st.subheader("Movies")

    columns = [
        "title",
        "release_year",
        "runtime",
        "vote_average",
        "vote_count",
        "popularity"
    ]

    st.dataframe(df[columns], use_container_width=True)


def show_classification(df):
    st.header("High Engagement Classification")


    model = load_classification_model()

    titles = (
        df["title"]
        .dropna()
        .sort_values()
        .tolist())

    selected_title = st.selectbox(
        "Select a movie",
        titles)

    movie = df[
        df["title"] == selected_title
    ].iloc[0]

    movie_df = movie.to_frame().T

    prediction = model.predict(
        movie_df
    )[0]

    score = get_prediction_score(
        model,
        movie_df)

    if prediction == 1:
        st.success(
            "Prediction: High engagement"
        )
    else:
        st.info(
            "Prediction: Normal engagement"
        )

    if score is not None:
        st.write(
            f"Model score: {score:.4f}"
        )

    st.subheader(
        "Movie information")

    st.write(
        f"**Title:** {movie['title']}")

    st.write(
        f"**Genres:** {movie['genres']}")

    st.write(
        f"**Runtime:** {movie['runtime']} minutes")

    st.write(
        f"**Rating:** {movie['vote_average']}")

    st.write(
        f"**Votes:** {movie['vote_count']}")


def show_clusters():
    st.header("Movie Clusters")


    cluster_data = load_cluster_data()
    cluster_profiles = load_cluster_profiles()

    st.subheader(
        "Cluster profiles")

    st.dataframe(
        cluster_profiles,
        use_container_width=True)

    st.subheader(
        "Cluster sizes")

    cluster_sizes = (
        cluster_data["cluster"]
        .value_counts()
        .sort_index())

    st.bar_chart(
        cluster_sizes)

    st.subheader(
        "2D cluster visualization")

    matrix, _ = load_tfidf()

    coordinates = create_svd_coordinates(
        matrix)

    plot_data = pd.DataFrame({
        "Component 1": coordinates[:, 0],
        "Component 2": coordinates[:, 1],
        "Cluster": cluster_data["cluster"].astype(str)
    })

    st.scatter_chart(
        plot_data,
        x="Component 1",
        y="Component 2",
        color="Cluster")

    selected_cluster = st.selectbox(
        "Select a cluster",
        sorted(
            cluster_data["cluster"].unique()
        ))

    movies = cluster_data[
        cluster_data["cluster"] == selected_cluster
    ]

    st.subheader(
        f"Movies in cluster {selected_cluster}")

    columns = [
        column
        for column in [
            "title",
            "movie_id",
            "overview"
        ]
        if column in movies.columns
    ]

    st.dataframe(
        movies[columns],
        use_container_width=True)


def main():

    st.title("🎬 Movie Analytics Platform")

    st.sidebar.title(
        "Navigation")

    page = st.sidebar.radio(
        "Go to",
        [
            "Dashboard",
            "Classification",
            "Clusters"
        ])

    if not os.path.exists(DATA_FILE):
        st.error(f"Missing file: {DATA_FILE}")
        return

    df = load_movies()

    if page == "Dashboard":
        show_dashboard(df)

    elif page == "Classification":
        show_classification(df)

    elif page == "Clusters":
        show_clusters()


if __name__ == "__main__":
    main()