from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


INPUT_FILE = Path("data/processed/movies_clean.csv")
FIGURES_DIR = Path("reports/figures")
REPORT_FILE = Path("reports/eda_interpretations.txt")


def load_data():
    return pd.read_csv(INPUT_FILE)


def prepare_data(df):
    df["release_date"] = pd.to_datetime(
        df["release_date"],
        errors="coerce"
    )

    df["release_year"] = df["release_date"].dt.year

    df["genres"] = df["genres"].fillna("")

    return df


def plot_ratings(df):
    plt.figure(figsize=(10, 6))

    sns.histplot(
        df["vote_average"].dropna(),
        kde=True
    )

    plt.title("Distribution des notes")
    plt.xlabel("Note moyenne")
    plt.ylabel("Nombre de films")
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "ratings_distribution.png")
    plt.close()


def plot_popularity(df):
    plt.figure(figsize=(10, 6))

    sns.histplot(
        df["popularity"].dropna(),
        kde=True
    )

    plt.title("Distribution de la popularité")
    plt.xlabel("Popularité")
    plt.ylabel("Nombre de films")
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "popularity_distribution.png")
    plt.close()


def plot_genres(df):
    genres = (
        df["genres"]
        .str.split(", ")
        .explode()
    )

    genres = genres[genres != ""]

    genre_counts = genres.value_counts()

    plt.figure(figsize=(10, 7))

    sns.barplot(
        x=genre_counts.values,
        y=genre_counts.index
    )

    plt.title("Films par genre")
    plt.xlabel("Nombre de films")
    plt.ylabel("Genre")
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "movies_by_genre.png")
    plt.close()

    return genre_counts
