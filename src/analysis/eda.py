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


def plot_releases_by_year(df):
    yearly_counts = (
        df["release_year"]
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
    )

    plt.figure(figsize=(12, 6))

    sns.lineplot(
        x=yearly_counts.index,
        y=yearly_counts.values
    )

    plt.title("Nombre de sorties par année")
    plt.xlabel("Année")
    plt.ylabel("Nombre de films")
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "releases_by_year.png")
    plt.close()

    return yearly_counts


def plot_runtime(df):
    runtime = df["runtime"].dropna()

    plt.figure(figsize=(10, 6))

    sns.histplot(
        runtime,
        kde=True
    )

    plt.title("Distribution de la durée des films")
    plt.xlabel("Durée (minutes)")
    plt.ylabel("Nombre de films")
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "runtime_distribution.png")
    plt.close()

    return runtime


def plot_budget_revenue(df):
    budget_revenue = df[
        (df["budget"] > 0) &
        (df["revenue"] > 0)
    ]

    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=budget_revenue,
        x="budget",
        y="revenue"
    )

    plt.xscale("log")
    plt.yscale("log")

    plt.title("Budget et revenus")
    plt.xlabel("Budget")
    plt.ylabel("Revenus")
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "budget_revenue.png")
    plt.close()

    return budget_revenue["budget"].corr(budget_revenue["revenue"])


def plot_votes_popularity(df):
    data = df[
        (df["vote_count"] > 0) &
        (df["popularity"] > 0)
    ]

    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=data,
        x="vote_count",
        y="popularity"
    )

    plt.xscale("log")

    plt.title("Votes et popularité")
    plt.xlabel("Nombre de votes")
    plt.ylabel("Popularité")
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "votes_popularity.png")
    plt.close()

    return data["vote_count"].corr(data["popularity"])

