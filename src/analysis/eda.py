from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from ..config import load_data


INPUT_FILE = Path("data/processed/movies_clean.csv")
FIGURES_DIR = Path("reports/figures")
REPORT_FILE = Path("reports/eda_interpretations.txt")


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


def plot_boxplots(df):
    columns = [
        "runtime",
        "budget",
        "revenue",
        "popularity",
        "vote_average",
        "vote_count"
    ]

    data = df[columns].copy()

    for column in ["budget", "revenue", "popularity", "vote_count"]:
        data[column] = data[column].apply(
            lambda x: x if x > 0 else None
        )

    plt.figure(figsize=(12, 8))

    sns.boxplot(data=data)

    plt.title("Boxplots des variables numériques")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "numeric_boxplots.png")
    plt.close()


def plot_correlation(df):
    columns = [
        "runtime",
        "budget",
        "revenue",
        "popularity",
        "vote_average",
        "vote_count"
    ]

    correlation = df[columns].corr()

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0
    )

    plt.title("Matrice de corrélation")
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "correlation_heatmap.png")
    plt.close()

    return correlation


def generate_interpretations(
    df,
    genre_counts,
    yearly_counts,
    runtime,
    budget_revenue_corr,
    votes_popularity_corr,
    correlation
):
    most_common_genre = genre_counts.index[0]
    most_common_genre_count = genre_counts.iloc[0]

    most_common_year = yearly_counts.idxmax()
    most_common_year_count = yearly_counts.max()

    median_runtime = runtime.median()

    rating_median = df["vote_average"].median()

    popularity_median = df["popularity"].median()

    strongest_pair = None
    strongest_value = 0

    for column in correlation.columns:
        for other in correlation.columns:
            if column != other:
                value = abs(correlation.loc[column, other])

                if value > strongest_value:
                    strongest_value = value
                    strongest_pair = (
                        column,
                        other
                    )

    interpretations = [
        f"Notes : la note médiane est de {rating_median:.2f}. "
        f"La distribution permet d'observer la concentration des évaluations.",

        f"Popularité : la popularité médiane est de {popularity_median:.2f}. "
        f"La distribution permet d'identifier une éventuelle forte asymétrie et des valeurs extrêmes.",

        f"Genres : {most_common_genre} est le genre le plus représenté "
        f"avec {most_common_genre_count} films parmi les données analysées.",

        f"Sorties : l'année {most_common_year} contient le plus grand nombre de films "
        f"avec {most_common_year_count} sorties.",

        f"Durée : la durée médiane est de {median_runtime:.0f} minutes. "
        f"Le graphique permet d'identifier les durées typiques et les valeurs extrêmes.",

        f"Budget / revenus : la corrélation entre le budget et les revenus est "
        f"de {budget_revenue_corr:.2f}. Une valeur positive indique qu'ils évoluent "
        f"globalement dans le même sens.",

        f"Votes / popularité : la corrélation entre le nombre de votes et la popularité "
        f"est de {votes_popularity_corr:.2f}. Le nuage de points montre la dispersion "
        f"et les éventuelles valeurs atypiques.",

        "Boxplots : ils permettent d'identifier les médianes, la dispersion "
        "et les valeurs extrêmes des variables numériques.",

        f"Corrélations : la plus forte corrélation absolue observée entre deux variables "
        f"numériques est entre {strongest_pair[0]} et {strongest_pair[1]} "
        f"avec une valeur absolue de {strongest_value:.2f}."
    ]

    return interpretations


def save_interpretations(interpretations):
    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        for interpretation in interpretations:
            file.write(interpretation + "\n\n")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    df = prepare_data(df)

    plot_ratings(df)
    plot_popularity(df)

    genre_counts = plot_genres(df)

    yearly_counts = plot_releases_by_year(df)

    runtime = plot_runtime(df)

    budget_revenue_corr = plot_budget_revenue(df)

    votes_popularity_corr = plot_votes_popularity(df)

    plot_boxplots(df)

    correlation = plot_correlation(df)

    interpretations = generate_interpretations(
        df,
        genre_counts,
        yearly_counts,
        runtime,
        budget_revenue_corr,
        votes_popularity_corr,
        correlation
    )

    save_interpretations(interpretations)

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "figures": 9,
        "report": str(REPORT_FILE)
    }


if __name__ == "__main__":
    result = main()
    
    print(f"{result['rows']} films analyzed")
    print(f"{result['columns']} columns analyzed")
    print(f"{result['figures']} figures created")
    print(f"Interpretations saved to {result['report']}")