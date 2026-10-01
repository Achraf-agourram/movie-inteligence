from pathlib import Path
import numpy as np
import pandas as pd


INPUT_FILE = Path("data/processed/movies_clean.csv")
OUTPUT_FILE = Path("data/processed/movies_features.csv")


def load_data():
    return pd.read_csv(INPUT_FILE)


def create_date_features(df):
    
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date"].dt.year
    df["release_month"] = df["release_date"].dt.month
    df["release_decade"] = (df["release_year"] // 10) * 10

    return df


def create_count_features(df):

    df["genre_count"] = (df["genres"].fillna("").apply(
            lambda x: len(x.split(", ")) if x else 0
        )
    )

    df["keyword_count"] = (df["keywords"].fillna("").apply(
            lambda x: len(x.split(", ")) if x else 0
        )
    )

    return df


def create_runtime_features(df):

    df["runtime_category"] = pd.cut(
        df["runtime"],
        bins=[-np.inf, 90, 120, 150, np.inf],
        labels=[
            "short",
            "medium",
            "long",
            "very_long"
        ]
    )

    return df


def create_text_features(df):

    df["overview"] = df["overview"].fillna("")
    df["overview_length"] = df["overview"].str.len()
    df["has_overview"] = (df["overview"].str.strip() != "").astype(int)

    return df


def create_budget_features(df):

    df["has_budget"] = (df["budget"] > 0).astype(int)
    df["budget_log"] = np.log1p(df["budget"].where(df["budget"] > 0))

    return df

