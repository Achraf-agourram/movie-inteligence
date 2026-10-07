import pandas as pd
from pathlib import Path
import numpy as np
from ..config import load_data, save_data

OUTPUT_FILE = Path("data/features/movies_features.csv")

def create_date_features(df):
    
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date"].dt.year
    df["release_month"] = df["release_date"].dt.month
    df["release_decade"] = (df["release_year"] // 10) * 10

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


def create_features(df):

    df = create_date_features(df)
    df = create_runtime_features(df)
    df = create_text_features(df)
    df = create_budget_features(df)

    return df


def main():
    
    df = load_data()
    df = create_features(df)
    save_data(df, OUTPUT_FILE)

    return df


if __name__ == "__main__":
    df = main()

    print(f"{len(df)} films processed")
    print(f"Features: {len(df.columns)}")
    print(f"Saved to {OUTPUT_FILE}")