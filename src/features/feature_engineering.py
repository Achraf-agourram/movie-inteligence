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
