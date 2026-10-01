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

