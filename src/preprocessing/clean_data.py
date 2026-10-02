import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.database.mongodb import load_to_mongodb, connect_mongodb


RAW_FILE = Path("data/raw/movies.json")
PROCESSED_DIR = Path("data/processed")
CLEAN_CSV = PROCESSED_DIR / "movies_clean.csv"
CLEAN_JSON = PROCESSED_DIR / "movies_clean.json"


NUMERIC_COLUMNS = [
    "runtime",
    "budget",
    "revenue",
    "popularity",
    "vote_average",
    "vote_count"
]

CATEGORICAL_COLUMNS = [
    "original_language",
    "genres"
]

TEXT_COLUMNS = [
    "title",
    "overview",
    "keywords"
]


def load_raw_data():
    with open(RAW_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def extract_names(values):
    if not isinstance(values, list):
        return []

    return [
        value.get("name")
        for value in values
        if isinstance(value, dict) and value.get("name")
    ]


def clean_data(data):
    df = pd.DataFrame(data)

    columns = [
        "movie_id",
        "title",
        "overview",
        "release_date",
        "runtime",
        "original_language",
        "genres",
        "keywords",
        "budget",
        "revenue",
        "popularity",
        "vote_average",
        "vote_count"
    ]

    df = df[columns]

    df["genres"] = df["genres"].apply(extract_names)
    df["keywords"] = df["keywords"].apply(extract_names)

    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in ["runtime", "budget", "revenue", "popularity", "vote_average", "vote_count"]:
        df.loc[df[column] < 0, column] = np.nan

    df.loc[df["vote_average"] > 10, "vote_average"] = np.nan

    df["title"] = df["title"].fillna("").str.strip()
    df["overview"] = df["overview"].fillna("").str.strip()
    df["original_language"] = (df["original_language"].fillna("").str.strip())

    df = df.dropna(subset=["movie_id", "title"])

    df = df.drop_duplicates(subset=["movie_id"], keep="first")

    return df


def separate_variables(df):
    numeric_df = df[NUMERIC_COLUMNS].copy()
    categorical_df = df[CATEGORICAL_COLUMNS].copy()
    text_df = df[TEXT_COLUMNS].copy()

    return numeric_df, categorical_df, text_df


def save_clean_data(df):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    csv_df = df.copy()
    csv_df["genres"] = csv_df["genres"].apply(lambda values: ", ".join(values))
    csv_df["keywords"] = csv_df["keywords"].apply(lambda values: ", ".join(values))
    
    df.to_json(CLEAN_JSON, orient="records", force_ascii=False, date_format="iso", indent=2)


def main():

    data = load_raw_data()
    clean_df = clean_data(data)
    numeric_df, categorical_df, text_df = separate_variables(clean_df)

    client, collection = connect_mongodb()
    save_clean_data(clean_df)
    load_to_mongodb(clean_df, client, collection)

    return {
        "clean_shape": clean_df.shape,
        "numeric_columns": numeric_df.columns.tolist(),
        "categorical_columns": categorical_df.columns.tolist(),
        "text_columns": text_df.columns.tolist()
    }


if __name__ == "__main__":
    result = main()
    print(result)