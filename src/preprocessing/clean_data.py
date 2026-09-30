import json
from pathlib import Path
import numpy as np
import pandas as pd


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
