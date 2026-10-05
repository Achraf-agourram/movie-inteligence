import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
accuracy_score,
classification_report,
confusion_matrix,
ConfusionMatrixDisplay,
f1_score,
precision_score,
recall_score,
roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

DATA_FILE = "data/features/movies_features.csv"
MODEL_DIR = "models/classification"
RESULTS_DIR = "data/results/classification"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data():
    return pd.read_csv(DATA_FILE)

def create_target(df):
    
    threshold = df["vote_count"].quantile(0.75)
    df["high_engagement"] = (df["vote_count"] >= threshold).astype(int)

    return df, threshold

def prepare_features(df):

    numeric_features = [
    "release_year",
    "release_month",
    "release_decade",
    "genre_count",
    "keyword_count",
    "runtime",
    "overview_length",
    "has_overview",
    "has_budget",
    "budget_log"
    ]

    categorical_features = ["original_language", "runtime_category"]

    X = df[numeric_features + categorical_features]

    y = df["high_engagement"]

    return X, y, numeric_features, categorical_features

def create_preprocessor(numeric_features, categorical_features):

    numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore"
        ))
    ])

    preprocessor = ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ])

    return preprocessor

