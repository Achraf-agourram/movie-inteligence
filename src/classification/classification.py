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
from sklearn.feature_extraction.text import TfidfVectorizer

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

    return df

def prepare_features(df):

    numeric_features = [
        "release_year",
        "release_month",
        "release_decade",
        "runtime",
        "overview_length",
        "has_overview",
        "has_budget",
        "budget_log"
    ]

    categorical_features = ["original_language", "runtime_category"]
    text_features = ["genres", "keywords"]

    X = df[numeric_features + categorical_features + text_features]

    y = df["high_engagement"]

    return X, y, numeric_features, categorical_features

def create_preprocessor(numeric_features, categorical_features):
    
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
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
        ),
        (
            "genres",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2)
            ),
            "genres"
        ),
        (
            "keywords",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2)
            ),
            "keywords"
        )
    ])

    return preprocessor

def create_models(numeric_features, categorical_features):

    preprocessor = create_preprocessor(numeric_features, categorical_features)

    models = {
        "Logistic Regression": Pipeline([
            ("preprocessing", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=12
                )
            )
        ]),

        "Random Forest": Pipeline([
            ("preprocessing", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=12,
                    n_jobs=-1,
                    class_weight="balanced"
                )
            )
        ]),

        "Linear SVM": Pipeline([
            ("preprocessing", preprocessor),
            (
                "model",
                LinearSVC(
                    class_weight="balanced",
                    random_state=12
                )
            )
        ])
    }

    return models

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)

    results = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        "Recall": recall_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        "F1-score": f1_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        "ROC-AUC": roc_auc_score(
            y_test,
            y_score
        )
    }

    return results, y_pred

def save_confusion_matrix(y_test, y_pred, model_name):
    
    cm = confusion_matrix(y_test, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Normal engagement",
            "High engagement"
        ]
    )

    display.plot()

    plt.title(f"Confusion Matrix - {model_name}")

    filename = model_name.lower()

    plt.savefig(f"{RESULTS_DIR}/confusion_matrix_{filename}.png", bbox_inches="tight")

    plt.close()

def train_and_evaluate():
    df = load_data()

    df = create_target(df)

    X, y, numeric_features, categorical_features = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=12,
        stratify=y
    )

    models = create_models(numeric_features, categorical_features)

    results = []

    for name, model in models.items():

        model.fit(X_train, y_train)

        metrics, y_pred = evaluate_model(model, X_test, y_test)

        metrics["Model"] = name
        results.append(metrics)

        save_confusion_matrix(y_test, y_pred, name)

        joblib.dump(model, f"{MODEL_DIR}/{name.lower().replace(' ', '_')}.pkl")

    results_df = pd.DataFrame(results)

    results_df = results_df[
        [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1-score",
            "ROC-AUC"
        ]
    ]

    results_df.to_csv(f"{RESULTS_DIR}/model_comparison.csv", index=False)

    return results_df

if __name__ == "__main__":
    train_and_evaluate()