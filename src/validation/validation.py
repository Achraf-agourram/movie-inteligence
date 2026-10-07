import os
import joblib
import pandas as pd
from sklearn.model_selection import (
train_test_split,
StratifiedKFold,
cross_validate,
GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from ..classification.classification import create_preprocessor, load_data, prepare_features, create_models, create_target

RESULTS_DIR = "data/results/classification/validation"
MODEL_DIR = "models/classification"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def cross_validate_models(models, X_train, y_train):

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=12)

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    results = []

    for name, model in models.items():

        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )

        results.append({
            "Model": name,
            "Accuracy": scores["test_accuracy"].mean(),
            "Precision": scores["test_precision"].mean(),
            "Recall": scores["test_recall"].mean(),
            "F1-score": scores["test_f1"].mean(),
            "ROC-AUC": scores["test_roc_auc"].mean()
        })

    return pd.DataFrame(results)

def optimize_svm(numeric_features, categorical_features, X_train, y_train):

    preprocessor = create_preprocessor(numeric_features, categorical_features)

    svm = Pipeline([
        (
            "preprocessing",
            preprocessor
        ),
        (
            "model",
            LinearSVC(random_state=12)
        )
    ])

    param_grid = {
        "model__C": [
            0.01,
            0.1,
            1,
            10,
            100
        ],
        "model__class_weight": [
            None,
            "balanced"
        ],
        "model__loss": [
            "hinge",
            "squared_hinge"
        ]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=12)

    grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=cv, scoring="f1", n_jobs=-1, return_train_score=True)

    grid_search.fit(X_train, y_train)

    return grid_search

def compare_svm_before_after(baseline_svm, optimized_svm, X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=12)

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    baseline_scores = cross_validate(
        baseline_svm,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    optimized_scores = cross_validate(
        optimized_svm,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    results = pd.DataFrame([
        {
            "Model": "Linear SVM - Before",
            "Accuracy": baseline_scores["test_accuracy"].mean(),
            "Precision": baseline_scores["test_precision"].mean(),
            "Recall": baseline_scores["test_recall"].mean(),
            "F1-score": baseline_scores["test_f1"].mean(),
            "ROC-AUC": baseline_scores["test_roc_auc"].mean()
        },
        {
            "Model": "Linear SVM - After",
            "Accuracy": optimized_scores["test_accuracy"].mean(),
            "Precision": optimized_scores["test_precision"].mean(),
            "Recall": optimized_scores["test_recall"].mean(),
            "F1-score": optimized_scores["test_f1"].mean(),
            "ROC-AUC": optimized_scores["test_roc_auc"].mean()
        }
    ])

    return results

