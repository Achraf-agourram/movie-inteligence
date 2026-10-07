# TMDB Movie Analytics & Recommendation Platform

A complete data and machine learning pipeline built from the **TMDB API** to analyze movies, classify highly engaging movies, discover similar movie clusters, and generate content-based recommendations.

The project combines **Python, Pandas, Scikit-learn, TF-IDF, K-Means, MongoDB, Airflow, Docker, and Streamlit**.

## Objectives

The project addresses three main machine learning objectives:

1. **Classification** — predict whether a movie belongs to a `high_engagement` category.
2. **Clustering** — automatically group movies according to their content similarity.
3. **Recommendation** — recommend movies similar to a selected movie using textual content.

The complete workflow is:

```text
TMDB API
   ↓
Extraction
   ↓
Cleaning
   ↓
Feature Engineering
   ↓
MongoDB
   ↓
TF-IDF / Machine Learning
   ├── Classification
   ├── Validation & Optimization
   ├── K-Means Clustering
   └── Content-Based Recommendation
   ↓
Streamlit
```

## Technologies

- **Python 3.14**
- **Pandas / NumPy** — data processing and feature engineering
- **Scikit-learn** — preprocessing and machine learning
- **TF-IDF** — text representation
- **TruncatedSVD** — dimensionality reduction for visualization
- **K-Means** — clustering
- **MongoDB** — movie data storage
- **Apache Airflow** — pipeline orchestration
- **Streamlit** — interactive application
- **Docker / Docker Compose** — containerization
- **Joblib** — model and vectorizer persistence
- **Matplotlib / Seaborn** — data visualization

## Project Structure

```text
.
├─ app/
│  └─ app.py
│
├─ dags/
│  └─ tmdb_pipeline_dag.py
│
├─ data/
│  ├─ exports/
│  ├─ features/
│  │  └─ movies_features.csv
│  ├─ processed/
│  │  ├─ tfidf/
│  │  │  ├─ tfidf_matrix.npz
│  │  │  ├─ tfidf_terms.csv
│  │  │  └─ tfidf_vectorizer.pkl
│  │  └─ movies_clean.json
│  ├─ raw/
│  │  └─ movies.json
│  └─ results/
│     ├─ classification/
│     │  ├─ validation/
│     │  │  ├─ cross_validation_results.csv
│     │  │  ├─ svm_grid_search_results.csv
│     │  │  └─ svm_optimization_comparison.csv
│     │  ├─ confusion_matrix_linear svm.png
│     │  ├─ confusion_matrix_logistic regression.png
│     │  ├─ confusion_matrix_random forest.png
│     │  └─ model_comparison.csv
│     └─ clustering/
│        ├─ cluster_profiles.csv
│        ├─ clusters_2d.png
│        ├─ kmeans_model.pkl
│        ├─ movies_clusters.csv
│        ├─ silhouette_scores.csv
│        └─ silhouette_scores.png
│
├─ database/
│  └─ mongodb_schema.md
│
├─ docker/
│  ├─ airflow/
│  │  ├─ Dockerfile
│  │  └─ requirements.txt
│  └─ streamlit/
│     └─ Dockerfile
│
├─ docs/
│  └─ Component Diagram.jpg
│
├─ models/
│  └─ classification/
│     ├─ linear_svm_optimized.pkl
│     ├─ linear_svm.pkl
│     ├─ logistic_regression.pkl
│     └─ random_forest.pkl
│
├─ reports/
│  └─ figures/
│     ├─ budget_revenue.png
│     ├─ correlation_heatmap.png
│     ├─ movies_by_genre.png
│     ├─ numeric_boxplots.png
│     ├─ popularity_distribution.png
│     ├─ ratings_distribution.png
│     ├─ releases_by_year.png
│     ├─ runtime_distribution.png
│     └─ votes_popularity.png
│
├─ src/
│  ├─ analysis/
│  │  └─ eda.py
│  ├─ classification/
│  │  └─ classification.py
│  ├─ clustering/
│  │  └─ clustering.py
│  ├─ database/
│  │  └─ mongodb.py
│  ├─ extraction/
│  │  └─ extract_movies.py
│  ├─ features/
│  │  └─ feature_engineering.py
│  ├─ models/
│  │  ├─ classification.py
│  │  ├─ cross_validation.py
│  │  ├─ evaluation.py
│  │  ├─ hyperparameter_tuning.py
│  │  └─ train_models.py
│  ├─ nlp/
│  │  └─ tfidf.py
│  ├─ preprocessing/
│  │  └─ clean_data.py
│  ├─ recommendation/
│  │  ├─ content_based.py
│  │  └─ similarity.py
│  ├─ validation/
│  │  └─ validation.py
│  └─ config.py
│
├─ docker-compose.yml
├─ requirements.txt
└─ README.md
```

Generated files such as `.git`, `.env`, `__pycache__`, and `.gitkeep` placeholders are excluded from the repository where appropriate.

## Data Source

Movie data is collected from the **TMDB API**.

The extraction keeps the raw API response so that the pipeline can be rerun without repeatedly requesting the same data from TMDB.

The main movie attributes include:

- `movie_id`
- `title`
- `overview`
- `release_date`
- `runtime`
- `original_language`
- `genres`
- `keywords`
- `budget`
- `revenue`
- `popularity`
- `vote_average`
- `vote_count`

## Environment Variables

Create a `.env` file at the project root.

Example:

```env
TMDB_API_KEY=your_tmdb_api_key

MONGO_URI=mongodb://localhost:27017/
MONGO_DATABASE=movies_db
MONGO_COLLECTION=movies
```

Do not commit the `.env` file.

## Installation

### 1. Clone the project

```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Pipeline Steps

### Step 1 — Extraction

`src/extraction/extract_movies.py` retrieves movie information from TMDB, handles API requests, pagination, and stores the raw result in:

```text
data/raw/movies.json
```

Run:

```bash
python src/extraction/extract_movies.py
```

### Step 2 — Cleaning

`src/preprocessing/clean_data.py` prepares the extracted data by handling data types, missing values, duplicates, inconsistent values, and dates.

Clean data is stored in:

```text
data/processed/movies_clean.json
```

Run:

```bash
python src/preprocessing/clean_data.py
```

### Step 3 — Exploratory Data Analysis

`src/analysis/eda.py` generates visualizations for:

- ratings
- popularity
- genres
- releases by year
- runtime
- budget and revenue
- votes and popularity
- numerical distributions
- correlations

The figures are stored in:

```text
reports/figures/
```

Run:

```bash
python src/analysis/eda.py
```

### Step 4 — Feature Engineering

`src/features/feature_engineering.py` creates structured features such as:

- release year
- release month
- release decade
- runtime category
- overview length
- overview availability
- budget availability
- log-transformed budget

The resulting dataset is stored in:

```text
data/features/movies_features.csv
```

Run:

```bash
python src/features/feature_engineering.py
```

### Step 5 — MongoDB Storage

MongoDB is used to store the cleaned movie data.

The MongoDB connection logic is implemented in:

```text
src/database/mongodb.py
```

The collection is configured through the `.env` variables.

The MongoDB schema is documented in:

```text
database/mongodb_schema.md
```

### Step 6 — TF-IDF

`src/nlp/tfidf.py` prepares movie overviews and transforms them into TF-IDF vectors.

The text preprocessing includes:

- lowercase conversion
- removal of unwanted characters
- whitespace normalization
- removal of empty overviews
- TF-IDF vectorization
- experimentation with `max_features` and `ngram_range`

Generated artifacts:

```text
data/processed/tfidf/
├─ tfidf_matrix.npz
├─ tfidf_terms.csv
└─ tfidf_vectorizer.pkl
```

Run:

```bash
python src/nlp/tfidf.py
```

### Step 7 — Classification

The target variable is `high_engagement`.

It is created from `vote_count` using the **75th percentile** as the threshold. Movies at or above the threshold are classified as high engagement.

`vote_count` is used to create the target and is therefore **not used as a model feature**.

The classification compares three models:

- Logistic Regression
- Random Forest
- Linear SVM

The models use a preprocessing `Pipeline` and `ColumnTransformer`.

The classification features include structured movie information and textual genre/keyword information rather than only `genre_count` and `keyword_count`.

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

Run:

```bash
python src/classification/classification.py
```

Results are stored in:

```text
data/results/classification/
```

Trained models are stored in:

```text
models/classification/
```

### Step 8 — Validation and Optimization

`src/validation/validation.py` uses:

- `StratifiedKFold`
- `cross_validate`
- `GridSearchCV`

Cross-validation is performed using 5 stratified folds.

The Linear SVM is optimized with a hyperparameter grid including:

- `C`
- `class_weight`
- `loss`

The best configuration is selected using the cross-validated **F1-score**.

Results are stored in:

```text
data/results/classification/validation/
├─ cross_validation_results.csv
├─ svm_grid_search_results.csv
└─ svm_optimization_comparison.csv
```

The optimized model is stored as:

```text
models/classification/linear_svm_optimized.pkl
```

Run:

```bash
python src/validation/validation.py
```

### Step 9 — Clustering

`src/clustering/clustering.py` applies **K-Means** to the TF-IDF matrix.

Several values of `K` are tested and evaluated using the **Silhouette Score**.

The best K is selected according to the highest Silhouette Score.

Cluster interpretation is based on representative TF-IDF terms and cluster characteristics.

For visualization, `TruncatedSVD` reduces the TF-IDF matrix to two dimensions.

Generated files:

```text
data/results/clustering/
├─ cluster_profiles.csv
├─ clusters_2d.png
├─ kmeans_model.pkl
├─ movies_clusters.csv
├─ silhouette_scores.csv
└─ silhouette_scores.png
```

Run:

```bash
python src/clustering/clustering.py
```

### Step 10 — Streamlit Integration

`app/app.py` provides an interactive web application with four main sections:

1. **Dashboard** — global movie statistics and visualizations.
2. **Classification** — predicts whether a selected movie has high engagement.
3. **Clusters** — displays K-Means clusters, profiles, and a 2D TF-IDF visualization.
4. **Recommendations** — generates content-based movie recommendations using cosine similarity on TF-IDF vectors.

Run locally:

```bash
streamlit run app/app.py
```

The application will be available through the local Streamlit URL displayed in the terminal.

## Airflow Automation

The Airflow DAG is:

```text
dags/tmdb_pipeline_dag.py
```

It orchestrates the pipeline using `BashOperator` tasks.

The main dependency chain is:

```text
Extraction
   ↓
Cleaning
   ↓
Features
   ↓
MongoDB
   ↓
ML
```

The ML task can execute the downstream machine learning scripts in sequence:

```text
TF-IDF
   ↓
Classification
   ↓
Validation / Optimization
   ↓
Clustering
```

The DAG is designed so that a downstream task starts only after the required upstream task succeeds.

## Docker

The project contains Docker configuration for Airflow and Streamlit:

```text
docker/
├─ airflow/
│  ├─ Dockerfile
│  └─ requirements.txt
└─ streamlit/
   └─ Dockerfile
```

The main orchestration file is:

```text
docker-compose.yml
```

Build the services with:

```bash
docker compose build
```

Start the services with:

```bash
docker compose up
```

Start them in detached mode:

```bash
docker compose up -d
```

Stop the services with:

```bash
docker compose down
```

The exact service URLs depend on the ports configured in `docker-compose.yml`.

## Machine Learning Outputs

### Classification

```text
data/results/classification/
├─ model_comparison.csv
├─ confusion_matrix_linear svm.png
├─ confusion_matrix_logistic regression.png
├─ confusion_matrix_random forest.png
└─ validation/
   ├─ cross_validation_results.csv
   ├─ svm_grid_search_results.csv
   └─ svm_optimization_comparison.csv
```

### Clustering

```text
data/results/clustering/
├─ cluster_profiles.csv
├─ clusters_2d.png
├─ kmeans_model.pkl
├─ movies_clusters.csv
├─ silhouette_scores.csv
└─ silhouette_scores.png
```

## Recommendation System

The recommendation system is based on **content similarity**.

The general process is:

```text
Movie overview
      ↓
Text cleaning
      ↓
TF-IDF
      ↓
Movie vector
      ↓
Cosine similarity
      ↓
Most similar movies
```

Recommendation logic is implemented in:

```text
src/recommendation/
├─ content_based.py
└─ similarity.py
```

## Architecture Diagram

The project architecture is documented visually in:

```text
docs/Component Diagram.jpg
```

## Git and Data Management

Sensitive configuration and generated Python cache files are excluded from version control.

Typically excluded files include:

```text
.env
__pycache__/
.git/
```

Large or generated datasets can also be excluded depending on the final project submission requirements.

## Running the Complete Pipeline Manually

A typical manual execution order is:

```bash
python src/extraction/extract_movies.py
python src/preprocessing/clean_data.py
python src/features/feature_engineering.py
# Run the MongoDB loading step implemented in src/database/mongodb.py
python src/nlp/tfidf.py
python src/classification/classification.py
python src/validation/validation.py
python src/clustering/clustering.py
streamlit run app/app.py
```

Use the Airflow DAG instead of running these commands manually when the automated workflow is configured.

## Project Pipeline Summary

```text
                     TMDB API
                         │
                         ▼
                  Data Extraction
                         │
                         ▼
                     Cleaning
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
      Feature Engineering        MongoDB
             │                       │
             └───────────┬───────────┘
                         ▼
                       TF-IDF
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
       Classification  Clustering  Recommendation
              │          │          │
              ▼          ▼          ▼
           Evaluation / Optimization
                         │
                         ▼
                     Streamlit
```

## Author

Movie analytics and recommendation project developed as part of an AI / data engineering and machine learning project.
