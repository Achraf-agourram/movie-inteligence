import datetime

from airflow.sdk import DAG
from airflow.providers.standard.operators.bash import BashOperator

PROJECT_DIR = "/opt/airflow"

with DAG(
    dag_id="movie_pipeline",
    start_date=datetime.datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["tmdb", "movies", "ml"]

) as dag:

    extract = BashOperator(
        task_id="extract",
        bash_command=("python -m src.extraction.extract_movies"),
        cwd=PROJECT_DIR,
    )

    clean = BashOperator(
        task_id="clean",
        bash_command=("python -m src.preprocessing.clean_data"),
        cwd=PROJECT_DIR,
    )

    features = BashOperator(
        task_id="features",
        bash_command=("python -m src.features.feature_engineering"),
        cwd=PROJECT_DIR,
    )

    mongodb = BashOperator(
        task_id="mongodb",
        bash_command=("python -m src.database.mongodb"),
        cwd=PROJECT_DIR,
    )

    ml = BashOperator(
        task_id="ml",
        bash_command=(
            "python -m src.nlp.tfidf && "
            "python -m src.classification.classification && "
            "python -m src.validation.validation && "
            "python -m src.clustering.clustering"
        ),
        cwd=PROJECT_DIR,
    )

    extract >> clean >> mongodb >> features >> ml