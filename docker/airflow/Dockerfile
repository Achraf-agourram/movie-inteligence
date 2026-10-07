FROM apache/airflow:3.1.0-python3.12

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

USER airflow
COPY requirements.txt /requirements-project.txt
COPY docker/airflow/requirements.txt /requirements-airflow.txt

RUN pip install --no-cache-dir \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.1.0/constraints-3.12.txt" \
    -r /requirements-project.txt \
    -r /requirements-airflow.txt