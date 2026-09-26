# The official MLflow image ships without a PostgreSQL driver, and the spec (section 2)
# requires a Postgres backend store.
FROM ghcr.io/mlflow/mlflow:v3.16.1

RUN pip install --no-cache-dir psycopg2-binary==2.9.10

EXPOSE 5000
