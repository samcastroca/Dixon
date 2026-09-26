#!/bin/sh
# MLflow keeps its tracking store in its own database on the same PostgreSQL instance.
set -eu

psql --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<SQL
    CREATE DATABASE mlflow OWNER $POSTGRES_USER;
SQL
