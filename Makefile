.PHONY: help up down logs check test lint types migrate revision fmt build ingest process

UV ?= uv
COMPOSE ?= docker compose

# Host tooling (alembic, pytest) reads .env when it exists; CI provides the same
# variables directly in the environment.
ifneq (,$(wildcard .env))
export UV_ENV_FILE := .env
endif

help:  ## Show the available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-10s %s\n", $$1, $$2}'

up:  ## Start db, api and mlflow and wait until they are healthy
	$(COMPOSE) up -d --build --wait

down:  ## Stop the stack (add ARGS=-v to drop the volumes)
	$(COMPOSE) down $(ARGS)

logs:  ## Follow the logs of every running service
	$(COMPOSE) logs -f

lint:  ## ruff lint + format check
	$(UV) run ruff check .
	$(UV) run ruff format --check .

types:  ## mypy (strict)
	$(UV) run mypy

test:  ## pytest
	$(UV) run pytest

check: lint types test  ## Everything CI runs: ruff + mypy + pytest

fmt:  ## Apply ruff formatting and safe fixes
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

COMPETITIONS ?= EPL,LALIGA
SEASONS ?= 2014-2025

ingest:  ## Fetch season files: make ingest COMPETITIONS=EPL,LALIGA SEASONS=2014-2025
	$(COMPOSE) run --rm --build ingest predictor ingest 		--source football_data_uk --competition $(COMPETITIONS) --seasons $(SEASONS)

process:  ## Clean and validate the staged rows: make process COMPETITIONS=EPL,LALIGA
	$(COMPOSE) run --rm --build pipeline predictor process --competition $(COMPETITIONS)

migrate:  ## Apply the migrations (alembic upgrade head)
	$(UV) run alembic upgrade head

revision:  ## Create a migration: make revision M="add matches"
	$(UV) run alembic revision -m "$(M)"

build:  ## Build the images without starting anything
	$(COMPOSE) build
