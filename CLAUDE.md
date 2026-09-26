# CLAUDE.md — sports results prediction platform

Full specification: `docs/SPEC.pdf`. Read the sections a task touches before writing code.

**Current phase: 0 (foundations) — complete.** Next up is phase 1 (ingestion).

## Non-negotiable rules

1. **One phase at a time.** Never implement anything belonging to a later phase of spec
   section 11, however small. If a task needs it, stop and say so.
2. **No data leakage.** Features may only use data with timestamps *strictly before* kickoff.
   Random train/test splits are forbidden; validation is walk-forward only. Every feature row
   stores `as_of_utc`, and `as_of_utc < kickoff_utc` is an invariant, not a guideline.
3. **Tests never touch the network.** Use recorded files in `tests/fixtures/` and mock the HTTP
   layer.
4. **Secrets only from environment variables** (`.env`, never committed; `.env.example` is).
   Never read a credential from `config/settings.yaml` or hard-code one.
5. **Every tunable lives in `config/settings.yaml`, not in code.** Competitions (EPL, LALIGA),
   their per-source codes, timezones and rules are configuration too. **Never hard-code a league
   name in logic** — iterate over `settings.competitions` and look leagues up by code.
6. **All times are stored in UTC**, converted from each competition's local timezone
   (`Competition.tzinfo()`), daylight saving included.
7. **Type hints everywhere; ruff and mypy clean; every new module has tests.**
8. **Schema changes only through Alembic migrations** (`make revision M="..."`), never by
   editing tables by hand.
9. **Run `make check` before calling a task done, and show the output.**
10. **If the spec is ambiguous, ask instead of guessing.**

## Architecture (spec section 2)

One `docker-compose.yml`, one internal network. The batch services (`ingest`, `pipeline`,
`train`) share the application image and differ only in their command.

| Service     | Responsibility                                                        | Phase |
| ----------- | --------------------------------------------------------------------- | ----- |
| `db`        | PostgreSQL 16: raw payloads, clean tables, features, predictions       | 0     |
| `api`       | FastAPI: health, fixtures, predictions, ratings, backtests             | 0 / 7 |
| `mlflow`    | Experiment tracking and model registry (Postgres backend, volume)      | 0     |
| `ingest`    | Per-source fetchers; writes raw payloads                               | 1     |
| `pipeline`  | Cleaning, pandera validation, entity resolution, feature building      | 2 / 3 |
| `train`     | Fitting, walk-forward backtests, calibration, registry                 | 4 / 6 |
| `dashboard` | Streamlit UI                                                           | 7     |

Data flow: `sources → ingest → raw_payloads → pipeline → teams/matches/odds → features →
train → MLflow + predictions → api / dashboard`.

Design constraints that must survive every phase: a `sport` column on competitions, a common
`Source` interface for every data source, and a common `MatchModel` interface for every model,
so more leagues and more sports need config, not new branches in logic. Team-strength models
(M1–M4) are fitted **per competition** — EPL and LALIGA teams never meet in league play.

## Repo layout (spec section 2)

```
CLAUDE.md  docker-compose.yml  Dockerfile  Makefile  pyproject.toml  uv.lock  .env.example
alembic/                  # migrations (baseline 0001_baseline is empty)
config/settings.yaml      # competitions, database, mlflow, api — every tunable
docker/                   # mlflow image, postgres init scripts
src/predictor/
  config.py  db.py  logging.py  cli.py
  api/                    # main.py (GET /health), routers/, schemas.py
  ingestion/ processing/ features/ models/ evaluation/ simulation/ dashboard/
tests/
  unit/ integration/ e2e/ fixtures/   # recorded source files; no network in tests
data/raw/                 # gitignored local cache
```

## Makefile targets

| Target          | What it does                                               |
| --------------- | ---------------------------------------------------------- |
| `make up`       | Build and start db + api + mlflow, waiting for healthy      |
| `make down`     | Stop the stack (`ARGS=-v` also drops the volumes)           |
| `make logs`     | Follow every service's logs                                 |
| `make check`    | `lint` + `types` + `test` — the gate CI runs                |
| `make lint`     | `ruff check` and `ruff format --check`                      |
| `make types`    | `mypy` (strict)                                             |
| `make test`     | `pytest`                                                    |
| `make fmt`      | Apply ruff fixes and formatting                             |
| `make migrate`  | `alembic upgrade head`                                      |
| `make revision` | `make revision M="add matches"`                             |
| `make build`    | Build the images without starting them                      |

## Conventions

- Configuration is loaded through `predictor.config.get_settings()`; every value can be
  overridden by `PREDICTOR_<SECTION>__<KEY>`, and `DATABASE_URL` overrides the whole connection
  string (that is what CI and host tooling use).
- Logging is structlog JSON via `configure_logging()`; bind one `run_id` per execution with
  `bind_run_id()`.
- Integration tests are marked `@pytest.mark.integration` and skip themselves when no database
  is reachable; `make up` before `make check` runs them for real.
- Tests live next to their kind: `tests/unit`, `tests/integration`, `tests/e2e`.
- The `later` compose profile holds services whose phase has not arrived yet.
