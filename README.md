# Sports Results Prediction Platform

1X2 probabilities and scoreline distributions for the English Premier League (EPL) and
LaLiga (LALIGA). Specification: `docs/SPEC.pdf`. Working rules for contributors and for
Claude Code: `CLAUDE.md`.

Everything runs under one `docker-compose.yml`: PostgreSQL 16, FastAPI, MLflow, and one
application image shared by the batch commands. Adding a league is a change to
`config/settings.yaml` and `config/team_aliases.yaml`, never to logic.

## Quick start

```bash
cp .env.example .env          # fill in POSTGRES_PASSWORD
make up                       # db + api + mlflow, waiting for healthy
curl http://127.0.0.1:8000/health
make migrate                  # alembic upgrade head
make check                    # ruff + mypy + pytest (the gate CI runs)
```

## Building the data

```bash
make ingest COMPETITIONS=EPL,LALIGA SEASONS=2014-2025   # raw payloads + staging rows
make process                                            # clean, validated match tables
make features VERSION=v1                                # point-in-time feature store
make backtest MODEL=baseline_market,baseline_frequency  # walk-forward metrics into MLflow
make report                                             # reports/backtest.md and .html
```

`sources → ingest → raw_payloads → pipeline → teams/matches/odds → features`. Each step is
idempotent: running it twice leaves the database exactly as it was.

With the seasons above that gives 8360 matches per run (11 complete seasons of 380 in each
league), their per-team statistics and odds from four bookmakers, and one feature row per
match with 86 pre-match features.

## Status

**Phase 4 (backtesting and baselines) — complete.** Next up is phase 5 (the statistical
models M1-M4). Phases are delivered one at a time, in the order of spec section 11.

| Phase | Delivered                                                              |
| ----- | ---------------------------------------------------------------------- |
| 0     | Repo skeleton, settings, structured logging, Alembic, `GET /health`     |
| 1     | `Source` interface and the football-data.co.uk ingester; raw payloads   |
| 2     | Entity resolution, cleaning, pandera validation, de-margined odds       |
| 3     | Chronological replay, Elo, form, schedule features, versioned store     |
| 4     | Walk-forward engine, RPS/log loss/Brier/ECE, bootstrap CIs, baselines   |
| 5-8   | Statistical models, ML and ensembling, serving, hardening               |

### No data leakage

The feature store is the part the later phases lean on hardest, so its correctness is
structural rather than conventional. Features may only read data timestamped **strictly
before** kickoff, they are built by replaying history in kickoff order, and the single door
to the past (`features/replay.py`) raises `LeakageError` for anything at or after its
cut-off. Matches sharing a kickoff never see each other's results. Validation is
walk-forward only; random train/test splits are forbidden.

The backtester inherits that door: a model is fitted on `HistoryView.played()` and predicts
from fixtures with their results stripped, so a model that tries to read the result of the
match it is predicting raises `LeakageError` instead of scoring suspiciously well.

### What there is to beat

Six test seasons (2019-20 to 2024-25) of both leagues, 4560 matches per model:

| Model                | RPS    | log loss | Brier  | ECE    |
| -------------------- | ------ | -------- | ------ | ------ |
| `baseline_market`    | 0.1934 | 0.9625   | 0.5709 | 0.0134 |
| `baseline_frequency` | 0.2300 | 1.0704   | 0.6477 | 0.0139 |

The market benchmark is 0.0366 RPS better, with a paired bootstrap interval of
[+0.0325, +0.0406] that excludes zero in both leagues separately. It is a benchmark and not
a model: it is marked undeployable, because its input is the closing price.

## Development

Dependencies are managed with `uv` and the lockfile is committed. `make help` lists every
target. Tests never touch the network — the HTTP layer is mocked against recorded files in
`tests/fixtures/`. Integration tests skip themselves unless a database is reachable, so run
`make up` before `make check` to exercise them for real; the backtest ones point MLflow at a
SQLite file of their own, so no tracking server is involved either.
