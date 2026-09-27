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
```

`sources → ingest → raw_payloads → pipeline → teams/matches/odds → features`. Each step is
idempotent: running it twice leaves the database exactly as it was.

With the seasons above that gives 8360 matches per run (11 complete seasons of 380 in each
league), their per-team statistics and odds from four bookmakers, and one feature row per
match with 86 pre-match features.

## Status

**Phase 3 (point-in-time features) — complete.** Next up is phase 4 (backtesting and
baselines). Phases are delivered one at a time, in the order of spec section 11.

| Phase | Delivered                                                              |
| ----- | ---------------------------------------------------------------------- |
| 0     | Repo skeleton, settings, structured logging, Alembic, `GET /health`     |
| 1     | `Source` interface and the football-data.co.uk ingester; raw payloads   |
| 2     | Entity resolution, cleaning, pandera validation, de-margined odds       |
| 3     | Chronological replay, Elo, form, schedule features, versioned store     |
| 4-8   | Backtesting, statistical models, ML and ensembling, serving, hardening   |

### No data leakage

The feature store is the part the later phases lean on hardest, so its correctness is
structural rather than conventional. Features may only read data timestamped **strictly
before** kickoff, they are built by replaying history in kickoff order, and the single door
to the past (`features/replay.py`) raises `LeakageError` for anything at or after its
cut-off. Matches sharing a kickoff never see each other's results. Validation is
walk-forward only; random train/test splits are forbidden.

## Development

Dependencies are managed with `uv` and the lockfile is committed. `make help` lists every
target. Tests never touch the network — the HTTP layer is mocked against recorded files in
`tests/fixtures/`. Integration tests skip themselves unless a database is reachable, so run
`make up` before `make check` to exercise them for real.
