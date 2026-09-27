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
make backtest MODEL=dixon_coles,baseline_market         # walk-forward metrics into MLflow
make report                                             # reports/backtest.md and .html
```

`sources → ingest → raw_payloads → pipeline → teams/matches/odds → features`. Each step is
idempotent: running it twice leaves the database exactly as it was.

With the seasons above that gives 8360 matches per run (11 complete seasons of 380 in each
league), their per-team statistics and odds from four bookmakers, and one feature row per
match with 86 pre-match features.

## Status

**Phase 5 (statistical models) — complete.** Next up is phase 6 (LightGBM, ensembling and
calibration). Phases are delivered one at a time, in the order of spec section 11.

| Phase | Delivered                                                              |
| ----- | ---------------------------------------------------------------------- |
| 0     | Repo skeleton, settings, structured logging, Alembic, `GET /health`     |
| 1     | `Source` interface and the football-data.co.uk ingester; raw payloads   |
| 2     | Entity resolution, cleaning, pandera validation, de-margined odds       |
| 3     | Chronological replay, Elo, form, schedule features, versioned store     |
| 4     | Walk-forward engine, RPS/log loss/Brier/ECE, bootstrap CIs, baselines   |
| 5     | Elo, Poisson, Dixon-Coles and a PyMC hierarchical model, per league     |
| 6-8   | ML and ensembling, serving, hardening                                   |

### The models

Five models sit behind one `MatchModel` interface, and `make backtest MODEL=<name>` runs any
of them over the same folds and the same matches.

| Name                    | Spec | What it is                                                     |
| ----------------------- | ---- | -------------------------------------------------------------- |
| `baseline_frequency`    | M0a  | The home, draw and away rates of the training matches           |
| `baseline_market`       | M0b  | The de-margined closing quote. A benchmark, never deployable    |
| `elo`                   | M1   | Phase 3 Elo ratings through an ordered logistic regression      |
| `poisson`               | M2   | Attack, defence, home advantage and intercept; sum-to-zero      |
| `dixon_coles`           | M3   | M2 plus the low-score correction and time-decay weights         |
| `bayesian_hierarchical` | M4   | PyMC hierarchical Poisson with partial pooling of team strength |

M1 to M4 are fitted **per competition** (spec 7.1): EPL and La Liga teams never meet, so one
joint set of strengths could not tell the leagues apart. M2 to M4 also return the scoreline
matrix, truncated at `models.max_goals`; the mass that falls outside is reported, not absorbed.

Three things are worth knowing before running one:

* `predictor backtest` with no `--model` runs **every** registered model, which now includes
  M4. Name the models explicitly to keep a run short.
* the training frame of a fold is the whole guarded past, both leagues included, so
  `PerCompetition` fits a member for each league on every fold and the run does roughly twice
  the fitting a single league needs. That is the price of letting phase 6 train jointly.
* **M4 is the slow one.** Measured on the 2019-2025 window (12 folds, 24 fits after the
  per-league split, 4 chains of 1000 draws over roughly 4000 matches and 30 teams): about 8
  seconds a fit and 3.2 minutes of the run. All five models together take around ten minutes.
  NUTS runs through `nutpie`, which compiles with numba and so needs no C compiler.

### What the models score

Six test seasons (2019-20 to 2024-25) of both leagues, 4560 matches per model, pooled. RPS is
the primary metric and lower is better; the gap is against the de-margined closing odds.

| model                   | RPS    | log loss | Brier  | ECE    | gap     | 95% CI             |
| ----------------------- | ------ | -------- | ------ | ------ | ------- | ------------------ |
| `baseline_market`       | 0.1934 | 0.9625   | 0.5709 | 0.0134 | —       | —                  |
| `elo`                   | 0.2007 | 0.9855   | 0.5871 | 0.0239 | +0.0073 | [+0.0055, +0.0090] |
| `dixon_coles`           | 0.2060 | 1.0029   | 0.5975 | 0.0227 | +0.0125 | [+0.0100, +0.0150] |
| `bayesian_hierarchical` | 0.2063 | 1.0025   | 0.5979 | 0.0155 | +0.0128 | [+0.0104, +0.0153] |
| `poisson`               | 0.2071 | 1.0059   | 0.5998 | 0.0302 | +0.0137 | [+0.0112, +0.0162] |
| `baseline_frequency`    | 0.2300 | 1.0704   | 0.6477 | 0.0139 | +0.0366 | [+0.0325, +0.0406] |

Nobody beats the market, which is the expected result: a closing price already contains
everyone else's information, and `baseline_market` is marked undeployable because that price
only exists after the market has closed. What the models have to clear is the frequency floor,
and every one of them does so **in each league separately** with a paired bootstrap interval
that excludes zero. Dixon-Coles edges plain Poisson pooled, as spec section 11 asks, though it
gets there by winning the EPL (0.2099 against 0.2144) and losing LaLiga (0.2020 against
0.1999).

Per league, the fitted parameters come out where the literature puts them: a home advantage
near 0.21 in the EPL and 0.28 in LaLiga, rho between -0.08 and +0.01, and a tuned decay mostly
at 0.5 per year. `make report` writes every one of them per league and per season, next to the
metrics above.

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

## Development

Dependencies are managed with `uv` and the lockfile is committed. `make help` lists every
target. Tests never touch the network — the HTTP layer is mocked against recorded files in
`tests/fixtures/`. Integration tests skip themselves unless a database is reachable, so run
`make up` before `make check` to exercise them for real; the backtest ones point MLflow at a
SQLite file of their own, so no tracking server is involved either.

`make check` takes about nine minutes with a database up and under a minute without one. The
difference is the tests marked `slow`: the ones that sample a posterior, and the walk-forward
over the real leagues. `uv run pytest -m "not slow"` skips them while you are iterating.

The recovery tests of the statistical models fit several independent simulated leagues rather
than one, and their tolerances are measured rather than chosen — `tests/unit/simulate.py`
explains why a single simulation would test the draw's luck instead of the model.
