# Sports Results Prediction Platform

1X2 probabilities and scoreline distributions for the English Premier League (EPL) and
LaLiga (LALIGA). Specification: `docs/SPEC.pdf`. Working rules for contributors and for
Claude Code: `CLAUDE.md`.

## Quick start

```bash
cp .env.example .env          # fill in POSTGRES_PASSWORD
make up                       # db + api + mlflow
curl http://127.0.0.1:8000/health
make migrate                  # alembic upgrade head
make check                    # ruff + mypy + pytest
```

Current status: **phase 0 (foundations)** — skeleton, configuration, health endpoint.
Ingestion, features, models and the dashboard arrive in later phases (spec section 11).
