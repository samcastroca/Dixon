"""baseline: empty starting point; tables arrive with their own phase

Revision ID: 0001_baseline
Revises:
Create Date: 2026-09-26

"""

from __future__ import annotations

revision: str = "0001_baseline"
down_revision: str | None = None
branch_labels: None = None
depends_on: None = None


def upgrade() -> None:
    """Nothing yet: phase 0 ships no tables."""


def downgrade() -> None:
    """Nothing to undo."""
