#!/usr/bin/env python
"""Seed the mock order/invoice backend's database with sample data.

    DATABASE_URL=postgresql://user:pass@host/db python scripts/seed_mock_backend.py

Safe to re-run: seeding is idempotent (existing rows are left untouched).
Defaults to a local SQLite file if $DATABASE_URL isn't set.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mcs.mock_backend.db import make_engine, make_session_factory, seed


def main() -> int:
    engine = make_engine()
    session_factory = make_session_factory(engine)
    with session_factory() as session:
        seed(session)
    print(f"seeded mock backend database: {engine.url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
