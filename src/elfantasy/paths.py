"""Filesystem locations used across the project."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
RULES_PATH = CONFIG_DIR / "game_rules.yaml"

# Data directory can be moved with ELF_DATA_DIR (e.g. to a larger disk).
DATA_DIR = Path(os.environ.get("ELF_DATA_DIR", REPO_ROOT / "data"))
RAW_CACHE_DIR = DATA_DIR / "raw"          # cached API responses (immutable resources only)
PROCESSED_DIR = DATA_DIR / "processed"    # tidy parquet tables

# Fantasy-app snapshots cannot be re-downloaded later, so they live in the repo (committed).
SNAPSHOT_DIR = REPO_ROOT / "snapshots"
