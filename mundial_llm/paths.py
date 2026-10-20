"""Where the project's data files live: the data/ folder at the repository root."""

import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")

HISTORICAL_MATCHES_FILE = os.path.join(DATA_DIR, "historical_matches_with_xg.csv")
LEARNING_MEMORY_FILE = os.path.join(DATA_DIR, "learning_memory.json")
TRAINED_MATCHES_FILE = os.path.join(DATA_DIR, "trained_matches.json")
