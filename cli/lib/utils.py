import json
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOP_WORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")

CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

DEFAULT_SEARCH_LIMIT = 5

BM25_K1 = 1.5
BM25_B = 0.75


def load_movies() -> list[dict[str, Any]]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
        return data["movies"]


def load_stopwords():
    with open(STOP_WORDS_PATH, "r") as f:
        return f.read().splitlines()
