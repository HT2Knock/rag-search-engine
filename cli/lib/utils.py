import json
import os
from typing import Any

DATA_PATH = os.path.join(os.getcwd(), "data", "movies.json")
STOP_WORDS_PATH = os.path.join(os.getcwd(), "data", "stopwords.txt")
DEFAULT_SEARCH_LIMIT = 5


def load_movies() -> list[dict[str, Any]]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
        return data["movies"]


def load_stopwords():
    with open(STOP_WORDS_PATH, "r") as f:
        return f.read().splitlines()
