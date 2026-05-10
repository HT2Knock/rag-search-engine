import math
import pickle
import string
from collections import Counter, defaultdict
from operator import itemgetter
from pathlib import Path

from nltk.stem.porter import PorterStemmer

from .utils import (
    BM25_B,
    BM25_K1,
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    load_movies,
    load_stopwords,
)

_stop_words = load_stopwords()
_stemmer = PorterStemmer()


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter] = defaultdict(Counter)
        self.doc_lengths: dict[int, int] = {}
        self.avg_doc_length: float = 0.0
        self.index_path = Path(CACHE_DIR) / "index.pkl"
        self.docmap_path = Path(CACHE_DIR) / "docmap.pkl"
        self.term_frequencies_path = Path(CACHE_DIR) / "term_frequencies.pkl"
        self.doc_lengths_path = Path(CACHE_DIR) / "doc_lengths.pkl"

    def _add_document(self, doc_id: int, text: str):
        tokens = tokenize_text(text)
        count = 0
        for token in tokens:
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1
            count += 1

        self.doc_lengths[doc_id] = count

    def _get_avg_doc_length(self) -> float:
        return self.avg_doc_length

    def get_document_ids(self, token: str) -> list[int]:
        return sorted(self.index[token])

    def get_tf(self, doc_id: int, token: str) -> int:
        return self.term_frequencies[doc_id][token]

    def get_idf(self, token: str) -> float:
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.index[token])
        return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

    def get_tf_idf(self, doc_id: int, token: str) -> float:
        return self.get_tf(doc_id, token) * self.get_idf(token)

    def get_bm25_idf(self, token: str) -> float:
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.index[token])
        return math.log(
            (total_doc_count - term_match_doc_count + 0.5)
            / (term_match_doc_count + 0.5)
            + 1
        )

    def get_bm25_tf(
        self, doc_id: int, token: str, k1: float = BM25_K1, b: float = BM25_B
    ) -> float:
        length_norm = (
            1 - b + b * (self.doc_lengths[doc_id] / self._get_avg_doc_length())
        )
        tf = self.get_tf(doc_id, token)
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def bm25(self, doc_id: int, token: str) -> float:
        return self.get_bm25_tf(doc_id, token) * self.get_bm25_idf(token)

    def bm25_search(self, query: str, limit: int) -> list[dict]:
        tokens = tokenize_text(query)
        if not tokens:
            return []

        candidate_docs = set()
        for token in tokens:
            candidate_docs.update(self.index[token])

        scores = defaultdict(float)
        for doc_id in candidate_docs:
            for token in tokens:
                scores[doc_id] += self.bm25(doc_id, token)

        sorted_results = sorted(
            scores.items(), key=itemgetter(1), reverse=True
        )
        return [
            {
                "score": score,
                "doc_id": doc_id,
                "title": self.docmap[doc_id]["title"],
            }
            for doc_id, score in sorted_results[:limit]
        ]

    def build(self):
        movies = load_movies()
        for movie in movies:
            doc_id = movie["id"]
            self._add_document(doc_id, f"{movie['title']} {movie['description']}")
            self.docmap[doc_id] = movie

        if self.doc_lengths:
            self.avg_doc_length = (
                sum(self.doc_lengths.values()) / len(self.doc_lengths)
            )

    def save(self):
        Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)

        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)

        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)

        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        if not self.index_path.exists():
            raise FileNotFoundError(f"Missing required file: {self.index_path}")

        if not self.docmap_path.exists():
            raise FileNotFoundError(f"Missing required file: {self.docmap_path}")

        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)

        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)

        with open(self.term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)

        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)

        if self.doc_lengths:
            self.avg_doc_length = (
                sum(self.doc_lengths.values()) / len(self.doc_lengths)
            )


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    idx.load()
    results = []

    query_tokens = tokenize_text(query)

    doc_ids = set()
    for token in query_tokens:
        doc_ids.update(idx.index[token])

    for doc in doc_ids:
        results.append(idx.docmap[doc])
        if len(results) >= limit:
            break

    return results


def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    return [_stemmer.stem(t) for t in tokens if t not in _stop_words]


def preprocess_text(text: str) -> str:
    return text.lower().translate(str.maketrans("", "", string.punctuation))
