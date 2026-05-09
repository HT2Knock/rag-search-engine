import math
import os
import pickle
import string
from collections import Counter, defaultdict
from itertools import islice
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


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter] = defaultdict(Counter)
        self.doc_lengths: dict[int, int] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequencies_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def __add_document(self, doc_id: int, text: str):
        tokens = tokenize_text(text)
        count = 0
        for token in tokens:
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1
            count += 1

        self.doc_lengths[doc_id] = count

    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0.0

        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def get_document_ids(self, term: str) -> list[int]:
        tokens = tokenize_text(term)
        if len(tokens) > 1:
            raise ValueError("Term length greater than 1")

        return sorted(self.index[tokens[0]])

    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = tokenize_text(term)
        if len(tokens) > 1:
            raise ValueError("Term length greater than 1")

        return self.term_frequencies[doc_id][tokens[0]]

    def get_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) > 1:
            raise ValueError("Term length greater than 1")

        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.get_document_ids(term))
        return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

    def get_tf_idf(self, doc_id: int, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) > 1:
            raise ValueError("Term length greater than 1")

        return self.get_tf(doc_id, term) * self.get_idf(term)

    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) > 1:
            raise ValueError("Term length greater than 1")

        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.get_document_ids(term))
        return math.log(
            (total_doc_count - term_match_doc_count + 0.5)
            / (term_match_doc_count + 0.5)
            + 1
        )

    def get_bm25_tf(
        self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
    ) -> float:
        length_norm = (
            1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())
        )
        tf = self.get_tf(doc_id, term)
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def bm25(self, doc_id: int, term: str) -> float:
        tf = self.get_bm25_tf(doc_id, term)
        idf = self.get_bm25_idf(term)

        return tf * idf

    def bm25_search(self, query, limit) -> list[dict]:
        tokens = tokenize_text(query)
        scores = defaultdict(int)

        for doc_id in self.docmap:
            for token in tokens:
                scores[doc_id] += self.bm25(doc_id, token)
        sorted_scores = dict(sorted(scores.items(), key=itemgetter(1), reverse=True))
        return [
            {
                "score": sorted_scores[doc_id],
                "doc_id": doc_id,
                "title": self.docmap[doc_id]["title"],
            }
            for doc_id in islice(sorted_scores, limit)
        ]

    def build(self):
        movies = load_movies()
        for movie in movies:
            doc_id = movie["id"]
            self.__add_document(doc_id, f"{movie['title']} {movie['description']}")
            self.docmap[doc_id] = movie

    def save(self):
        os.makedirs(CACHE_DIR, exist_ok=True)

        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)

        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)

        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)

        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        if not Path(self.index_path).exists():
            raise FileNotFoundError(f"Missing required file: {self.index_path}")

        if not Path(self.docmap_path).exists():
            raise FileNotFoundError(f"Missing required file: {self.docmap_path}")

        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)

        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)

        with open(self.term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)

        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    idx.load()
    results = []

    stemmer = PorterStemmer()
    query_tokens = tokenize_text(query, stemmer)

    doc_ids = set()
    for token in query_tokens:
        doc_ids.update(idx.get_document_ids(token))

    for id in doc_ids:
        results.append(idx.docmap[id])
        if len(results) >= limit:
            break

    return results


def tokenize_text(text: str, stemmer: PorterStemmer | None = None) -> list[str]:
    if not stemmer:
        stemmer = PorterStemmer()

    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []

    for token in tokens:
        if token not in _stop_words:
            stemmed = stemmer.stem(token)
            valid_tokens.append(stemmed)

    return valid_tokens


def preprocess_text(text: str) -> str:
    return text.lower().translate(str.maketrans("", "", string.punctuation))
