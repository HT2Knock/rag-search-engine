import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from .utils import CACHE_DIR, load_movies


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.embeddings: np.ndarray | None = None
        self.documents: list[dict] | None = None
        self.document_map = {}
        self.movie_embeddings_path = Path(CACHE_DIR) / "movie_embeddings.npy"

    def generate_embedding(self, text: str):
        if not text.strip():
            raise ValueError("Input text are empty for generate embedding")

        return self.model.encode([text])[0]

    def build_embedding(self, documents):
        Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

        movies = [f"{doc['title']}: {doc['description']}" for doc in documents]
        embeddings = self.model.encode(movies, show_progress_bar=True)

        self.embeddings = embeddings
        np.save(self.movie_embeddings_path, embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

        if self.movie_embeddings_path.exists():
            self.embeddings = np.load(self.movie_embeddings_path)

            if len(self.embeddings) == len(self.documents):
                return self.embeddings

        return self.build_embedding(documents)

    def search(self, query: str, limit: int):
        if limit < 1:
            raise ValueError("limit must be a positive integer")

        if self.embeddings is None or len(self.embeddings) < 1:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        if self.documents is None or len(self.documents) < 1:
            raise ValueError(
                "No documents loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)

        similarity_scores = []
        for index, embedding in enumerate(self.embeddings):
            score = cosine_similarity(query_embedding, embedding)
            similarity_scores.append((score, self.documents[index]))

        sorted_scores = sorted(similarity_scores, key=lambda x: x[0], reverse=True)
        return [
            {
                "score": score,
                "title": document["title"],
                "description": document["description"],
            }
            for score, document in sorted_scores[:limit]
        ]


def verify_model():
    semantic_search = SemanticSearch()
    print(f"Model loaded1: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")


def verify_embeddings():
    semantic_search = SemanticSearch()
    documents = load_movies()
    embeddings = semantic_search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_text(text: str):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def embed_query_text(query: str):
    embed_text(query)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def search(query: str, limit: int) -> None:
    semantic_search = SemanticSearch()
    movies = load_movies()
    semantic_search.load_or_create_embeddings(movies)
    results = semantic_search.search(query, limit)

    for index, result in enumerate(results):
        print(
            f"{index}. {result['title']} (score: {result['score']}) \n{result['description']}"
        )


def chunk(text: str, size: int, overlap: int):
    if size <= 0:
        raise ValueError("Chunk size must be greater than 0")

    if overlap >= size:
        raise ValueError("Overlap must be smaller than chunk size")

    words = text.split()
    if not words:
        return

    print(f"Chunking {len(text)} characters")

    index = 0
    count = 1
    step = size - overlap
    while index <= len(words):
        print(f"{count}. {' '.join(words[index : index + size])}")
        index += step
        count += 1

        if index + overlap >= len(words):
            break


def semantic_chunk(text: str, max_size: int, overlap: int):
    if max_size <= 0:
        raise ValueError("Chunk size must be greater than 0")

    if overlap >= max_size:
        raise ValueError("Overlap must be smaller than chunk size")

    words = re.split(r"(?<=[.!?])\s+", text)
    if not words:
        return

    print(f"Semantically chunking {len(text)} characters")

    index = 0
    count = 1
    step = max_size - overlap
    while index <= len(words):
        print(f"{count}. {' '.join(words[index : index + max_size])}")
        index += step
        count += 1

        if index + overlap >= len(words):
            break
