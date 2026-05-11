from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from .utils import CACHE_DIR, load_movies


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
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
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")
