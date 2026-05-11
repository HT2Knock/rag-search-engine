from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def generate_embedding(self, text: str):
        if not text.strip():
            raise ValueError("Input text are empty for generate embedding")

        return self.model.encode([text])[0]


def verify_model():
    semanticSearch = SemanticSearch()
    print(f"Model loaded1: {semanticSearch.model}")
    print(f"Max sequence length: {semanticSearch.model.max_seq_length}")


def embed_text(text: str):
    semanticSearch = SemanticSearch()
    embedding = semanticSearch.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")
