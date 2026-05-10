from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def verify_model():
    semanticSearch = SemanticSearch()
    print(f"Model loaded1: {semanticSearch.model}")
    print(f"Max sequence length: {semanticSearch.model.max_seq_length}")
