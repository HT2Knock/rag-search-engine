import argparse

from lib.search import InvertedIndex, build_command, search_command, tokenize_text
from lib.utils import BM25_B, BM25_K1


def _get_term_token(term: str) -> str | None:
    tokens = tokenize_text(term)
    if not tokens:
        print("Error: term contains no valid content (only stop words)")
        return None
    if len(tokens) > 1:
        print(f"Warning: using first token '{tokens[0]}' from '{term}'")
    return tokens[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movie using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build inverted index from movie data")

    tf_parser = subparsers.add_parser(
        "tf", help="Show how many times a term appears in a specific document"
    )
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to look up")

    idf_parser = subparsers.add_parser(
        "idf", help="Show how rare or common a term is across all documents"
    )
    idf_parser.add_argument("term", type=str, help="Term to look up")

    tfidf_parser = subparsers.add_parser(
        "tfidf", help="Compute TF-IDF relevance score for a term in a document"
    )
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to score")

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument(
        "term", type=str, help="Term to get BM25 IDF score for"
    )

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )

    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("--limit", type=int, default=5, help="Query limit")
    args = parser.parse_args()

    match args.command:
        case "search":
            print("Searching for:", args.query)
            results = search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']}")

        case "build":
            print("Start building inverted index")
            build_command()
            print("Finish building inverted index")

        case "tf":
            idx = InvertedIndex()
            idx.load()
            token = _get_term_token(args.term)
            if token is None:
                return
            print(f"Searching for {args.term} in {args.doc_id}:")
            print(idx.get_tf(args.doc_id, token))

        case "idf":
            idx = InvertedIndex()
            idx.load()
            token = _get_term_token(args.term)
            if token is None:
                return
            idf = idx.get_idf(token)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "tfidf":
            idx = InvertedIndex()
            idx.load()
            token = _get_term_token(args.term)
            if token is None:
                return
            tf_idf = idx.get_tf_idf(args.doc_id, token)
            print(
                f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}"
            )

        case "bm25idf":
            idx = InvertedIndex()
            idx.load()
            token = _get_term_token(args.term)
            if token is None:
                return
            bm25idf = idx.get_bm25_idf(token)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")

        case "bm25tf":
            idx = InvertedIndex()
            idx.load()
            token = _get_term_token(args.term)
            if token is None:
                return
            print(f"Searching for {args.term} in {args.doc_id}:")
            bm25tf = idx.get_bm25_tf(args.doc_id, token)
            print(
                f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}"
            )

        case "bm25search":
            idx = InvertedIndex()
            idx.load()
            results = idx.bm25_search(args.query, args.limit)
            for i, res in enumerate(results, 1):
                print(
                    f"{i}. ({res['doc_id']}) {res['title']} - Score: {res['score']:.2f}"
                )

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
