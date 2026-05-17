#!/usr/bin/env python3

import argparse

from lib.semantic_search import (
    chunk,
    embed_query_text,
    embed_text,
    search,
    verify_embeddings,
    verify_model,
)


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify the embeddings model")
    subparsers.add_parser("verify_embeddings", help="Verify the embeddings cache")
    embed_text_parser = subparsers.add_parser(
        "embed_text", help="Generate embedding from text"
    )

    embed_text_parser.add_argument("text", type=str, help="Text to generate embedding")

    embed_query_parser = subparsers.add_parser(
        "embed_query", help="Generate embedding from query"
    )
    embed_query_parser.add_argument(
        "query", type=str, help="Query to generate embedding"
    )

    search_parser = subparsers.add_parser("search", help="Search document embeddings")
    search_parser.add_argument("query", type=str, help="Query to search embedding")
    search_parser.add_argument(
        "--limit", default=5, type=int, help="Limit the number of results"
    )

    chunk_parser = subparsers.add_parser("chunk", help="Chunk long text for embedding")
    chunk_parser.add_argument("text", type=str, help="Characters to chunks")
    chunk_parser.add_argument(
        "--chunk-size", type=int, default=200, help="Size of the chunk"
    )
    chunk_parser.add_argument("--overlap", type=int, help="Tunnable overlap data")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()

        case "verify_embeddings":
            verify_embeddings()

        case "embed_text":
            embed_text(args.text)

        case "embed_query":
            embed_query_text(args.query)

        case "search":
            search(args.query, args.limit)

        case "chunk":
            chunk(args.text, args.chunk_size, args.overlap)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
