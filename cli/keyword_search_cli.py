import argparse

from lib.search import InvertedIndex, build_command, search_command, tf_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movie using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build movies inverted index")

    tf_parser = subparsers.add_parser(
        "tf", help="Get term frequencies from a movie doc"
    )
    tf_parser.add_argument("doc_id", type=int, help="Movie doc id")
    tf_parser.add_argument("term", type=str, help="Query term")

    args = parser.parse_args()

    match args.command:
        case "search":
            print("Searching for:", args.query)
            results = search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']}")

        case "build":
            print("Start buidling inverted index")
            build_command()
            print("Finish buidling inverted index")

        case "tf":
            print(f"Searching for {args.term} in {args.doc_id}:")
            print(tf_command(args.doc_id, args.term))

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
