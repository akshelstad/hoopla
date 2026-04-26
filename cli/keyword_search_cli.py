#!/usr/bin/env python3

import argparse
import sys

from lib.errors import (
    KeywordSearchError,
    MissingIndexError,
    CorruptIndexError,
    InvalidQueryError,
    InvalidDocumentError,
    DataLoadError,
)

from lib.keyword_search import (
    search_command,
    build_command,
    tf_command,
    idf_command,
    tfidf_command,
    bm25_idf_command,
    bm25_tf_command,
    bm25_search_command,
)

from lib.search_utils import BM25_K1, BM25_B, DEFAULT_SEARCH_LIMIT


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build database of movies")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument(
        "--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Results limit"
    )
    bm25search_parser.add_argument(
        "--k1", type=float, default=BM25_K1, help="Tunable BM25 k1 parameter"
    )
    bm25search_parser.add_argument(
        "--b", type=float, default=BM25_B, help="Tunable BM25 b parameter"
    )

    tf_parser = subparsers.add_parser(
        "tf", help="Get term frequency for a given document ID and term"
    )
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Search term for term frequency")

    idf_parser = subparsers.add_parser(
        "idf", help="Get inverse document frequency for a given term"
    )
    idf_parser.add_argument(
        "term", type=str, help="Search term for inverse document frequency"
    )

    tfidf_parser = subparsers.add_parser(
        "tfidf", help="Get TF-IDF score for a given term and document ID"
    )
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Search term for TF-IDF score")

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument(
        "term", type=str, help="Search term for BM25 IDF score"
    )

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 k1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )

    args = parser.parse_args()

    try:
        match args.command:
            case "build":
                print("Building inverted index...")
                build_command()
                print("Index built successfully.")
            case "search":
                print("Searching for:", args.query)
                results = search_command(args.query)
                for i, res in enumerate(results, 1):
                    print(f"{i}. ({res['id']}) {res['title']}")
            case "bm25search":
                print("Searching for:", args.query)
                results = bm25_search_command(args.query, args.limit, args.k1, args.b)
                for i, res in enumerate(results, 1):
                    print(
                        f"{i}. ({res['id']}) {res['title']} - Score: {res['score']:.2f}"
                    )

            case "tf":
                print("Searching term frequency for:", args.term)
                tf = tf_command(args.term, args.doc_id)
                if tf != -1:
                    print(
                        f"Term '{args.term}' appeared {tf} times in document '{args.doc_id}'"
                    )
            case "idf":
                print("Searching inverse document frequency for:", args.term)
                idf = idf_command(args.term)
                if idf != -1:
                    print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
            case "tfidf":
                print("Searching TF-IDF score for:", args.term)
                tfidf = tfidf_command(args.term, args.doc_id)
                if tfidf != -1:
                    print(
                        f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tfidf:.2f}"
                    )
            case "bm25idf":
                print("Searching BM25 IDF score for:", args.term)
                bm25_idf = bm25_idf_command(args.term)
                if bm25_idf != -1:
                    print(f"BM25 IDF score of '{args.term}': {bm25_idf:.2f}")
            case "bm25tf":
                print("Searching BM25 TF score for:", args.term)
                bm25_tf = bm25_tf_command(args.term, args.doc_id, args.k1, args.b)
                if bm25_tf != -1:
                    print(
                        f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25_tf:.2f}"
                    )
            case _:
                parser.print_help()
                sys.exit(1)
    except (
        MissingIndexError,
        CorruptIndexError,
        InvalidQueryError,
        InvalidDocumentError,
        DataLoadError,
    ) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)
    except KeywordSearchError as e:
        print(f"Search error: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"Unexpected internal error: {e}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
