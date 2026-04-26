from collections import Counter, defaultdict
from pathlib import Path

import pytest

from cli.lib.errors import (
    CorruptIndexError,
    InvalidDocumentError,
    InvalidQueryError,
    MissingIndexError,
)
from cli.lib.keyword_search import (
    InvertedIndex,
    bm25_idf_command,
    load_index,
    search_command,
    tf_command,
)


@pytest.fixture
def indexed_document(monkeypatch: pytest.MonkeyPatch) -> InvertedIndex:
    monkeypatch.setattr("cli.lib.keyword_search.load_stopwords", lambda: [])
    idx = InvertedIndex()
    idx.docmap = {1: {"id": 1, "title": "Alien", "description": "Space horror"}}
    idx.index = defaultdict(set, {"alien": {1}})
    idx.term_frequencies = defaultdict(Counter, {1: Counter({"alien": 2})})
    idx.doc_lengths = defaultdict(int, {1: 3})
    return idx


def test_inverted_index_load_raises_when_cache_files_are_missing(tmp_path: Path) -> None:
    idx = InvertedIndex()
    idx.index_path = str(tmp_path / "index.pkl")
    idx.docmap_path = str(tmp_path / "docmap.pkl")
    idx.term_frequencies_path = str(tmp_path / "term_frequencies.pkl")
    idx.doc_lengths_path = str(tmp_path / "doc_lengths.pkl")

    with pytest.raises(MissingIndexError, match="Keyword index is missing cache files"):
        idx.load()


def test_inverted_index_load_raises_when_cache_is_corrupt(tmp_path: Path) -> None:
    idx = InvertedIndex()
    idx.index_path = str(tmp_path / "index.pkl")
    idx.docmap_path = str(tmp_path / "docmap.pkl")
    idx.term_frequencies_path = str(tmp_path / "term_frequencies.pkl")
    idx.doc_lengths_path = str(tmp_path / "doc_lengths.pkl")

    for path in (
        idx.index_path,
        idx.docmap_path,
        idx.term_frequencies_path,
        idx.doc_lengths_path,
    ):
        Path(path).write_bytes(b"not-a-pickle")

    with pytest.raises(CorruptIndexError, match="Keyword index cache is corrupted"):
        idx.load()


def test_get_tf_raises_for_unknown_document(indexed_document: InvertedIndex) -> None:
    with pytest.raises(InvalidDocumentError, match="Unknown document id: 99"):
        indexed_document.get_tf(99, "alien")


@pytest.mark.parametrize(
    ("query", "method_name"),
    [
        ("", "get_tf"),
        ("", "get_idf"),
        ("", "get_bm25_idf"),
        ("alien queen", "get_tf"),
        ("alien queen", "get_idf"),
        ("alien queen", "get_bm25_idf"),
    ],
)
def test_single_term_methods_raise_for_invalid_queries(
    indexed_document: InvertedIndex, query: str, method_name: str
) -> None:
    method = getattr(indexed_document, method_name)
    args = (1, query) if method_name == "get_tf" else (query,)

    with pytest.raises(InvalidQueryError):
        method(*args)


def test_load_index_propagates_missing_index_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raising_load(self: InvertedIndex) -> None:
        raise MissingIndexError("index missing")

    monkeypatch.setattr(InvertedIndex, "load", raising_load)

    with pytest.raises(MissingIndexError, match="index missing"):
        load_index()


def test_tf_command_propagates_invalid_document_errors(
    monkeypatch: pytest.MonkeyPatch, indexed_document: InvertedIndex
) -> None:
    monkeypatch.setattr("cli.lib.keyword_search.load_index", lambda: indexed_document)

    with pytest.raises(InvalidDocumentError, match="Unknown document id: 99"):
        tf_command("alien", 99)


def test_bm25_idf_command_propagates_invalid_query_errors(
    monkeypatch: pytest.MonkeyPatch, indexed_document: InvertedIndex
) -> None:
    monkeypatch.setattr("cli.lib.keyword_search.load_index", lambda: indexed_document)

    with pytest.raises(InvalidQueryError):
        bm25_idf_command("alien queen")


def test_search_command_propagates_missing_index_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raising_load_index() -> InvertedIndex:
        raise MissingIndexError("build required")

    monkeypatch.setattr("cli.lib.keyword_search.load_index", raising_load_index)

    with pytest.raises(MissingIndexError, match="build required"):
        search_command("alien")
