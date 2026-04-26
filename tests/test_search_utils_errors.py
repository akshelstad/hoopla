import builtins
from pathlib import Path

import pytest

from cli.lib.errors import DataLoadError, InvalidDataFormatError, MissingDataFileError
from cli.lib import search_utils


def test_load_movies_raises_when_data_file_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(search_utils, "DATA_PATH", "missing-movies.json")

    with pytest.raises(MissingDataFileError, match="Movies data file not found"):
        search_utils.load_movies()


def test_load_movies_raises_when_json_is_invalid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bad_movies = tmp_path / "movies.json"
    bad_movies.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setattr(search_utils, "DATA_PATH", str(bad_movies))

    with pytest.raises(InvalidDataFormatError, match="Invalid JSON format"):
        search_utils.load_movies()


def test_load_stopwords_raises_when_file_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(search_utils, "STOPWORDS_PATH", "missing-stopwords.txt")

    with pytest.raises(MissingDataFileError, match="Stopwords data file not found"):
        search_utils.load_stopwords()


def test_load_stopwords_wraps_os_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def raising_open(*args: object, **kwargs: object) -> object:
        raise OSError("permission denied")

    monkeypatch.setattr(builtins, "open", raising_open)

    with pytest.raises(DataLoadError, match="Error loading stopwords data file"):
        search_utils.load_stopwords()
