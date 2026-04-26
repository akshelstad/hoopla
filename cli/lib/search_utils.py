import json
import os

from cli.lib.errors import DataLoadError, MissingDataFileError, InvalidDataFormatError

DEFAULT_SEARCH_LIMIT = 5

BM25_K1 = 1.5
BM25_B = 0.75

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")

CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")


def load_movies() -> list[dict]:
    try:
        with open(DATA_PATH, "r") as f:
            data = json.load(f)
            return data["movies"]
    except FileNotFoundError as e:
        raise MissingDataFileError(f"Movies data file not found: {DATA_PATH}") from e
    except json.JSONDecodeError as e:
        raise InvalidDataFormatError(
            f"Invalid JSON format in movies data file: {DATA_PATH}"
        ) from e


def load_stopwords() -> list[str]:
    try:
        with open(STOPWORDS_PATH, "r") as f:
            return f.read().splitlines()
    except FileNotFoundError as e:
        raise MissingDataFileError(
            f"Stopwords data file not found: {STOPWORDS_PATH}"
        ) from e
    except Exception as e:
        raise DataLoadError(
            f"Error loading stopwords data file: {STOPWORDS_PATH}"
        ) from e
