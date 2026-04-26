import os
import pickle
import string
import math
from collections import defaultdict, Counter

from nltk.stem import PorterStemmer

from .search_utils import (
    BM25_K1,
    BM25_B,
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    load_movies,
    load_stopwords,
)

from .errors import (
    MissingIndexError,
    CorruptIndexError,
    InvalidQueryError,
    InvalidDocumentError,
)


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies = defaultdict(Counter)
        self.doc_lengths = defaultdict(int)
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequencies_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def build(self) -> None:
        movies = load_movies()
        for m in movies:
            self.__add_document(m["id"], f"{m['title']} {m['description']}")
            self.docmap[m["id"]] = m

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)

        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self) -> None:
        required = [
            self.index_path,
            self.docmap_path,
            self.term_frequencies_path,
            self.doc_lengths_path,
        ]
        missing = [path for path in required if not os.path.exists(path)]
        if missing:
            raise MissingIndexError(
                f"Keyword index is missing cache files: {','.join(os.path.basename(p) for p in missing)}"
            )

        try:
            with open(self.index_path, "rb") as f:
                self.index = pickle.load(f)
            with open(self.docmap_path, "rb") as f:
                self.docmap = pickle.load(f)
            with open(self.term_frequencies_path, "rb") as f:
                self.term_frequencies = pickle.load(f)
            with open(self.doc_lengths_path, "rb") as f:
                self.doc_lengths = pickle.load(f)
        except (pickle.UnpicklingError, EOFError, AttributeError, ValueError) as e:
            raise CorruptIndexError("Keyword index cache is corrupted") from e

    def bm25(
        self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
    ) -> float:
        bm25_tf = self.get_bm25_tf(doc_id, term, k1, b)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf

    def bm25_search(
        self,
        query: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> dict[int, float]:
        query_tokens = tokenize_text(query)

        scores = {}
        for token in query_tokens:
            matching_doc_ids = self.get_documents(token)
            for doc_id in matching_doc_ids:
                doc_score = self.bm25(doc_id, token, k1, b)
                scores[doc_id] = scores.get(doc_id, 0) + doc_score

        ranked_doc_ids = dict(
            sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
        )
        return ranked_doc_ids

    def __add_document(self, doc_id: int, text: str):
        tokens = tokenize_text(text)
        self.term_frequencies[doc_id].update(tokens)
        self.doc_lengths[doc_id] = len(tokens)
        for token in set(tokens):
            self.index[token].add(doc_id)

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths or len(self.doc_lengths) == 0:
            return 0.0

        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def get_tf(self, doc_id: int, term: str) -> int:
        if doc_id not in self.docmap:
            raise InvalidDocumentError(f"Unknown document id: {doc_id}")

        tokens = tokenize_text(term)
        if not tokens:
            raise InvalidQueryError("Query is empty after preprocessing")
        if len(tokens) != 1:
            raise InvalidQueryError("Term must resolve to exactly one token")

        token = tokens[0]
        return self.term_frequencies[doc_id][token]

    def get_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if not tokens:
            raise InvalidQueryError("Query is empty after preprocessing")
        if len(tokens) != 1:
            raise InvalidQueryError("Term must resolve to exactly one token")

        token = tokens[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count + 1) / (term_doc_count + 1))

    def get_tfidf(self, doc_id: int, term: str) -> float:
        return self.get_tf(doc_id, term) * self.get_idf(term)

    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if not tokens:
            raise InvalidQueryError("Query is empty after preprocessing")
        if len(tokens) != 1:
            raise InvalidQueryError("Term must resolve to exactly one token")

        token = tokens[0]
        n = len(self.docmap)  # total documents
        df = len(self.index[token])  # document frequency of term
        return math.log((n - df + 0.5) / (df + 0.5) + 1)

    def get_bm25_tf(
        self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
    ) -> float:
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths.get(doc_id, 0)
        avg_doc_length = self.__get_avg_doc_length()
        if avg_doc_length > 0:
            norm_length = 1 - b + b * (doc_length / avg_doc_length)
        else:
            norm_length = 1
        return (tf * (k1 + 1)) / (tf + k1 * norm_length)


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = load_index()
    results = []
    query_tokens = tokenize_text(query)
    seen, results = set(), []
    for token in query_tokens:
        matching_doc_ids = idx.get_documents(token)
        for doc_id in matching_doc_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            doc = idx.docmap[doc_id]
            results.append(doc)
            if len(results) >= limit:
                return results

    return results


def bm25_search_command(
    query: str,
    limit: int = DEFAULT_SEARCH_LIMIT,
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> list[dict]:
    idx = load_index()
    results = []
    ranked_doc_ids = idx.bm25_search(query, limit, k1, b)
    for doc_id, score in ranked_doc_ids.items():
        doc = idx.docmap[doc_id]
        doc["score"] = score
        results.append(doc)
        if len(results) >= limit:
            return results

    return results


def tf_command(query: str, doc_id: int) -> int:
    idx = load_index()
    return idx.get_tf(doc_id, query)


def bm25_tf_command(
    query: str, doc_id: int, k1: float = BM25_K1, b: float = BM25_B
) -> float:
    idx = load_index()
    return idx.get_bm25_tf(doc_id, query, k1, b)


def idf_command(query: str) -> float:
    idx = load_index()
    return idx.get_idf(query)


def bm25_idf_command(query: str) -> float:
    idx = load_index()
    return idx.get_bm25_idf(query)


def tfidf_command(query: str, doc_id: int) -> float:
    idx = load_index()
    return idx.get_tfidf(doc_id, query)


def load_index() -> InvertedIndex:
    idx = InvertedIndex()

    try:
        idx.load()
    except FileNotFoundError as e:
        raise MissingIndexError(
            "Keyword index not found. Run the build command first."
        ) from e

    return idx


def has_matching_tokens(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tokenize_text(text: str) -> list[str]:
    stemmer = PorterStemmer()
    stop_words = load_stopwords()
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token and token not in stop_words:
            stemmed_token = stemmer.stem(token)
            valid_tokens.append(stemmed_token)
    return valid_tokens


# alternate version

# def tokenize_text(text: str) -> list[str]:
#     text = preprocess_text(text)
#     tokens = text.split()
#     valid_tokens = []
#     for token in tokens:
#         if token:
#             valid_tokens.append(token)
#     stop_words = load_stopwords()
#     filtered_words = []
#     for word in valid_tokens:
#         if word not in stop_words:
#             filtered_words.append(word)
#     stemmer = PorterStemmer()
#     stemmed_words = []
#     for word in filtered_words:
#         stemmed_words.append(stemmer.stem(word))
#     return stemmed_words
