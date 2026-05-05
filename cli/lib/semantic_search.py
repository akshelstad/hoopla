import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import similarity

from .search_utils import CACHE_DIR, DEFAULT_SEARCH_LIMIT, load_movies
from .errors import SemanticSearchError


class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        self.embeddings_path = os.path.join(CACHE_DIR, "movie_embeddings.npy")

    def generate_embedding(self, text: str):
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty text")

        return self.model.encode([text])[0]

    def build_embeddings(self, documents):
        doc_list = self.__build_documents(documents)
        self.embeddings = self.model.encode(doc_list, show_progress_bar=True)

        os.makedirs(os.path.dirname(self.embeddings_path), exist_ok=True)

        with open(self.embeddings_path, "wb") as f:
            np.save(f, self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.__build_documents(documents)

        if os.path.exists(self.embeddings_path):
            with open(self.embeddings_path, "rb") as f:
                self.embeddings = np.load(f)

            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)

    def search(self, query, limit):
        if self.embeddings is None or self.embeddings.size == 0:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        if self.documents is None or len(self.documents) == 0:
            raise ValueError(
                "No documents loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)

        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = cosine_similarity(query_embedding, doc_embedding)
            similarities.append((similarity, self.documents[i]))

        similarities.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, doc in similarities[:limit]:
            results.append(
                {
                    "score": score,
                    "title": doc["title"],
                    "description": doc["description"],
                }
            )

        return results

    def __build_documents(self, documents):
        self.documents = documents
        self.document_map = {}
        doc_list = []

        for doc in documents:
            self.document_map[doc["id"]] = doc
            doc_list.append(f"{doc['title']}: {doc['description']}")

        return doc_list


def verify_model() -> None:
    search_instance = SemanticSearch()
    print(f"Model loaded: {search_instance.model}")
    print(f"Max sequence length: {search_instance.model.max_seq_length}")


def embed_text(text: str) -> None:
    search_instance = SemanticSearch()
    embedding = search_instance.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    search_instance = SemanticSearch()
    movies = load_movies()
    embeddings = search_instance.load_or_create_embeddings(movies)
    if embeddings is None:
        raise SemanticSearchError("Failed to verify embeddings")
    print(f"Number of docs: {len(movies)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_query_text(query: str):
    search_instance = SemanticSearch()
    embedding = search_instance.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def semantic_search(query: str, limit: int):
    search_instance = SemanticSearch()
    documents = load_movies()
    search_instance.load_or_create_embeddings(documents)

    results = search_instance.search(query, limit)

    print(f"Query: {query}")
    print(f"Top {len(results)} results:")
    print()

    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"    {result['description'][:100]}...")
        print()
