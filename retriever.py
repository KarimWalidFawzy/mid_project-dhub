"""Document loading and retrieval using the installed vector libraries.

The assistant prefers sentence embeddings when they are available, while keeping
an offline TF-IDF fallback for small or restricted environments.
"""

from __future__ import annotations

import csv
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer as SentenceTransformers
except ImportError:  # pragma: no cover - available when dependency is installed.
    SentenceTransformers = None

TOKEN_RE = re.compile(r"[a-zA-Z0-9_']+")
SUPPORTED_SUFFIXES = {".txt", ".md", ".csv", ".pdf"}


@dataclass(frozen=True)
class Document:
    source: str
    text: str
    chunk: int = 0


@dataclass(frozen=True)
class SearchResult:
    document: Document
    score: float


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def _read_file(path: Path) -> str:
    if path.suffix.lower() == ".csv":
        try:
            frame = pd.read_csv(path, dtype=str, na_filter=False)
            if frame.empty:
                return ""
            rows = frame.fillna("").astype(str)
            return "\n".join(
                " | ".join(str(value) for value in row)
                for _, row in rows.iterrows()
            )
        except Exception:
            with path.open(newline="", encoding="utf-8-sig") as handle:
                return "\n".join(" | ".join(row) for row in csv.reader(handle))
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as error:
            raise RuntimeError("Install pypdf to index PDF files: pip install pypdf") from error
        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def load_documents(directory: str | Path, chunk_size: int = 900) -> list[Document]:
    """Load supported files and split them into manageable text chunks."""
    root = Path(directory)
    if not root.exists():
        return []
    documents: list[Document] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        text = _read_file(path).strip()
        words = text.split()
        for index in range(0, len(words), chunk_size):
            chunk = " ".join(words[index : index + chunk_size]).strip()
            if chunk:
                documents.append(Document(str(path), chunk, index // chunk_size))
    return documents


class TfidfRetriever:
    """Retrieve chunks with sentence embeddings and a local TF-IDF fallback."""

    def __init__(
        self,
        documents: list[Document],
        embedding_model_name: str = "all-MiniLM-L6-v2",
        use_embeddings: bool = True,
    ) -> None:
        self.documents = documents
        self.embedding_model_name = embedding_model_name
        self.term_frequencies = [Counter(_tokens(document.text)) for document in documents]
        self.document_frequency = Counter(
            token for frequencies in self.term_frequencies for token in frequencies
        )
        self._embedding_model = None
        self._embeddings: np.ndarray | None = None

        if use_embeddings and SentenceTransformers is not None and documents:
            try:
                self._embedding_model = SentenceTransformers(embedding_model_name)
                self._embeddings = self._embedding_model.encode(
                    [document.text for document in documents],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            except Exception:
                self._embedding_model = None
                self._embeddings = None

    @property
    def backend(self) -> str:
        """Return the active retrieval strategy for diagnostics and startup logs."""
        return "sentence-transformers" if self._embeddings is not None else "tf-idf"

    def search(self, query: str, limit: int = 4) -> list[SearchResult]:
        if self._embedding_model is not None and self._embeddings is not None:
            return self._semantic_search(query, limit)
        return self._tfidf_search(query, limit)

    def _semantic_search(self, query: str, limit: int = 4) -> list[SearchResult]:
        if not query.strip():
            return []
        query_embedding = self._embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        similarities = self._embeddings @ query_embedding
        ranked = sorted(
            zip(self.documents, similarities.tolist()),
            key=lambda item: item[1],
            reverse=True,
        )
        return [
            SearchResult(document, float(score))
            for document, score in ranked[:limit]
            if score > 0.0
        ]

    def _tfidf_search(self, query: str, limit: int = 4) -> list[SearchResult]:
        query_terms = Counter(_tokens(query))
        if not query_terms:
            return []
        query_vector = self._vector(query_terms, len(self.documents))
        results: list[SearchResult] = []
        for document, frequencies in zip(self.documents, self.term_frequencies):
            vector = self._vector(frequencies, len(self.documents))
            score = sum(query_vector.get(term, 0.0) * vector.get(term, 0.0) for term in query_vector)
            if score > 0:
                results.append(SearchResult(document, score))
        return sorted(results, key=lambda result: result.score, reverse=True)[:limit]

    def _vector(self, frequencies: Counter[str], count: int) -> dict[str, float]:
        vector: dict[str, float] = {}
        for term, frequency in frequencies.items():
            idf = math.log((1 + count) / (1 + self.document_frequency[term])) + 1
            vector[term] = (1 + math.log(frequency)) * idf
        length = math.sqrt(sum(value * value for value in vector.values())) or 1
        return {term: value / length for term, value in vector.items()}
