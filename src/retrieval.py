"""Embedding index and cosine-similarity retrieval.

Embeddings are unit-norm, so dot product equals cosine similarity. FAISS is
used when installed and an index file exists; otherwise a NumPy matmul ranks
the whole set, which stays fast for a few thousand images.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from encode_images import EMBEDDINGS_NAME, FAISS_NAME, FILENAMES_NAME, load_labels

#: Above this many candidates the NumPy path is noticeably slower than FAISS.
FAISS_HINT_SIZE = 20_000


class RetrievalError(RuntimeError):
    """Raised when the embedding cache is missing or inconsistent."""


@dataclass(frozen=True)
class Hit:
    """One ranked retrieval result."""

    rank: int
    filename: str
    label: str
    score: float


def faiss_available() -> bool:
    try:
        import faiss  # type: ignore  # noqa: F401

        return True
    except ImportError:
        return False


class EmbeddingIndex:
    """Image embeddings plus filenames, searchable by text embedding."""

    def __init__(
        self,
        embeddings: np.ndarray,
        filenames: Sequence[str],
        labels: Sequence[str] | None = None,
        use_faiss: bool = False,
        faiss_path: Path | None = None,
    ) -> None:
        if embeddings.ndim != 2:
            raise RetrievalError(f"expected a 2-D embedding matrix, got shape {embeddings.shape}")
        if embeddings.shape[0] != len(filenames):
            raise RetrievalError(
                f"{embeddings.shape[0]} embeddings for {len(filenames)} filenames"
            )
        self.embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
        self.filenames = list(filenames)
        self.labels = list(labels) if labels is not None else [""] * len(self.filenames)
        if len(self.labels) != len(self.filenames):
            raise RetrievalError("labels and filenames have different lengths")
        self.index = None
        self.backend = "numpy"
        if use_faiss:
            self.index, self.backend = self._load_faiss(faiss_path)

    # ------------------------------------------------------------------- load
    @classmethod
    def load(cls, embeddings_dir: Path, use_faiss: bool = True) -> "EmbeddingIndex":
        embed_path = embeddings_dir / EMBEDDINGS_NAME
        if not embed_path.is_file():
            raise RetrievalError(
                f"no embeddings at {embed_path} - run 'python src/encode_images.py' first"
            )
        embeddings = np.load(embed_path)
        filenames = json.loads((embeddings_dir / FILENAMES_NAME).read_text(encoding="utf-8"))
        labels = load_labels(embeddings_dir)
        return cls(
            embeddings,
            filenames["filenames"],
            labels,
            use_faiss=use_faiss,
            faiss_path=embeddings_dir / FAISS_NAME,
        )

    def _load_faiss(self, faiss_path: Path | None):
        if faiss_path is None or not faiss_path.is_file():
            return None, "numpy" if not faiss_available() else "numpy (no index file)"
        try:
            import faiss  # type: ignore
        except ImportError:
            return None, "numpy (faiss not installed)"
        index = faiss.read_index(str(faiss_path))
        if index.ntotal != self.embeddings.shape[0]:
            return None, f"numpy (stale FAISS index: {index.ntotal} != {self.embeddings.shape[0]})"
        return index, "faiss"

    # -------------------------------------------------------------- properties
    @property
    def size(self) -> int:
        return len(self.filenames)

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])

    def norms(self) -> np.ndarray:
        return np.linalg.norm(self.embeddings, axis=1)

    def label_for(self, filename: str) -> str:
        try:
            return self.labels[self.filenames.index(filename)]
        except ValueError:
            return ""

    def restrict(self, filenames: Sequence[str]) -> "EmbeddingIndex":
        """A candidate-limited view (used so evaluation ranks only its own split)."""
        wanted = set(filenames)
        positions = [i for i, name in enumerate(self.filenames) if name in wanted]
        if not positions:
            raise RetrievalError("no overlapping filenames when restricting the index")
        return EmbeddingIndex(
            self.embeddings[positions],
            [self.filenames[i] for i in positions],
            [self.labels[i] for i in positions],
        )

    # ---------------------------------------------------------------- searching
    def search(self, query: np.ndarray, k: int = 5, backend: str | None = None) -> list[Hit]:
        """Rank images against one query vector."""
        vector = np.asarray(query, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.dim:
            raise RetrievalError(f"query has {vector.shape[0]} dims, index has {self.dim}")
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            raise RetrievalError("query embedding is all zeros")
        vector = vector / norm
        k = max(1, min(int(k), self.size))

        chosen = backend or ("faiss" if self.index is not None else "numpy")
        if chosen == "faiss" and self.index is not None:
            scores, indices = self.index.search(vector.reshape(1, -1), k)
            pairs = list(zip(indices[0].tolist(), scores[0].tolist()))
        else:
            scores = self.embeddings @ vector
            top = np.argpartition(-scores, k - 1)[:k]
            top = top[np.argsort(-scores[top])]
            pairs = [(int(i), float(scores[i])) for i in top]

        return [
            Hit(rank=rank, filename=self.filenames[i], label=self.labels[i], score=float(score))
            for rank, (i, score) in enumerate(pairs, start=1)
            if 0 <= i < self.size
        ]

    def search_text(self, query: str, embedder: object, k: int = 5) -> list[Hit]:
        """Convenience wrapper: encode a raw string, then search."""
        text = (query or "").strip()
        if not text:
            raise RetrievalError("empty query")
        vector = embedder.encode_texts([text])[0]  # type: ignore[attr-defined]
        return self.search(vector, k=k)

    def rank_of(self, filename: str, query: np.ndarray) -> int:
        """1-based rank of one filename for a query (used by evaluation)."""
        if filename not in self.filenames:
            raise RetrievalError(f"{filename} is not part of this index")
        vector = np.asarray(query, dtype=np.float32).reshape(-1)
        vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
        order = np.argsort(-(self.embeddings @ vector), kind="stable")
        target = self.filenames.index(filename)
        return int(np.where(order == target)[0][0]) + 1

    def add(self, embeddings: np.ndarray, filenames: Sequence[str], labels: Sequence[str]) -> None:
        """Append vectors in memory (used by the fine-tuning evaluator)."""
        if embeddings.shape[0] != len(filenames):
            raise RetrievalError("embeddings and filenames have different lengths")
        self.embeddings = np.ascontiguousarray(
            np.vstack([self.embeddings, embeddings.astype(np.float32)])
        )
        self.filenames.extend(filenames)
        self.labels.extend(labels)
        self.index = None


def recall_at_k(ranks: Sequence[int], k: int) -> float:
    """Fraction of queries whose relevant item sits in the top ``k``."""
    if not ranks:
        return 0.0
    return sum(1 for rank in ranks if rank <= k) / len(ranks)


def summarize_ranks(ranks: Sequence[int], ks: Sequence[int] = (1, 5, 10)) -> dict[str, float]:
    """Recall@k plus mean/median rank for one query set."""
    summary = {f"recall@{k}": round(recall_at_k(ranks, k), 4) for k in ks}
    if ranks:
        summary["mean_rank"] = round(float(np.mean(ranks)), 2)
        summary["median_rank"] = float(np.median(ranks))
        summary["mrr"] = round(float(np.mean([1.0 / rank for rank in ranks])), 4)
    else:
        summary.update({"mean_rank": 0.0, "median_rank": 0.0, "mrr": 0.0})
    return summary
