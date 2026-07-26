"""Exact nearest-neighbor search baseline (cosine similarity).

Ground-truth for recalling HNSW later — prioritize correctness.
Embeddings are assumed L2-normalized so cosine == dot product.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


class BruteForceIndex:
    def __init__(self, embeddings: np.ndarray, metadata: list[dict]) -> None:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be 2-D (N, 512)")
        if len(metadata) != embeddings.shape[0]:
            raise ValueError(
                f"metadata length ({len(metadata)}) != N ({embeddings.shape[0]})"
            )
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.metadata = metadata

    def search(self, query_vector: np.ndarray, k: int = 10) -> list[dict]:
        """Return top-k metadata dicts (with score) by cosine similarity."""
        q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        # Cosine similarity = dot product when both sides are L2-normalized
        scores = self.embeddings @ q

        n = scores.shape[0]
        k = min(k, n)
        if k <= 0:
            return []

        # O(N) select top-k, then sort only those k
        if k < n:
            top_idx = np.argpartition(-scores, k - 1)[:k]
        else:
            top_idx = np.arange(n)
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        results: list[dict] = []
        for i in top_idx:
            item = dict(self.metadata[int(i)])
            item["score"] = float(scores[int(i)])
            results.append(item)
        return results


def main() -> None:
    emb_path = DATA_DIR / "embeddings.npy"
    meta_path = DATA_DIR / "metadata.json"
    if not emb_path.exists() or not meta_path.exists():
        raise SystemExit(
            f"Missing {emb_path.name} or {meta_path.name} in {DATA_DIR}. "
            "Run: python data/prepare_dataset.py"
        )

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from data.prepare_dataset import embed_text

    embeddings = np.load(emb_path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    index = BruteForceIndex(embeddings, metadata)

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Query: ").strip()
    if not query:
        raise SystemExit("Empty query")

    query_vec = embed_text(query)
    results = index.search(query_vec, k=5)

    print(f'\nTop-5 for "{query}":\n')
    for rank, item in enumerate(results, start=1):
        name = item.get("filename") or item.get("label") or item.get("path")
        print(f"  {rank}. {name}  score={item['score']:.4f}")


if __name__ == "__main__":
    main()
