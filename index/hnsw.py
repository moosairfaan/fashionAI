"""Simplified Hierarchical Navigable Small World (HNSW) ANN index.

Portfolio-oriented from-scratch implementation of the core ideas in
Malkov & Yashunin (2018). Readable over maximally optimized — no FAISS /
hnswlib.

Key idea
--------
Build a multi-layer proximity graph. Upper layers are sparse "highways"
for fast coarse navigation; layer 0 is dense and holds every point for
fine-grained search. A query greedily walks toward the query vector on
each layer, then drops down, finally returning the top-k at layer 0.

Distance
--------
Cosine similarity via dot product. Embeddings are assumed L2-normalized
so ``sim(a, b) = a · b`` (higher is closer). Internally we maximize
similarity rather than minimize a distance.
"""

from __future__ import annotations

import json
import math
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


class HNSWIndex:
    """Approximate nearest-neighbor index over L2-normalized vectors."""

    def __init__(
        self,
        dim: int,
        M: int = 16,
        ef_construction: int = 200,
        num_layers: int = 4,
        seed: int = 42,
    ) -> None:
        """Create an empty HNSW graph.

        Parameters
        ----------
        dim:
            Embedding dimensionality (512 for CLIP ViT-B/32).
        M:
            Max neighbors retained per node on each layer. Also sets the
            layer-sampling scale ``mL = 1 / ln(M)``.
        ef_construction:
            Width of the candidate list while inserting (higher → better
            graph quality, slower build).
        num_layers:
            Fixed hierarchy height. Layer ``num_layers - 1`` is the top;
            layer ``0`` is the bottom and contains every inserted node.
        seed:
            RNG seed for reproducible layer assignment.
        """
        if M < 2:
            raise ValueError("M must be >= 2")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.dim = dim
        self.M = M
        self.ef_construction = ef_construction
        self.num_layers = num_layers
        # Standard HNSW multiplier: higher layers are exponentially rarer.
        self.mL = 1.0 / math.log(M)
        self._rng = random.Random(seed)

        # node_id -> L2-normalized float32 vector
        self.vectors: dict[int, np.ndarray] = {}
        # node_id -> highest layer index this node appears on (inclusive)
        self.node_level: dict[int, int] = {}
        # graph[layer][node_id] = list of neighbor ids at that layer
        self.graph: dict[int, dict[int, list[int]]] = {
            layer: {} for layer in range(num_layers)
        }

        # A node that lives on the current highest occupied layer.
        self.entry_point: int | None = None
        self.max_level: int = -1

    # ------------------------------------------------------------------
    # Similarity helpers
    # ------------------------------------------------------------------

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        v = np.asarray(vector, dtype=np.float32).reshape(-1)
        if v.shape[0] != self.dim:
            raise ValueError(f"expected dim={self.dim}, got {v.shape[0]}")
        norm = float(np.linalg.norm(v))
        if norm < 1e-12:
            return v
        return v / norm

    def _sim(self, query: np.ndarray, node_id: int) -> float:
        """Cosine similarity = dot product for normalized vectors."""
        return float(np.dot(query, self.vectors[node_id]))

    def _assign_level(self) -> int:
        """Sample a node level: ``floor(-ln(U) * mL)``, capped at top layer.

        Most nodes land on layer 0; each higher layer is roughly ``1/M``
        as likely as the one below it.
        """
        u = self._rng.random()
        level = int(math.floor(-math.log(max(u, 1e-12)) * self.mL))
        return min(level, self.num_layers - 1)

    # ------------------------------------------------------------------
    # Core graph search primitives
    # ------------------------------------------------------------------

    def _greedy_search_layer(
        self,
        query: np.ndarray,
        entry_id: int,
        layer: int,
    ) -> int:
        """Walk to a local similarity maximum on one layer.

        From ``entry_id``, repeatedly move to the neighbor with the highest
        cosine similarity to ``query`` until no neighbor is closer. This is
        the fast "highway" navigation used on upper layers.
        """
        current = entry_id
        best_sim = self._sim(query, current)

        while True:
            neighbors = self.graph[layer].get(current, [])
            moved = False
            for nb in neighbors:
                s = self._sim(query, nb)
                if s > best_sim:
                    best_sim = s
                    current = nb
                    moved = True
            if not moved:
                return current

    def _search_layer(
        self,
        query: np.ndarray,
        entry_points: list[int],
        layer: int,
        ef: int,
    ) -> list[tuple[float, int]]:
        """Beam search on one layer; return up to ``ef`` best (sim, id).

        Maintains:
        - ``candidates``: nodes still to expand (best-first by similarity)
        - ``results``: the current ef-nearest found so far

        Expansion stops when the best unexpanded candidate is worse than
        the worst node already in ``results`` (and ``results`` is full).
        """
        visited: set[int] = set()
        # candidates: list of (similarity, id), kept sorted descending
        candidates: list[tuple[float, int]] = []
        # results: list of (similarity, id), kept sorted ascending so
        # results[0] is the worst (lowest similarity) in the beam
        results: list[tuple[float, int]] = []

        for ep in entry_points:
            if ep in visited:
                continue
            s = self._sim(query, ep)
            visited.add(ep)
            candidates.append((s, ep))
            results.append((s, ep))

        candidates.sort(key=lambda x: x[0], reverse=True)
        results.sort(key=lambda x: x[0])

        while candidates:
            sim_c, c = candidates[0]
            worst_sim = results[0][0] if results else float("-inf")
            # Best candidate cannot beat the worst result → done.
            if len(results) >= ef and sim_c < worst_sim:
                break
            candidates.pop(0)

            for nb in self.graph[layer].get(c, []):
                if nb in visited:
                    continue
                visited.add(nb)
                s = self._sim(query, nb)
                worst_sim = results[0][0] if results else float("-inf")
                if s > worst_sim or len(results) < ef:
                    candidates.append((s, nb))
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    results.append((s, nb))
                    results.sort(key=lambda x: x[0])
                    if len(results) > ef:
                        results.pop(0)  # drop worst

        # Return best-first (highest similarity first).
        out = sorted(results, key=lambda x: x[0], reverse=True)
        return out

    def _select_neighbors(
        self,
        candidates: list[tuple[float, int]],
        M: int,
    ) -> list[int]:
        """Keep the ``M`` highest-similarity candidates (simple heuristic)."""
        return [node_id for _, node_id in candidates[:M]]

    def _connect(self, a: int, b: int, layer: int) -> None:
        """Add a bidirectional edge on ``layer``, then prune to ≤ M edges."""
        if a == b:
            return
        for src, dst in ((a, b), (b, a)):
            nbrs = self.graph[layer].setdefault(src, [])
            if dst not in nbrs:
                nbrs.append(dst)
            if len(nbrs) > self.M:
                self._prune_neighbors(src, layer)

    def _prune_neighbors(self, node_id: int, layer: int) -> None:
        """Shrink a node's neighbor list to the M most similar nodes."""
        nbrs = self.graph[layer].get(node_id, [])
        if len(nbrs) <= self.M:
            return
        vec = self.vectors[node_id]
        scored = sorted(
            ((float(np.dot(vec, self.vectors[n])), n) for n in nbrs),
            key=lambda x: x[0],
            reverse=True,
        )
        keep = {n for _, n in scored[: self.M]}
        self.graph[layer][node_id] = [n for n in nbrs if n in keep]
        # Drop reverse edges that are no longer reciprocated from our side
        # only when we removed them — keep the graph consistent.
        for _, n in scored[self.M :]:
            rev = self.graph[layer].get(n, [])
            if node_id in rev:
                rev.remove(node_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def insert(self, vector: np.ndarray, id: int) -> None:
        """Insert ``vector`` under ``id`` into the multi-layer graph.

        Steps
        -----
        1. L2-normalize the vector and sample a random maximum layer.
        2. If the index is empty, this node becomes the entry point.
        3. Otherwise, greedily descend from the top layer down to
           ``level + 1`` to get a good entry into the insertion layers.
        4. For each layer from ``level`` down to ``0``: run an
           ``ef_construction``-wide search, connect to the top-M hits,
           and prune any neighbor lists that exceed M.
        """
        if id in self.vectors:
            raise ValueError(f"id {id} already inserted")

        vec = self._normalize(vector)
        level = self._assign_level()
        self.vectors[id] = vec
        self.node_level[id] = level

        # Ensure adjacency-list entries exist on every layer this node owns.
        for layer in range(level + 1):
            self.graph[layer].setdefault(id, [])

        # First point: nothing to connect to yet.
        if self.entry_point is None:
            self.entry_point = id
            self.max_level = level
            return

        # Phase 1 — greedy descent on layers above the new node's level.
        ep = self.entry_point
        for layer in range(self.max_level, level, -1):
            ep = self._greedy_search_layer(vec, ep, layer)

        # Phase 2 — insert into layers [level .. 0].
        entry_points = [ep]
        for layer in range(min(level, self.max_level), -1, -1):
            candidates = self._search_layer(
                vec, entry_points, layer, self.ef_construction
            )
            neighbors = self._select_neighbors(candidates, self.M)
            for nb in neighbors:
                self._connect(id, nb, layer)
            # Next lower layer starts from the best candidates found here.
            entry_points = [node_id for _, node_id in candidates[: max(1, self.M)]]

        if level > self.max_level:
            self.max_level = level
            self.entry_point = id

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        ef_search: int = 50,
    ) -> list[dict]:
        """Approximate top-k search by cosine similarity.

        Steps
        -----
        1. Normalize the query.
        2. From the entry point, greedily walk each upper layer until a
           local maximum, then drop one layer.
        3. On layer 0, expand a candidate beam of size ``ef_search``.
        4. Return the ``k`` highest-similarity hits as
           ``{"id": …, "score": …}`` dicts (score = cosine similarity).
        """
        if self.entry_point is None or not self.vectors:
            return []

        query = self._normalize(query_vector)
        ef = max(ef_search, k)
        ep = self.entry_point

        # Upper layers: cheap greedy zoom-in toward the query.
        for layer in range(self.max_level, 0, -1):
            ep = self._greedy_search_layer(query, ep, layer)

        # Bottom layer: wider beam search for recall.
        candidates = self._search_layer(query, [ep], layer=0, ef=ef)
        top = candidates[:k]
        return [{"id": int(node_id), "score": float(sim)} for sim, node_id in top]

    # Optional persistence for the FastAPI backend -------------------------

    def save(self, path: str | Path) -> None:
        """Serialize the index to a pickle file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dim": self.dim,
            "M": self.M,
            "ef_construction": self.ef_construction,
            "num_layers": self.num_layers,
            "mL": self.mL,
            "vectors": self.vectors,
            "node_level": self.node_level,
            "graph": self.graph,
            "entry_point": self.entry_point,
            "max_level": self.max_level,
        }
        with path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "HNSWIndex":
        """Load an index previously written by :meth:`save`."""
        with Path(path).open("rb") as f:
            payload = pickle.load(f)
        idx = cls(
            dim=payload["dim"],
            M=payload["M"],
            ef_construction=payload["ef_construction"],
            num_layers=payload["num_layers"],
        )
        idx.mL = payload["mL"]
        idx.vectors = payload["vectors"]
        idx.node_level = payload["node_level"]
        idx.graph = payload["graph"]
        idx.entry_point = payload["entry_point"]
        idx.max_level = payload["max_level"]
        return idx


def main() -> None:
    emb_path = DATA_DIR / "embeddings.npy"
    meta_path = DATA_DIR / "metadata.json"
    if not emb_path.exists() or not meta_path.exists():
        raise SystemExit(
            f"Missing embeddings/metadata in {DATA_DIR}. "
            "Run: python data/prepare_dataset.py"
        )

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from data.prepare_dataset import embed_text

    embeddings = np.load(emb_path).astype(np.float32)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    by_id = {int(row["id"]): row for row in metadata}

    print(f"Building HNSW over {len(embeddings)} vectors (dim={embeddings.shape[1]})…")
    index = HNSWIndex(dim=embeddings.shape[1], M=16, ef_construction=200, num_layers=4)

    t0 = time.perf_counter()
    for i, vec in enumerate(embeddings):
        index.insert(vec, id=int(metadata[i]["id"]))
    build_s = time.perf_counter() - t0
    print(f"Index construction: {build_s:.2f}s")

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Query: ").strip() or "red summer dress"

    query_vec = embed_text(query)
    results = index.search(query_vec, k=5, ef_search=50)

    print(f'\nTop-5 HNSW results for "{query}":\n')
    for rank, hit in enumerate(results, start=1):
        row = by_id.get(hit["id"], {})
        name = row.get("filename") or row.get("label") or hit["id"]
        print(f"  {rank}. {name}  score={hit['score']:.4f}")


if __name__ == "__main__":
    main()
