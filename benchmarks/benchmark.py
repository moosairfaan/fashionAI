"""Benchmark brute-force vs from-scratch HNSW.

Experiment A — latency vs dataset size
Experiment B — recall@10 vs latency (ef_search tradeoff)

Plots land in benchmarks/ for the README.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.prepare_dataset import embed_text
from index.brute_force import BruteForceIndex
from index.hnsw import HNSWIndex

DATA_DIR = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent

# Fashion-ish text queries — embedded once, reused across experiments.
SAMPLE_QUERIES = [
    "red summer dress",
    "navy blue blazer",
    "white sneakers",
    "black leather jacket",
    "floral midi skirt",
    "denim jeans",
    "striped t-shirt",
    "beige trench coat",
    "green hoodie",
    "brown ankle boots",
    "silk blouse",
    "grey wool sweater",
    "yellow raincoat",
    "pink handbag",
    "checkered shirt",
    "sports shorts",
    "formal black shoes",
    "casual polo",
    "winter scarf",
    "party heels",
]

SIZE_TARGETS = [1000, 5000, 10000]  # plus len(embeddings) appended at runtime
EF_VALUES = [10, 25, 50, 100, 200]
N_QUERIES = 20
K = 10
M = 16
EF_CONSTRUCTION = 100
NUM_LAYERS = 4


def recall_at_k(approx_ids: list[int], truth_ids: list[int]) -> float:
    """Fraction of ground-truth top-k that appear in the approximate top-k."""
    if not truth_ids:
        return 0.0
    return len(set(approx_ids) & set(truth_ids)) / len(truth_ids)


def build_indexes(
    embeddings: np.ndarray,
    metadata: list[dict],
) -> tuple[BruteForceIndex, HNSWIndex, float]:
    """Build brute-force + HNSW on a subsample; return (brute, hnsw, hnsw_build_s)."""
    brute = BruteForceIndex(embeddings, metadata)

    t0 = time.perf_counter()
    hnsw = HNSWIndex(
        dim=embeddings.shape[1],
        M=M,
        ef_construction=EF_CONSTRUCTION,
        num_layers=NUM_LAYERS,
    )
    for i, vec in enumerate(embeddings):
        hnsw.insert(vec, id=int(metadata[i]["id"]))
    build_s = time.perf_counter() - t0
    return brute, hnsw, build_s


def mean_query_latency_ms(
    search_fn,
    query_vectors: list[np.ndarray],
) -> float:
    """Average wall-clock latency (ms) over the query set."""
    latencies = []
    for q in query_vectors:
        t0 = time.perf_counter()
        search_fn(q)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(latencies))


def experiment_a_latency_vs_size(
    embeddings: np.ndarray,
    metadata: list[dict],
    query_vectors: list[np.ndarray],
) -> list[dict]:
    """Measure avg query latency for brute force and HNSW at several N."""
    n = len(embeddings)
    sizes = sorted({s for s in SIZE_TARGETS + [n] if s <= n})
    rows: list[dict] = []

    print("\n=== Experiment A: latency vs dataset size ===")
    for size in sizes:
        sub_emb = embeddings[:size]
        sub_meta = metadata[:size]
        print(f"\nBuilding indexes for N={size}…")
        brute, hnsw, build_s = build_indexes(sub_emb, sub_meta)
        print(f"  HNSW build: {build_s:.2f}s")

        brute_ms = mean_query_latency_ms(
            lambda q: brute.search(q, k=K),
            query_vectors,
        )
        hnsw_ms = mean_query_latency_ms(
            lambda q: hnsw.search(q, k=K, ef_search=50),
            query_vectors,
        )
        speedup = brute_ms / hnsw_ms if hnsw_ms > 0 else float("inf")
        row = {
            "dataset_size": size,
            "brute_force_latency_ms": brute_ms,
            "hnsw_latency_ms": hnsw_ms,
            "speedup": speedup,
            "hnsw_build_s": build_s,
        }
        rows.append(row)
        print(
            f"  brute={brute_ms:.3f} ms  hnsw={hnsw_ms:.3f} ms  "
            f"speedup={speedup:.2f}x"
        )

    return rows


def experiment_b_recall_vs_latency(
    embeddings: np.ndarray,
    metadata: list[dict],
    query_vectors: list[np.ndarray],
) -> list[dict]:
    """Sweep ef_search: recall@10 vs average HNSW latency."""
    print("\n=== Experiment B: recall@10 vs latency (ef_search) ===")
    print(f"Building indexes on full dataset N={len(embeddings)}…")
    brute, hnsw, build_s = build_indexes(embeddings, metadata)
    print(f"  HNSW build: {build_s:.2f}s")

    # Ground-truth top-10 for each query (exact).
    ground_truth: list[list[int]] = []
    for q in query_vectors:
        hits = brute.search(q, k=K)
        ground_truth.append([int(h["id"]) for h in hits])

    rows: list[dict] = []
    for ef in EF_VALUES:
        recalls = []
        latencies = []
        for q, truth in zip(query_vectors, ground_truth):
            t0 = time.perf_counter()
            hits = hnsw.search(q, k=K, ef_search=ef)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            approx_ids = [int(h["id"]) for h in hits]
            recalls.append(recall_at_k(approx_ids, truth))

        row = {
            "ef_search": ef,
            "recall_at_10": float(np.mean(recalls)),
            "latency_ms": float(np.mean(latencies)),
        }
        rows.append(row)
        print(
            f"  ef_search={ef:3d}  recall@10={row['recall_at_10']:.3f}  "
            f"latency={row['latency_ms']:.3f} ms"
        )

    return rows


def plot_latency_vs_size(rows: list[dict], path: Path) -> None:
    sizes = [r["dataset_size"] for r in rows]
    brute = [r["brute_force_latency_ms"] for r in rows]
    hnsw = [r["hnsw_latency_ms"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(sizes, brute, marker="o", label="Brute force", color="#1a3a32")
    ax.plot(sizes, hnsw, marker="s", label="HNSW (ef=50)", color="#0b6e4f")
    ax.set_yscale("log")
    ax.set_xlabel("Dataset size (N)")
    ax.set_ylabel("Avg query latency (ms, log scale)")
    ax.set_title("Latency vs dataset size")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_recall_vs_latency(rows: list[dict], path: Path) -> None:
    latencies = [r["latency_ms"] for r in rows]
    recalls = [r["recall_at_10"] for r in rows]
    efs = [r["ef_search"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(latencies, recalls, marker="o", color="#0b6e4f")
    for ef, x, y in zip(efs, latencies, recalls):
        ax.annotate(
            f"ef={ef}",
            (x, y),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
        )
    ax.set_xlabel("Avg HNSW query latency (ms)")
    ax.set_ylabel("Recall@10")
    ax.set_ylim(0, 1.05)
    ax.set_title("Recall@10 vs latency (ef_search tradeoff)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def print_summary(size_rows: list[dict], recall_rows: list[dict]) -> None:
    """Print final summary: size, speedup, recall@10 at ef_search=50."""
    recall_at_50 = next(
        (r["recall_at_10"] for r in recall_rows if r["ef_search"] == 50),
        float("nan"),
    )

    print("\n=== Summary ===")
    print(f"{'N':>8}  {'brute ms':>10}  {'hnsw ms':>10}  {'speedup':>8}  {'R@10 (ef=50)':>14}")
    print("-" * 58)
    for r in size_rows:
        # Recall is measured on the full dataset only — show it on the last row.
        recall_cell = f"{recall_at_50:.3f}" if r is size_rows[-1] else "—"
        print(
            f"{r['dataset_size']:8d}  "
            f"{r['brute_force_latency_ms']:10.3f}  "
            f"{r['hnsw_latency_ms']:10.3f}  "
            f"{r['speedup']:7.2f}x  "
            f"{recall_cell:>14}"
        )
    print(
        f"\nRecall@10 at ef_search=50 (full dataset): {recall_at_50:.3f}"
    )


def main() -> None:
    emb_path = DATA_DIR / "embeddings.npy"
    meta_path = DATA_DIR / "metadata.json"
    if not emb_path.exists() or not meta_path.exists():
        raise SystemExit(
            f"Missing {emb_path.name} or {meta_path.name} in {DATA_DIR}. "
            "Run: python data/prepare_dataset.py"
        )

    embeddings = np.load(emb_path).astype(np.float32)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if len(metadata) != len(embeddings):
        raise SystemExit(
            f"metadata ({len(metadata)}) / embeddings ({len(embeddings)}) mismatch"
        )

    print(f"Loaded {len(embeddings)} embeddings (dim={embeddings.shape[1]})")

    # Embed a fixed set of text queries once (CLIP is the expensive part).
    texts = SAMPLE_QUERIES[:N_QUERIES]
    print(f"Embedding {len(texts)} text queries…")
    query_vectors = [embed_text(t) for t in texts]

    size_rows = experiment_a_latency_vs_size(embeddings, metadata, query_vectors)
    recall_rows = experiment_b_recall_vs_latency(embeddings, metadata, query_vectors)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_latency_vs_size(size_rows, OUT_DIR / "latency_vs_size.png")
    plot_recall_vs_latency(recall_rows, OUT_DIR / "recall_vs_latency.png")

    metrics = {
        "n_embeddings": len(embeddings),
        "n_queries": len(query_vectors),
        "latency_vs_size": size_rows,
        "recall_vs_latency": recall_rows,
    }
    metrics_path = OUT_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {metrics_path}")

    print_summary(size_rows, recall_rows)


if __name__ == "__main__":
    main()
