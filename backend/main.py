"""FastAPI backend for fashion text search (CLIP + brute force / HNSW)."""

from __future__ import annotations

import json
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.prepare_dataset import DATA_DIR, _load_clip, embed_text  # noqa: E402
from index.brute_force import BruteForceIndex  # noqa: E402
from index.hnsw import HNSWIndex  # noqa: E402

EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
METADATA_PATH = DATA_DIR / "metadata.json"
IMAGES_DIR = DATA_DIR / "images"

Method = Literal["hnsw", "brute_force", "both"]

state: dict[str, Any] = {
    "embeddings": None,
    "metadata": [],
    "by_id": {},
    "brute": None,
    "hnsw": None,
}


def _enrich_hit(hit: dict) -> dict:
    """Merge a search hit {id, score} with metadata + a static image URL."""
    row = state["by_id"].get(int(hit["id"]), {})
    filename = str(row.get("filename") or "")
    # metadata path is relative to data/, e.g. "images/images/10000.jpg"
    # Static mount is data/images/ at /images → drop the leading "images/".
    rel = Path(str(row.get("path") or filename))
    parts = rel.parts
    if parts and parts[0] == "images":
        static_rel = Path(*parts[1:]).as_posix() if len(parts) > 1 else filename
    else:
        static_rel = rel.as_posix() if rel.parts else filename

    return {
        "id": int(hit["id"]),
        "score": float(hit["score"]),
        "filename": filename,
        "label": str(row.get("label") or filename),
        "path": str(row.get("path") or ""),
        "image_url": f"/images/{static_rel}",
        "master_category": str(row.get("master_category") or ""),
        "subcategory": str(row.get("subcategory") or ""),
        "color": str(row.get("color") or ""),
    }


def _timed_search(method: Literal["hnsw", "brute_force"], query_vec: np.ndarray, k: int):
    t0 = time.perf_counter()
    if method == "brute_force":
        raw = state["brute"].search(query_vec, k=k)
    else:
        raw = state["hnsw"].search(query_vec, k=k, ef_search=50)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return [_enrich_hit(h) for h in raw], latency_ms


def load_everything() -> None:
    """Load embeddings, build indexes, and warm the CLIP model."""
    if not EMBEDDINGS_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {EMBEDDINGS_PATH.name} or {METADATA_PATH.name}. "
            "Run: python data/prepare_dataset.py"
        )

    print("Loading embeddings + metadata…")
    embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if len(metadata) != len(embeddings):
        raise RuntimeError(
            f"metadata ({len(metadata)}) != embeddings ({len(embeddings)})"
        )

    print(f"Building BruteForceIndex (N={len(embeddings)})…")
    brute = BruteForceIndex(embeddings, metadata)

    print("Building HNSWIndex…")
    t0 = time.perf_counter()
    hnsw = HNSWIndex(dim=embeddings.shape[1], M=16, ef_construction=100, num_layers=4)
    for i, vec in enumerate(embeddings):
        hnsw.insert(vec, id=int(metadata[i]["id"]))
    print(f"HNSW ready in {time.perf_counter() - t0:.2f}s")

    print("Loading CLIP model (once)…")
    _load_clip()  # warm the lru_cache used by embed_text
    _ = embed_text("warmup")  # also exercise the forward path once

    state["embeddings"] = embeddings
    state["metadata"] = metadata
    state["by_id"] = {int(row["id"]): row for row in metadata}
    state["brute"] = brute
    state["hnsw"] = hnsw
    print("Startup complete.")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_everything()
    yield


app = FastAPI(title="fashionAI", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve product photos: data/images/... → /images/...
if IMAGES_DIR.exists():
    app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    k: int = Field(default=10, ge=1, le=50)
    method: Method = "hnsw"


class SearchResponse(BaseModel):
    results: dict[str, list[dict]]
    latency_ms: dict[str, float]
    query: str
    k: int


@app.get("/health")
def health() -> dict:
    emb = state["embeddings"]
    return {
        "status": "ok" if emb is not None else "empty",
        "n_items": 0 if emb is None else int(emb.shape[0]),
        "dim": None if emb is None else int(emb.shape[1]),
        "hnsw_ready": state["hnsw"] is not None,
        "brute_ready": state["brute"] is not None,
    }


@app.post("/search", response_model=SearchResponse)
def search(body: SearchRequest) -> SearchResponse:
    """Text search with HNSW, brute force, or both (side-by-side)."""
    if state["brute"] is None or state["hnsw"] is None:
        raise HTTPException(status_code=503, detail="Indexes not loaded yet")

    query_vec = embed_text(body.query)

    results: dict[str, list[dict]] = {}
    latency_ms: dict[str, float] = {}

    methods: list[Literal["hnsw", "brute_force"]]
    if body.method == "both":
        methods = ["hnsw", "brute_force"]
    else:
        methods = [body.method]

    for method in methods:
        hits, ms = _timed_search(method, query_vec, body.k)
        results[method] = hits
        latency_ms[method] = round(ms, 3)

    return SearchResponse(
        results=results,
        latency_ms=latency_ms,
        query=body.query,
        k=body.k,
    )
