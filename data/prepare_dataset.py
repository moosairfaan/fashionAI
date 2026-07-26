"""Embed a local folder of fashion product images with CLIP.

Expects:
  data/images/           — product photos, or a nested images/ (Kaggle Myntra layout)
  data/images/styles.csv — optional metadata (id, productDisplayName, …)

Writes:
  data/embeddings.npy  — float32 array of shape (N, 512)
  data/metadata.json   — parallel list of image filenames / metadata
"""

from __future__ import annotations

import argparse
import csv
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

DATA_DIR = Path(__file__).resolve().parent
IMAGES_DIR = DATA_DIR / "images"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
METADATA_PATH = DATA_DIR / "metadata.json"

MODEL_NAME = "openai/clip-vit-base-patch32"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
BATCH_SIZE = 32


@lru_cache(maxsize=1)
def _load_clip(device: str | None = None) -> tuple[CLIPProcessor, CLIPModel, str]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME)
    model.eval()
    model.to(device)
    return processor, model, device


def list_image_paths(images_dir: Path = IMAGES_DIR) -> list[Path]:
    """Collect image files under images_dir (handles nested Kaggle layout)."""
    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"Images directory not found: {images_dir}\n"
            f"Put fashion product images in {images_dir} and re-run."
        )

    def _is_image(p: Path) -> bool:
        return p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS

    # 1) Flat: data/images/*.jpg
    direct = sorted(p for p in images_dir.iterdir() if _is_image(p))
    if direct:
        return direct

    # 2) Kaggle fashion-product-images: data/images/images/*.jpg
    nested = images_dir / "images"
    if nested.is_dir():
        nested_paths = sorted(p for p in nested.iterdir() if _is_image(p))
        if nested_paths:
            return nested_paths

    # 3) Fallback: recursive, dedupe by filename (skip myntradataset copies)
    by_name: dict[str, Path] = {}
    for p in sorted(images_dir.rglob("*")):
        if not _is_image(p):
            continue
        # Prefer shorter paths when duplicates exist
        prev = by_name.get(p.name)
        if prev is None or len(p.parts) < len(prev.parts):
            by_name[p.name] = p
    paths = sorted(by_name.values(), key=lambda p: p.name)
    if not paths:
        raise FileNotFoundError(f"No images found in {images_dir}")
    return paths


def load_styles(images_dir: Path) -> dict[str, dict]:
    """Map filename stem → styles.csv row (e.g. '10000' → {...})."""
    candidates = [
        images_dir / "styles.csv",
        images_dir / "myntradataset" / "styles.csv",
        images_dir.parent / "styles.csv",
    ]
    styles_path = next((p for p in candidates if p.is_file()), None)
    if styles_path is None:
        return {}

    by_id: dict[str, dict] = {}
    with styles_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = str(row.get("id", "")).strip()
            if key:
                by_id[key] = row
    return by_id


def embed_images(
    image_paths: list[Path],
    batch_size: int = BATCH_SIZE,
    device: str | None = None,
) -> np.ndarray:
    """Embed images with CLIP → (N, 512) L2-normalized float32."""
    processor, model, device = _load_clip(device)
    vectors: list[np.ndarray] = []

    for start in tqdm(range(0, len(image_paths), batch_size), desc="Embedding images"):
        batch_paths = image_paths[start : start + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        vectors.append(feats.detach().cpu().numpy().astype(np.float32))
        for im in images:
            im.close()

    return np.vstack(vectors)


def embed_text(query: str, device: str | None = None) -> np.ndarray:
    """Embed a text query with the same CLIP model → (512,) float32."""
    processor, model, device = _load_clip(device)
    inputs = processor(text=[query], return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.detach().cpu().numpy().astype(np.float32)[0]


def prepare_dataset(
    data_dir: Path = DATA_DIR,
    batch_size: int = BATCH_SIZE,
    device: str | None = None,
    max_samples: int | None = None,
) -> tuple[np.ndarray, list[dict]]:
    data_dir = Path(data_dir).resolve()
    images_dir = data_dir / "images"
    image_paths = list_image_paths(images_dir)

    if max_samples is not None and max_samples > 0:
        image_paths = image_paths[:max_samples]

    styles = load_styles(images_dir)
    print(f"Embedding {len(image_paths)} images from {images_dir}…")

    embeddings = embed_images(image_paths, batch_size=batch_size, device=device)

    metadata: list[dict] = []
    for i, p in enumerate(image_paths):
        row: dict = {
            "id": i,
            "filename": p.name,
            "path": str(p.relative_to(data_dir)),
        }
        style = styles.get(p.stem)
        if style:
            row["label"] = style.get("productDisplayName") or p.name
            row["master_category"] = style.get("masterCategory") or ""
            row["subcategory"] = style.get("subCategory") or ""
            row["article_type"] = style.get("articleType") or ""
            row["color"] = style.get("baseColour") or ""
            row["gender"] = style.get("gender") or ""
            row["style_id"] = style.get("id") or p.stem
        else:
            row["label"] = p.name
        metadata.append(row)

    emb_path = data_dir / "embeddings.npy"
    meta_path = data_dir / "metadata.json"
    np.save(emb_path, embeddings)
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote {emb_path} shape={embeddings.shape}")
    print(f"Wrote {meta_path} ({len(metadata)} items)")
    return embeddings, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed local fashion images with CLIP")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing an images/ subdirectory (default: data/)",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", default=None, help="cpu | cuda (default: auto)")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2000,
        help="Cap how many images to embed (default: 2000; use 0 for all)",
    )
    args = parser.parse_args()

    max_samples = None if args.max_samples == 0 else args.max_samples
    prepare_dataset(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        device=args.device,
        max_samples=max_samples,
    )


if __name__ == "__main__":
    main()
