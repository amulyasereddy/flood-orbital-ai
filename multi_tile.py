"""
multi_tile.py
=============
Process multiple SAR tiles and rank by flood confidence.
Demonstrates the orbital triage concept across a batch.

Usage:
    python multi_tile.py --tiles_dir data/test_tiles/
    python multi_tile.py  # uses sample.tif split into 4 quadrants
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import rasterio
from rasterio.enums import Resampling
from pathlib import Path

from flood_model import FloodDetectionModel
from preprocess import normalise_terramind, compute_flood_area_km2


def split_into_tiles(image: np.ndarray, n_tiles: int = 4) -> list:
    """Split a (C, H, W) image into n_tiles equal quadrant tiles."""
    _, h, w = image.shape
    tiles = []
    if n_tiles == 4:
        positions = [
            (0,    h//2, 0,    w//2),  # top-left
            (0,    h//2, w//2, w),     # top-right
            (h//2, h,    0,    w//2),  # bottom-left
            (h//2, h,    w//2, w),     # bottom-right
        ]
        for r0, r1, c0, c1 in positions:
            tiles.append(image[:, r0:r1, c0:c1])
    return tiles


def score_tile(model, tile: np.ndarray, device) -> dict:
    """Run TerraMind inference on a single tile. Returns confidence and flood area."""
    tile = normalise_terramind(tile)
    tensor = torch.from_numpy(tile).float().unsqueeze(0)
    tensor = F.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)
    tensor = tensor.to(device)

    with torch.no_grad():
        out = model(tensor)

    flood_mask = out["flood_mask"].squeeze().cpu().numpy()
    return {
        "confidence": out["confidence"],
        "flood_area_km2": compute_flood_area_km2(flood_mask),
    }


def run(tiles_dir: str = None,
        sample_path: str = "sample.tif",
        checkpoint: str = "models/flood_head_best.pt",
        top_k: int = 2):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FloodDetectionModel(pretrained_encoder=True).to(device)
    if Path(checkpoint).exists():
        model.load_head(checkpoint, str(device))
    model.eval()

    tiles_data = []

    # Option A: real tile directory
    if tiles_dir and Path(tiles_dir).exists():
        for path in sorted(Path(tiles_dir).glob("*.tif")):
            with rasterio.open(path) as src:
                img = src.read(out_shape=(src.count, 256, 256),
                               resampling=Resampling.bilinear).astype(np.float32)
            if img.shape[0] == 1: img = np.concatenate([img, img], axis=0)
            elif img.shape[0] >= 2: img = img[:2]
            score = score_tile(model, img, device)
            tiles_data.append({"name": path.name, **score})

    # Option B: split sample.tif into quadrants
    elif Path(sample_path).exists():
        print(f"📂 Splitting {sample_path} into 4 quadrant tiles...")
        with rasterio.open(sample_path) as src:
            img = src.read(out_shape=(src.count, 512, 512),
                           resampling=Resampling.bilinear).astype(np.float32)
        if img.shape[0] == 1: img = np.concatenate([img, img], axis=0)
        elif img.shape[0] >= 2: img = img[:2]

        for i, tile in enumerate(split_into_tiles(img, n_tiles=4)):
            score = score_tile(model, tile, device)
            tiles_data.append({"name": f"quadrant_{i}", **score})
    else:
        print("⚠️  No tiles found — using synthetic data")
        for i in range(4):
            dummy = np.random.randn(2, 256, 256).astype(np.float32)
            score = score_tile(model, dummy, device)
            tiles_data.append({"name": f"synthetic_tile_{i}", **score})

    # ── Rank and report ──
    tiles_sorted = sorted(tiles_data, key=lambda x: x["confidence"], reverse=True)

    print(f"\n🛰  TILE PRIORITY RANKING ({len(tiles_sorted)} tiles)")
    print("─" * 50)
    for rank, t in enumerate(tiles_sorted, 1):
        flag = "🚀 SEND" if rank <= top_k else "❌ SKIP"
        print(f"  #{rank}  {t['name']:<20} "
              f"conf={t['confidence']:.4f}  "
              f"area={t['flood_area_km2']:.3f}km²  {flag}")
    print("─" * 50)

    sent  = tiles_sorted[:top_k]
    total = len(tiles_sorted)
    bw    = (1 - top_k / total) * 100
    print(f"\n  Downlinked {top_k}/{total} tiles ({top_k/total*100:.0f}%)")
    print(f"  Bandwidth saved: {bw:.0f}%")

    return tiles_sorted


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles_dir",  default=None,                        type=str)
    parser.add_argument("--sample",     default="sample.tif",                type=str)
    parser.add_argument("--checkpoint", default="models/flood_head_best.pt", type=str)
    parser.add_argument("--top_k",      default=2, type=int)
    args = parser.parse_args()

    run(args.tiles_dir, args.sample, args.checkpoint, args.top_k)