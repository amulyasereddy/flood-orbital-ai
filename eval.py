"""
eval.py
=======
Evaluate TerraMind flood model vs OTSU baseline on Sen1Floods11 test split.
This file produces the headline numbers for your presentation slide 4.

Usage:
    python eval.py --data_dir data/sen1floods11 --checkpoint models/flood_head_best.pt

Expected output (example):
    ┌─────────────────────────────────────────────┐
    │  Baseline (OTSU threshold on SAR)            │
    │    mIoU   : 0.312                            │
    │    F1     : 0.447                            │
    │                                              │
    │  TerraMind-small (ours)                      │
    │    mIoU   : 0.681   (+118% vs baseline)      │
    │    F1     : 0.789   (+76%  vs baseline)      │
    │    Prec   : 0.812                            │
    │    Recall : 0.768                            │
    └─────────────────────────────────────────────┘
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

from flood_model import FloodDetectionModel
from train import Sen1Floods11Dataset


# ─────────────────────────────────────────────
# Baseline: OTSU threshold on raw SAR
# ─────────────────────────────────────────────
def otsu_threshold(image: np.ndarray) -> np.ndarray:
    """
    Simple OTSU-style threshold on SAR channel 0 (VV).
    Returns binary mask: 1=flood, 0=no flood.
    Low VV backscatter = calm water = flood.
    """
    vv = image[0]
    threshold = np.percentile(vv, 20)   # bottom 20% backscatter = water
    return (vv < threshold).astype(np.float32)


def run_baseline(data_dir: str) -> dict:
    """Run OTSU baseline on test split. Returns metric dict."""
    import rasterio
    from rasterio.enums import Resampling
    from pathlib import Path

    ds = Sen1Floods11Dataset(data_dir, split="test")
    all_preds, all_labels = [], []

    print("📊 Running OTSU baseline...")
    for idx in tqdm(range(len(ds))):
        with rasterio.open(ds.s1_files[idx]) as src:
            raw = src.read(
                out_shape=(src.count, 224, 224),
                resampling=Resampling.bilinear,
            ).astype(np.float32)

        if raw.shape[0] == 1:
            raw = np.concatenate([raw, raw], axis=0)
        elif raw.shape[0] >= 2:
            raw = raw[:2]

        pred = otsu_threshold(raw)

        with rasterio.open(ds.label_files[idx]) as src:
            label = src.read(
                1,
                out_shape=(224, 224),
                resampling=Resampling.nearest,
            ).astype(np.float32)

        label = np.clip(label, 0, 1)
        all_preds.append(pred.flatten())
        all_labels.append(label.flatten())

    preds  = np.concatenate(all_preds).astype(int)
    labels = np.concatenate(all_labels).astype(int)

    return {
        "mIoU":      _miou(preds, labels),
        "F1":        f1_score(labels, preds, zero_division=0),
        "Precision": precision_score(labels, preds, zero_division=0),
        "Recall":    recall_score(labels, preds, zero_division=0),
    }


# ─────────────────────────────────────────────
# TerraMind evaluation
# ─────────────────────────────────────────────
def run_terramind(data_dir: str, checkpoint: str) -> dict:
    """Run TerraMind model on test split. Returns metric dict."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds     = Sen1Floods11Dataset(data_dir, split="test")
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2)

    model = FloodDetectionModel(pretrained_encoder=True).to(device)
    model.load_head(checkpoint, device=str(device))
    model.eval()

    all_preds, all_labels = [], []

    print("🛰  Running TerraMind model...")
    with torch.no_grad():
        for sar, label in tqdm(loader):
            sar = sar.to(device)

            # FIX: use _get_embeddings to handle list output
            embeddings = model._get_embeddings(sar)
            logits     = model.head(embeddings)
            preds      = (torch.sigmoid(logits) > 0.5).float()

            all_preds.append(preds.cpu().numpy().flatten())
            all_labels.append(label.numpy().flatten())

    preds  = np.concatenate(all_preds).astype(int)
    labels = np.concatenate(all_labels).astype(int)

    return {
        "mIoU":      _miou(preds, labels),
        "F1":        f1_score(labels, preds, zero_division=0),
        "Precision": precision_score(labels, preds, zero_division=0),
        "Recall":    recall_score(labels, preds, zero_division=0),
    }


def _miou(preds: np.ndarray, labels: np.ndarray) -> float:
    intersection = (preds * labels).sum()
    union        = preds.sum() + labels.sum() - intersection
    return float(intersection / (union + 1e-8)) if union > 0 else 1.0


# ─────────────────────────────────────────────
# Pretty print
# ─────────────────────────────────────────────
def print_results(baseline: dict, terramind: dict):
    def pct(new, old):
        return f"+{((new-old)/old*100):.0f}%" if new > old else f"{((new-old)/old*100):.0f}%"

    print("\n" + "─" * 50)
    print("  EVALUATION RESULTS — Sen1Floods11 Test Split")
    print("─" * 50)
    print(f"  {'Metric':<12} {'Baseline (OTSU)':>16} {'TerraMind (ours)':>18} {'Delta':>8}")
    print("  " + "─" * 46)
    for k in ["mIoU", "F1", "Precision", "Recall"]:
        b = baseline[k]
        t = terramind[k]
        print(f"  {k:<12} {b:>16.4f} {t:>18.4f} {pct(t,b):>8}")
    print("─" * 50)
    print(f"\n  → Headline: mIoU {baseline['mIoU']:.3f} → {terramind['mIoU']:.3f} "
          f"({pct(terramind['mIoU'], baseline['mIoU'])} improvement)")
    print("  → Put this number on slide 4.\n")

    import json
    with open("eval_results.json", "w") as f:
        json.dump({"baseline": baseline, "terramind": terramind}, f, indent=2)
    print("  💾 Saved to eval_results.json")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="data/sen1floods11",        type=str)
    parser.add_argument("--checkpoint", default="models/flood_head_best.pt", type=str)
    args = parser.parse_args()

    baseline_metrics  = run_baseline(args.data_dir)
    terramind_metrics = run_terramind(args.data_dir, args.checkpoint)
    print_results(baseline_metrics, terramind_metrics)