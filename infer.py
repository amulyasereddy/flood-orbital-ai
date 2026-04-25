"""
infer.py
========
Run flood detection inference on a single SAR .tif tile.
This is the required submission entry point.

Usage:
    python infer.py --input sample.tif
    python infer.py --input sample.tif --checkpoint models/flood_head_best.pt --output results/

Output:
    - flood_mask.png       : binary flood mask visualisation
    - flood_result.json    : area stats in machine-readable format
    - flood_overlay.png    : input + mask overlay side-by-side
"""

import argparse
import json
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio
from rasterio.enums import Resampling
from pathlib import Path

from flood_model import FloodDetectionModel
from preprocess import preprocess_tile, compute_flood_area_km2


def run_inference(input_path: str,
                  checkpoint_path: str = None,
                  output_dir: str = "results",
                  pixel_size_m: float = 10.0) -> dict:
    """
    Full inference pipeline.
    Returns dict with flood stats.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥  Device     : {device}")
    print(f"📂 Input      : {input_path}")

    # ── Load model ──
    print("🔄 Loading TerraMind-small...")
    model = FloodDetectionModel(pretrained_encoder=True).to(device)

    if checkpoint_path and Path(checkpoint_path).exists():
        model.load_head(checkpoint_path, device=str(device))
        print(f"✅ Loaded checkpoint: {checkpoint_path}")
    else:
        print("⚠️  No checkpoint — using random head weights (for testing only)")

    model.eval()

    # ── Load & preprocess SAR tile ──
    print("📡 Preprocessing SAR tile...")
    sar_tensor = preprocess_tile(input_path).to(device)  # (1, 2, 224, 224)
    print(f"   SAR tensor shape : {sar_tensor.shape}")

    # ── Inference ──
    print("🛰  Running TerraMind inference...")
    with torch.no_grad():
        out = model(sar_tensor)

    prob_map   = out["prob_map"].squeeze().cpu().numpy()   # (224, 224)
    flood_mask = out["flood_mask"].squeeze().cpu().numpy() # (224, 224) binary
    confidence = out["confidence"]

    # ── Compute area stats ──
    flood_area_km2 = compute_flood_area_km2(flood_mask, pixel_size_m)
    total_area_km2 = compute_flood_area_km2(np.ones_like(flood_mask), pixel_size_m)
    flood_pct      = (flood_mask.sum() / flood_mask.size) * 100

    result = {
        "input_file":      str(input_path),
        "confidence":      round(float(confidence), 4),
        "flood_area_km2":  round(float(flood_area_km2), 3),
        "total_area_km2":  round(float(total_area_km2), 3),
        "flood_pct":       round(float(flood_pct), 2),
        "downlink_decision": "SEND" if confidence > 0.5 else "SKIP",
        "model":           "TerraMind-small + FloodSegHead",
    }

    # ── Save outputs ──
    # 1. Raw SAR input (first channel)
    with rasterio.open(input_path) as src:
        raw = src.read(
            1,
            out_shape=(224, 224),
            resampling=Resampling.bilinear,
        ).astype(np.float32)
    raw = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)

    # 2. Flood mask PNG
    mask_path = os.path.join(output_dir, "flood_mask.png")
    plt.figure(figsize=(6, 6))
    plt.imshow(flood_mask, cmap="Blues", vmin=0, vmax=1)
    plt.title(f"Flood Mask  (confidence={confidence:.3f})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(mask_path, dpi=150)
    plt.close()

    # 3. Probability heatmap
    heat_path = os.path.join(output_dir, "flood_heatmap.png")
    plt.figure(figsize=(6, 6))
    im = plt.imshow(prob_map, cmap="RdYlBu_r", vmin=0, vmax=1)
    plt.colorbar(im, fraction=0.046)
    plt.title("Flood Probability Map")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(heat_path, dpi=150)
    plt.close()

    # 4. Side-by-side overlay
    overlay_path = os.path.join(output_dir, "flood_overlay.png")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(raw, cmap="gray");           axes[0].set_title("SAR Input (VV)");  axes[0].axis("off")
    axes[1].imshow(prob_map, cmap="RdYlBu_r");  axes[1].set_title("Flood Probability"); axes[1].axis("off")
    axes[2].imshow(raw, cmap="gray")
    axes[2].imshow(flood_mask, cmap="Blues", alpha=0.5)
    axes[2].set_title(f"Flood Overlay\n{flood_area_km2:.2f} km² flooded")
    axes[2].axis("off")
    fig.suptitle(
        f"TerraMind Flood Detection  |  Confidence: {confidence:.3f}  |  "
        f"Decision: {'🚀 SEND' if result['downlink_decision']=='SEND' else '❌ SKIP'}",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(overlay_path, dpi=150)
    plt.close()

    # 5. JSON result
    json_path = os.path.join(output_dir, "flood_result.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    # ── Print summary ──
    print("\n" + "─" * 45)
    print("  INFERENCE RESULT")
    print("─" * 45)
    print(f"  Confidence       : {confidence:.4f}")
    print(f"  Flooded area     : {flood_area_km2:.3f} km²")
    print(f"  Flood coverage   : {flood_pct:.2f}%")
    print(f"  Downlink decision: {result['downlink_decision']}")
    print("─" * 45)
    print(f"  Outputs saved to : {output_dir}/")
    print(f"    flood_mask.png, flood_heatmap.png, flood_overlay.png, flood_result.json")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TerraMind Flood Inference")
    parser.add_argument("--input",      required=True,
                        help="Path to SAR .tif tile")
    parser.add_argument("--checkpoint", default="models/flood_head_best.pt",
                        help="Path to trained head checkpoint")
    parser.add_argument("--output",     default="results",
                        help="Output directory for results")
    parser.add_argument("--pixel_size", default=10.0, type=float,
                        help="Pixel size in metres (default: 10m for S1 GRD)")
    args = parser.parse_args()

    run_inference(
        input_path=args.input,
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        pixel_size_m=args.pixel_size,
    )