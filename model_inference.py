"""
model_inference.py
==================
Standalone inference test — verifies the full TerraMind pipeline works
from raw .tif to flood output. Good for debugging.
Usage:
    python model_inference.py
    python model_inference.py --input data/your_tile.tif
"""

import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from flood_model import FloodDetectionModel
from preprocess import preprocess_tile, compute_flood_area_km2


def run(input_path: str = None, checkpoint: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥  Device: {device}")

    # ── Model ──
    print("🔄 Loading TerraMind-small encoder + flood head...")
    model = FloodDetectionModel(pretrained_encoder=True).to(device)

    if checkpoint and Path(checkpoint).exists():
        model.load_head(checkpoint, str(device))
    else:
        print("⚠️  No checkpoint path given — head weights are random (smoke test only)")

    model.eval()
    print("✅ Model ready")

    # ── Input ──
    if input_path and Path(input_path).exists():
        print(f"📂 Loading: {input_path}")
        sar_tensor = preprocess_tile(input_path).to(device)
    else:
        print("📂 No .tif found — using synthetic SAR (2ch, 224×224)")
        sar_tensor = torch.randn(1, 2, 224, 224).to(device)

    print(f"   Input shape: {sar_tensor.shape}")
    print(f"   Value range: [{sar_tensor.min():.3f}, {sar_tensor.max():.3f}]")

    # ── Inference ──
    print("\n🛰  Running inference...")
    with torch.no_grad():
        # Step 1: TerraMind encoder → patch embeddings
        embeddings = model.encoder({"S1GRD": sar_tensor})

        # FIX: TerraMind returns a list of tensors — take the last one
        if isinstance(embeddings, list):
            embeddings = embeddings[-1]

        print(f"   TerraMind embeddings : {embeddings.shape}  (196 patches × 768 dims)")

        # Step 2: Flood head → logits
        logits = model.head(embeddings)
        print(f"   Flood logits         : {logits.shape}")

        # Step 3: Full model output
        out = model(sar_tensor)

    prob_map  = out["prob_map"].squeeze().cpu().numpy()
    flood_mask = out["flood_mask"].squeeze().cpu().numpy()
    confidence = out["confidence"]
    flood_km2  = compute_flood_area_km2(flood_mask)

    print(f"\n📊 Results:")
    print(f"   Confidence     : {confidence:.4f}")
    print(f"   Flooded pixels : {int(flood_mask.sum()):,} / {224*224:,}")
    print(f"   Flooded area   : {flood_km2:.3f} km²")
    print(f"   Decision       : {'🚀 SEND' if confidence > 0.5 else '❌ SKIP'}")
    print("\n✅ Inference pipeline verified.")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      default=None,                        type=str)
    parser.add_argument("--checkpoint", default="models/flood_head_best.pt", type=str)
    args = parser.parse_args()
    run(args.input, args.checkpoint)