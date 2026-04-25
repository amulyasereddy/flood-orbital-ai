"""
preprocess.py
=============
SAR tile loading and normalisation for TerraMind S1GRD modality.
TerraMind expects 2-channel input (VV, VH) — NOT 3-channel RGB.
"""

import numpy as np
import torch
import rasterio
from rasterio.enums import Resampling
from pathlib import Path


# TerraMind S1GRD pretraining statistics (from TerraMesh dataset)
# Apply these for best performance — matches what the model was pretrained on
S1GRD_MEAN = {"VV": -12.59, "VH": -20.26}
S1GRD_STD  = {"VV":  5.26,  "VH":  5.91}


def load_sar_tile(path: str, target_size: int = 224) -> np.ndarray:
    """
    Load a Sentinel-1 GRD .tif file.
    Returns numpy array of shape (2, H, W) — channels: VV, VH.
    Resamples to target_size × target_size.
    """
    with rasterio.open(path) as src:
        # Read all bands
        img = src.read(
            out_shape=(src.count, target_size, target_size),
            resampling=Resampling.bilinear,
        )

    img = img.astype(np.float32)

    # Handle different band counts
    if img.shape[0] == 1:
        # Single band — duplicate to get 2 channels
        img = np.concatenate([img, img], axis=0)
    elif img.shape[0] >= 2:
        # Take first two bands (VV, VH)
        img = img[:2]

    return img  # (2, 224, 224)


def normalise_terramind(img: np.ndarray) -> np.ndarray:
    """
    Apply TerraMind S1GRD pretraining normalisation.
    Input  : (2, H, W) raw SAR backscatter (dB scale)
    Output : (2, H, W) normalised to ~[-1, 1] range
    """
    means = np.array([S1GRD_MEAN["VV"], S1GRD_MEAN["VH"]], dtype=np.float32)
    stds  = np.array([S1GRD_STD["VV"],  S1GRD_STD["VH"]],  dtype=np.float32)

    # Reshape for broadcasting: (2,) → (2, 1, 1)
    means = means[:, None, None]
    stds  = stds[:, None, None]

    img = (img - means) / (stds + 1e-8)
    return img


def normalise_minmax(img: np.ndarray) -> np.ndarray:
    """
    Fallback: simple min-max normalisation per channel.
    Use when you don't know if input is in dB scale.
    """
    out = np.zeros_like(img)
    for c in range(img.shape[0]):
        mn, mx = img[c].min(), img[c].max()
        out[c] = (img[c] - mn) / (mx - mn + 1e-8)
    return out


def tile_to_tensor(img: np.ndarray) -> torch.Tensor:
    """
    Convert (2, H, W) numpy array to (1, 2, H, W) float32 tensor.
    The batch dimension = 1 (single tile inference).
    """
    tensor = torch.from_numpy(img).float()
    return tensor.unsqueeze(0)  # (1, 2, H, W)


def preprocess_tile(path: str, use_terramind_stats: bool = True) -> torch.Tensor:
    """
    Full pipeline: .tif path → model-ready tensor.
    Returns (1, 2, 224, 224) tensor.
    """
    img = load_sar_tile(path, target_size=224)

    if use_terramind_stats:
        img = normalise_terramind(img)
    else:
        img = normalise_minmax(img)

    return tile_to_tensor(img)


def compute_flood_area_km2(flood_mask: np.ndarray,
                           pixel_size_m: float = 10.0) -> float:
    """
    Compute flooded area in km² from a binary flood mask.
    Default pixel size = 10m (Sentinel-1 GRD standard resolution).
    """
    flooded_pixels = float(flood_mask.sum())
    area_m2 = flooded_pixels * (pixel_size_m ** 2)
    return area_m2 / 1_000_000  # convert to km²


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "sample.tif"

    if Path(path).exists():
        tensor = preprocess_tile(path)
        print(f"✅ Loaded: {path}")
        print(f"   Tensor shape : {tensor.shape}")
        print(f"   Value range  : [{tensor.min():.3f}, {tensor.max():.3f}]")
    else:
        # Demo with synthetic data
        print("⚠️  No .tif found — running with synthetic SAR data")
        dummy = np.random.randn(2, 224, 224).astype(np.float32) * 5 - 15
        tensor = tile_to_tensor(normalise_terramind(dummy))
        print(f"   Tensor shape : {tensor.shape}")
        print(f"   Value range  : [{tensor.min():.3f}, {tensor.max():.3f}]")