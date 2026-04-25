"""
flood_model.py
==============
TerraMind-powered flood segmentation model.
Encoder : TerraMind-small (frozen) — S1GRD SAR modality
Head    : UNet-style decoder fine-tuned on Sen1Floods11
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from terratorch import BACKBONE_REGISTRY


# ─────────────────────────────────────────────
# 1. TerraMind Encoder (frozen during training)
# ─────────────────────────────────────────────
def build_terramind_encoder(pretrained: bool = True):
    """
    Returns TerraMind-small encoder loaded with S1GRD (SAR) modality.
    Output shape per forward pass: (B, 196, 768)
      196 = 14×14 patches of a 224×224 input
      768 = embedding dimension
    """
    encoder = BACKBONE_REGISTRY.build(
        "terramind_v1_small",
        pretrained=pretrained,
        modalities=["S1GRD"],   # Sentinel-1 SAR — 2 channels: VV, VH
    )
    return encoder


# ─────────────────────────────────────────────
# 2. Flood Segmentation Head
# ─────────────────────────────────────────────
class FloodSegHead(nn.Module):
    """
    Lightweight decoder on top of frozen TerraMind patch embeddings.
    Input  : (B, 196, 768) patch embeddings from TerraMind
    Output : (B, 1, 224, 224) flood probability map (logits)
    """

    def __init__(self, embed_dim: int = 384, output_size: int = 224):
        super().__init__()
        self.output_size = output_size

        # Project embeddings to smaller dim
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
        )

        # Upsampling conv decoder: 14×14 → 224×224
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 14→28
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 28→56
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),   # 56→112
            nn.GELU(),
            nn.ConvTranspose2d(16,  1, kernel_size=4, stride=2, padding=1),   # 112→224
        )

    def forward(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        """
        patch_embeddings : (B, 196, 768)
        returns          : (B, 1, 224, 224) logits
        """
        B = patch_embeddings.shape[0]
        x = self.proj(patch_embeddings)      # (B, 196, 128)
        x = x.permute(0, 2, 1)              # (B, 128, 196)
        x = x.reshape(B, 128, 14, 14)       # (B, 128, 14, 14)
        x = self.decoder(x)                  # (B, 1, 224, 224)
        return x


# ─────────────────────────────────────────────
# 3. Full model wrapper (encoder + head)
# ─────────────────────────────────────────────
class FloodDetectionModel(nn.Module):
    """
    Complete inference model: SAR tile → flood mask + confidence score.
    Use this in app.py and infer.py.
    """

    def __init__(self, pretrained_encoder: bool = True):
        super().__init__()
        self.encoder = build_terramind_encoder(pretrained=pretrained_encoder)
        self.head    = FloodSegHead()

        # Freeze encoder — only head trains
        for param in self.encoder.parameters():
            param.requires_grad = False

    def _get_embeddings(self, sar_tensor: torch.Tensor) -> torch.Tensor:
        """
        Helper: run encoder and handle list vs tensor output.
        TerraMind may return a list of feature maps — we always want
        the last one which is (B, 196, 768).
        """
        embeddings = self.encoder({"S1GRD": sar_tensor})
        # FIX: TerraMind returns a list of tensors — take the last one
        if isinstance(embeddings, list):
            embeddings = embeddings[-1]
        return embeddings

    def forward(self, sar_tensor: torch.Tensor):
        """
        sar_tensor : (B, 2, 224, 224) — 2-channel SAR (VV, VH), normalised
        returns    : dict with 'logits', 'prob_map', 'confidence', 'flood_mask'
        """
        embeddings = self._get_embeddings(sar_tensor)   # (B, 196, 768)
        logits     = self.head(embeddings)               # (B, 1, 224, 224)
        prob_map   = torch.sigmoid(logits)               # (B, 1, 224, 224)
        confidence = prob_map.mean().item()
        flood_mask = (prob_map > 0.5).float()

        return {
            "logits":     logits,
            "prob_map":   prob_map,
            "confidence": confidence,
            "flood_mask": flood_mask,
        }

    def load_head(self, checkpoint_path: str, device: str = "cpu"):
        """Load fine-tuned head weights from checkpoint."""
        state = torch.load(checkpoint_path, map_location=device)
        self.head.load_state_dict(state)
        print(f"✅ Loaded head weights from {checkpoint_path}")


# ─────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("🔄 Building FloodDetectionModel...")
    model = FloodDetectionModel(pretrained_encoder=True)
    model.eval()

    dummy_sar = torch.randn(1, 2, 224, 224)   # 1 tile, 2-ch SAR
    with torch.no_grad():
        out = model(dummy_sar)

    print("✅ Forward pass OK")
    print(f"   prob_map shape : {out['prob_map'].shape}")
    print(f"   confidence     : {out['confidence']:.4f}")
    print(f"   flood_mask     : {out['flood_mask'].sum().item():.0f} / {224*224} pixels flagged")