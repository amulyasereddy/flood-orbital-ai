"""
train.py
========
Fine-tune the FloodSegHead on Sen1Floods11 dataset.
TerraMind encoder is FROZEN — only the head learns.

Usage:
    python train.py --data_dir data/sen1floods11 --epochs 20 --lr 1e-4
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import rasterio
import wandb
from tqdm import tqdm
from sklearn.metrics import f1_score

from flood_model import FloodDetectionModel
from preprocess import normalise_terramind, normalise_minmax


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class Sen1Floods11Dataset(Dataset):
    def __init__(self, root: str, split: str = "train", target_size: int = 224):
        self.target_size = target_size
        s1_dir    = Path(root) / split / "s1"
        label_dir = Path(root) / split / "label"

        self.s1_files    = sorted(s1_dir.glob("*.tif"))
        self.label_files = sorted(label_dir.glob("*.tif"))

        assert len(self.s1_files) == len(self.label_files), \
            f"Mismatch: {len(self.s1_files)} SAR vs {len(self.label_files)} labels"
        print(f"  [{split}] {len(self.s1_files)} tiles loaded")

    def __len__(self):
        return len(self.s1_files)

    def __getitem__(self, idx):
        from rasterio.enums import Resampling

        # ── Load SAR tile ──
        with rasterio.open(self.s1_files[idx]) as src:
            sar = src.read(
                out_shape=(src.count, self.target_size, self.target_size),
                resampling=Resampling.bilinear,
            ).astype(np.float32)

        if sar.shape[0] == 1:
            sar = np.concatenate([sar, sar], axis=0)
        elif sar.shape[0] >= 2:
            sar = sar[:2]

        # Replace NaN/Inf in SAR
        sar = np.nan_to_num(sar, nan=0.0, posinf=0.0, neginf=0.0)
        sar = normalise_terramind(sar)

        # ── Load label mask ──
        with rasterio.open(self.label_files[idx]) as src:
            label = src.read(
                1,
                out_shape=(self.target_size, self.target_size),
                resampling=Resampling.nearest,
            ).astype(np.float32)

        # Replace invalid values (-1, 255, NaN) → 0
        label = np.nan_to_num(label, nan=0.0, posinf=0.0, neginf=0.0)
        label = np.clip(label, 0, 1)

        return (
            torch.from_numpy(sar).float(),
            torch.from_numpy(label).float().unsqueeze(0),  # (1, H, W)
        )


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────
def compute_miou(preds: torch.Tensor, labels: torch.Tensor,
                 threshold: float = 0.5) -> float:
    preds_bin  = (torch.sigmoid(preds) > threshold).float().cpu().numpy().flatten()
    labels_bin = labels.cpu().numpy().flatten()
    intersection = (preds_bin * labels_bin).sum()
    union        = preds_bin.sum() + labels_bin.sum() - intersection
    if union == 0:
        return 1.0
    return float(intersection / (union + 1e-8))


def compute_f1(preds: torch.Tensor, labels: torch.Tensor,
               threshold: float = 0.5) -> float:
    preds_bin  = (torch.sigmoid(preds) > threshold).cpu().numpy().flatten().astype(int)
    labels_bin = labels.cpu().numpy().flatten().astype(int)
    return f1_score(labels_bin, preds_bin, zero_division=0)


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥  Device: {device}")

    # ── Datasets ──
    print("📂 Loading Sen1Floods11...")
    train_ds = Sen1Floods11Dataset(args.data_dir, split="train")
    val_ds   = Sen1Floods11Dataset(args.data_dir, split="val")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0)  # num_workers=0 for Windows
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    # ── Model ──
    print("🔄 Loading TerraMind-small encoder...")
    model = FloodDetectionModel(pretrained_encoder=True).to(device)

    # Freeze encoder
    for p in model.encoder.parameters():
        p.requires_grad = False

    encoder_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    head_params    = sum(p.numel() for p in model.head.parameters()    if p.requires_grad)
    print(f"   Encoder trainable params : {encoder_params:,} (should be 0)")
    print(f"   Head trainable params    : {head_params:,}")

    # ── Loss & optimiser ──
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(device))
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── W&B ──
    wandb.init(
        project="flood-terramind-418",
        config=vars(args),
        name=f"terramind-small-sen1floods11-lr{args.lr}",
    )

    # ── Training ──
    best_val_miou = 0.0
    os.makedirs("models", exist_ok=True)

    for epoch in range(1, args.epochs + 1):

        # ── Train epoch ──
        model.train()
        model.encoder.eval()   # Keep encoder in eval mode (frozen)
        train_loss = 0.0
        valid_batches = 0

        for sar, label in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]"):
            sar, label = sar.to(device), label.to(device)
            optimizer.zero_grad()

            # FIX: encoder under no_grad (frozen), then detach so head gets clean tensor
            with torch.no_grad():
                embeddings = model.encoder({"S1GRD": sar})
                if isinstance(embeddings, list):
                    embeddings = embeddings[-1]
            
            # Detach from encoder graph — head will build its own graph
            embeddings = embeddings.detach()

            logits = model.head(embeddings)  # (B, 1, H, W)

            # Safety check — skip batch if shapes mismatch
            if logits.shape != label.shape:
                print(f"  ⚠️  Shape mismatch: logits={logits.shape} label={label.shape}, skipping")
                continue

            loss = criterion(logits, label)

            # Skip if NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  ⚠️  NaN/Inf loss detected, skipping batch")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.head.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            valid_batches += 1

        train_loss = train_loss / valid_batches if valid_batches > 0 else float('nan')

        # ── Val epoch ──
        model.eval()
        val_loss = val_miou = val_f1 = 0.0
        val_batches = 0

        with torch.no_grad():
            for sar, label in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]  "):
                sar, label = sar.to(device), label.to(device)

                embeddings = model.encoder({"S1GRD": sar})
                if isinstance(embeddings, list):
                    embeddings = embeddings[-1]

                logits = model.head(embeddings)

                if logits.shape != label.shape:
                    continue

                l = criterion(logits, label)
                if torch.isnan(l) or torch.isinf(l):
                    continue

                val_loss  += l.item()
                val_miou  += compute_miou(logits, label)
                val_f1    += compute_f1(logits, label)
                val_batches += 1

        if val_batches > 0:
            val_loss /= val_batches
            val_miou /= val_batches
            val_f1   /= val_batches
        
        scheduler.step()

        print(f"  Epoch {epoch:02d} | "
              f"train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | "
              f"val_mIoU={val_miou:.4f} | "
              f"val_F1={val_f1:.4f}")

        wandb.log({
            "epoch":      epoch,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "val_mIoU":   val_miou,
            "val_F1":     val_f1,
            "lr":         scheduler.get_last_lr()[0],
        })

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            torch.save(model.head.state_dict(), "models/flood_head_best.pt")
            print(f"  💾 Saved best model (val_mIoU={best_val_miou:.4f})")

    print(f"\n✅ Training complete. Best val mIoU: {best_val_miou:.4f}")
    wandb.finish()


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train flood detection head on Sen1Floods11")
    parser.add_argument("--data_dir",   default="data/sen1floods11", type=str)
    parser.add_argument("--epochs",     default=20,   type=int)
    parser.add_argument("--batch_size", default=8,    type=int)
    parser.add_argument("--lr",         default=1e-4, type=float)
    args = parser.parse_args()
    train(args)