import torch
import torch.nn.functional as F
import rasterio
import numpy as np
from transformers import AutoModel
import torch.nn as nn

# Load model
model = AutoModel.from_pretrained("facebook/dino-vitb16")
model.eval()

class FloodHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(768, 1)

    def forward(self, x):
        return self.fc(x)

head = FloodHead()

# Load image
with rasterio.open("data/sample.tif") as src:
    image = src.read()

# Preprocess
if image.shape[0] == 2:
    image = np.stack([image[0], image[1], image[0]])
else:
    image = image[:3]

image = image.astype(np.float32)
image = (image - image.min()) / (image.max() - image.min())

# Split into 4 tiles
tiles = []
h, w = image.shape[1], image.shape[2]

tiles.append(image[:, :h//2, :w//2])
tiles.append(image[:, :h//2, w//2:])
tiles.append(image[:, h//2:, :w//2])
tiles.append(image[:, h//2:, w//2:])

scores = []

for i, tile in enumerate(tiles):
    tile_tensor = torch.tensor(tile).unsqueeze(0)
    tile_tensor = F.interpolate(tile_tensor, size=(224, 224), mode='bilinear', align_corners=False)

    with torch.no_grad():
        output = model(tile_tensor)

    features = output.last_hidden_state[:, 1:, :]
    flood_pred = head(features)
    flood_prob = torch.sigmoid(flood_pred)

    confidence = flood_prob.mean().item()
    scores.append((i, confidence))

# Rank tiles
scores.sort(key=lambda x: x[1], reverse=True)

print("\n🛰️ TILE PRIORITY:")
for idx, score in scores:
    print(f"Tile {idx} → Score: {score:.3f}")

print("\n🚀 Selected Tiles (Top 2):")
for idx, score in scores[:2]:
    print(f"Tile {idx}")