import torch
import torch.nn as nn
import torch.nn.functional as F
import rasterio
import numpy as np
from transformers import AutoModel

# -----------------------------
# Step 1: Load model
# -----------------------------
model_name = "facebook/dino-vitb16"
model = AutoModel.from_pretrained(model_name)
model.eval()

# -----------------------------
# Step 2: Simple segmentation head
# -----------------------------
class FloodHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(768, 1)  # 768 → 1 (flood probability)

    def forward(self, x):
        x = self.fc(x)
        return x

head = FloodHead()

# -----------------------------
# Step 3: Load and preprocess image
# -----------------------------
image_path = "data/sample.tif"

with rasterio.open(image_path) as src:
    image = src.read()

# Fix channels
if image.shape[0] == 2:
    image = np.stack([image[0], image[1], image[0]])
else:
    image = image[:3]

# Normalize
image = image.astype(np.float32)
image = (image - image.min()) / (image.max() - image.min())

image_tensor = torch.tensor(image).unsqueeze(0)

# Resize
image_tensor = F.interpolate(image_tensor, size=(224, 224), mode='bilinear', align_corners=False)

# -----------------------------
# Step 4: Get features
# -----------------------------
with torch.no_grad():
    output = model(image_tensor)

features = output.last_hidden_state  # (1, 197, 768)

# Remove CLS token
features = features[:, 1:, :]  # (1, 196, 768)

# -----------------------------
# Step 5: Predict flood
# -----------------------------
flood_pred = head(features)  # (1, 196, 1)

# Reshape to 14x14 grid
flood_map = flood_pred.reshape(1, 14, 14)

print("Flood map shape:", flood_map.shape)