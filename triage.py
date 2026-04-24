import torch
import torch.nn.functional as F
import rasterio
import numpy as np
from transformers import AutoModel
import torch.nn as nn

# Load model
model = AutoModel.from_pretrained("facebook/dino-vitb16")
model.eval()

# Flood head
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

if image.shape[0] == 2:
    image = np.stack([image[0], image[1], image[0]])
else:
    image = image[:3]

image = image.astype(np.float32)
image = (image - image.min()) / (image.max() - image.min())

image_tensor = torch.tensor(image).unsqueeze(0)

# Resize
image_tensor = F.interpolate(image_tensor, size=(224, 224), mode='bilinear', align_corners=False)

# Model forward
with torch.no_grad():
    output = model(image_tensor)

features = output.last_hidden_state[:, 1:, :]

# Flood prediction
flood_pred = head(features)

# Convert to probability
flood_prob = torch.sigmoid(flood_pred)

# Confidence score
confidence = flood_prob.mean().item()

print("Flood confidence score:", confidence)

# Decision
if confidence > 0.5:
    print("🚀 SEND TO EARTH (important tile)")
else:
    print("❌ SKIP (not important)")