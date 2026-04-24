import torch.nn.functional as F
import torch
import rasterio
import numpy as np
from transformers import AutoModel

# -----------------------------
# Step 1: Load model
# -----------------------------
model_name = "facebook/dino-vitb16"
model = AutoModel.from_pretrained(model_name)

model.eval()

print("✅ Model loaded")

# -----------------------------
# Step 2: Load and preprocess image
# -----------------------------
image_path = "data/sample.tif"

with rasterio.open(image_path) as src:
    image = src.read()

# Fix channels
if image.shape[0] == 2:
    third_channel = image[0]
    image = np.stack([image[0], image[1], third_channel])
else:
    image = image[:3]

# Normalize
image = image.astype(np.float32)
image = (image - image.min()) / (image.max() - image.min())

# Convert to tensor
image_tensor = torch.tensor(image).unsqueeze(0)
# Resize to 224x224 (VERY IMPORTANT)
image_tensor = F.interpolate(image_tensor, size=(224, 224), mode='bilinear', align_corners=False)

print("📥 Image ready:", image_tensor.shape)

# -----------------------------
# Step 3: Pass through model
# -----------------------------
with torch.no_grad():
    output = model(image_tensor)

# -----------------------------
# Step 4: Print output
# -----------------------------
try:
    features = output.last_hidden_state
    print("📤 Output features shape:", features.shape)
except:
    print("Output:", output)