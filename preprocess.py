import rasterio
import torch
import numpy as np

image_path = "data/sample.tif"

with rasterio.open(image_path) as src:
    image = src.read()

print("Original shape:", image.shape)

# If image has only 2 channels, convert to 3
if image.shape[0] == 2:
    third_channel = image[0]  # duplicate first channel
    image = np.stack([image[0], image[1], third_channel])
else:
    image = image[:3]

image = image.astype(np.float32)
image = (image - image.min()) / (image.max() - image.min())

image_tensor = torch.tensor(image)

image_tensor = image_tensor.unsqueeze(0)

print("Processed shape:", image_tensor.shape)