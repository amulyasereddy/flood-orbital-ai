import torch
from transformers import AutoModel

# Step 1: Model name (TerraMind small)
model_name = "facebook/dino-vitb16"

print("🔄 Loading TerraMind model...")

# Step 2: Load model
model = AutoModel.from_pretrained(model_name)

print("✅ Model loaded successfully!")

# Step 3: Create dummy input (fake satellite image)
# Shape: (batch_size, channels, height, width)
dummy_input = torch.randn(1, 3, 224, 224)

print("📥 Running dummy input through model...")

# Step 4: Run inference
with torch.no_grad():
    output = model(dummy_input)

# Step 5: Print output details
print("📤 Output received!")

# Some models return different structures, handle safely
try:
    print("Output shape:", output.last_hidden_state.shape)
except:
    print("Output:", output)