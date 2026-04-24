import streamlit as st
import torch
import torch.nn.functional as F
import rasterio
import numpy as np
from transformers import AutoModel
import torch.nn as nn
import folium
from streamlit_folium import st_folium

# -----------------------------
# Title
# -----------------------------
st.title("🛰️ AI-Powered Orbital Flood Intelligence System")

st.markdown("""
### 🧠 System Overview
- Detects flood regions from satellite imagery  
- Estimates flood impact  
- Ranks regions based on importance  
- Simulates onboard satellite decision-making  
""")

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    return AutoModel.from_pretrained("facebook/dino-vitb16")

model = load_model()
model.eval()

# -----------------------------
# Flood head
# -----------------------------
class FloodHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(768, 1)

    def forward(self, x):
        return self.fc(x)

head = FloodHead()

# -----------------------------
# Load image
# -----------------------------
# -----------------------------
# Upload option
# -----------------------------
uploaded_file = st.file_uploader("📤 Upload Satellite Image (.tif)", type=["tif"])

if uploaded_file is not None:
    with rasterio.open(uploaded_file) as src:
        image = src.read()
else:
    st.warning("Using default sample image")
    with rasterio.open("data/sample.tif") as src:
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
image_tensor = F.interpolate(image_tensor, size=(224, 224), mode='bilinear', align_corners=False)

# -----------------------------
# Model forward
# -----------------------------
with torch.no_grad():
    output = model(image_tensor)

features = output.last_hidden_state[:, 1:, :]

# -----------------------------
# Flood prediction
# -----------------------------
flood_pred = head(features)
flood_prob = torch.sigmoid(flood_pred)
confidence = flood_prob.mean().item()

# -----------------------------
# Create flood map
# -----------------------------
flood_map = flood_pred.reshape(14, 14).detach().numpy()

flood_map = torch.tensor(flood_map).unsqueeze(0).unsqueeze(0)
flood_map = F.interpolate(flood_map, size=(512, 512), mode='bilinear', align_corners=False)
flood_map = flood_map.squeeze().numpy()

# -----------------------------
# Flood area
# -----------------------------
flood_binary = flood_map > 0.5
flood_percentage = (flood_binary.sum() / flood_map.size) * 100

# -----------------------------
# LAND TYPE ESTIMATION (heuristic)
# -----------------------------

# Use original image intensity
base = image[0]  # grayscale-like

# Simple thresholds
farmland = (base > 0.3) & (base < 0.6)
urban = base > 0.6
water = base < 0.3

# Flooded areas
flooded = flood_map > 0.5

# Combine
flooded_farmland = farmland & flooded
flooded_urban = urban & flooded

# Percentages
farmland_damage = (flooded_farmland.sum() / flood_map.size) * 100
urban_damage = (flooded_urban.sum() / flood_map.size) * 100

# -----------------------------
# Display images
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Image")
    st.image(image[0], clamp=True)

with col2:
    st.subheader("Flood Heatmap")
    st.image(flood_map, clamp=True)

# -----------------------------
# Metrics
# -----------------------------
st.write(f"### 📊 Confidence Score: {confidence:.2f}")
st.write(f"### 🌊 Flood Area: {flood_percentage:.2f}%")
st.write(f"### 🌾 Farmland Affected: {farmland_damage:.2f}%")
st.write(f"### 🏙️ Urban Area Affected: {urban_damage:.2f}%")

if confidence > 0.5:
    st.success("🚀 SEND TO EARTH")
else:
    st.error("❌ SKIP")

# -----------------------------
# SINGLE TILE MAP
# -----------------------------
st.markdown("### 🗺️ Flood Region Map")

m = folium.Map(location=[26.1, 91.7], zoom_start=10)

folium.Rectangle(
    bounds=[[26.0, 91.6], [26.2, 91.8]],
    color="red" if confidence > 0.5 else "blue",
    fill=True,
    fill_opacity=0.4,
).add_to(m)

st_folium(m, width=700, height=400)

# -----------------------------
# MULTI-TILE ANALYSIS
# -----------------------------
st.markdown("### 🛰️ Tile Priority System")

tiles = []
h, w = image.shape[1], image.shape[2]

tiles.append(image[:, :h//2, :w//2])
tiles.append(image[:, :h//2, w//2:])
tiles.append(image[:, h//2:, :w//2])
tiles.append(image[:, h//2:, w//2:])

tile_scores = []

for i, tile in enumerate(tiles):
    tile_tensor = torch.tensor(tile).unsqueeze(0)
    tile_tensor = F.interpolate(tile_tensor, size=(224, 224), mode='bilinear', align_corners=False)

    with torch.no_grad():
        output = model(tile_tensor)

    features = output.last_hidden_state[:, 1:, :]
    flood_pred = head(features)
    flood_prob = torch.sigmoid(flood_pred)

    score = flood_prob.mean().item()
    tile_scores.append((i, score))

# Sort tiles
tile_scores.sort(key=lambda x: x[1], reverse=True)

# -----------------------------
# Tile ranking display
# -----------------------------
st.markdown("### 📊 Tile Ranking")
st.markdown("### 📊 Tile Decision Dashboard")

for i, (idx, score) in enumerate(tile_scores):
    if i < 2:
        st.success(f"Tile {idx} → {score:.2f} → SEND")
    else:
        st.warning(f"Tile {idx} → {score:.2f} → SKIP")

for idx, score in tile_scores:
    st.write(f"Tile {idx} → Score: {score:.2f}")

# -----------------------------
# MULTI-TILE MAP
# -----------------------------
st.markdown("### 🗺️ Priority Map")

m2 = folium.Map(location=[26.1, 91.7], zoom_start=10)

coords = [
    [[26.0, 91.6], [26.1, 91.7]],
    [[26.0, 91.7], [26.1, 91.8]],
    [[26.1, 91.6], [26.2, 91.7]],
    [[26.1, 91.7], [26.2, 91.8]],
]

for i, (idx, score) in enumerate(tile_scores):
    color = "red" if i < 2 else "blue"

    folium.Rectangle(
        bounds=coords[idx],
        color=color,
        fill=True,
        fill_opacity=0.4,
        tooltip=f"Tile {idx} | Score: {score:.2f}"
    ).add_to(m2)

st_folium(m2, width=700, height=400)
st.markdown("### 🧠 AI Explanation")

if confidence > 0.5:
    st.write("""
    The system detected strong flood patterns.
    This region is likely heavily affected.
    Sending this tile helps disaster response.
    """)
else:
    st.write("""
    The region shows low flood probability.
    Satellite skips this tile to save bandwidth.
    """)
st.markdown("### 📡 Bandwidth Optimization")

total_tiles = len(tile_scores)
sent_tiles = 2

saved = ((total_tiles - sent_tiles) / total_tiles) * 100

st.write(f"Total Tiles: {total_tiles}")
st.write(f"Tiles Sent: {sent_tiles}")
st.write(f"🚀 Bandwidth Saved: {saved:.2f}%")