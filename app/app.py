import streamlit as st
import torch
import torch.nn.functional as F
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
from transformers import ViTModel
import os

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="Flood AI", layout="wide")

st.title("🛰️ AI-Powered Orbital Flood Intelligence System")
st.markdown("Real-time AI system that prioritizes satellite data transmission for disaster response.")

st.markdown("---")

# ----------------------------
# LOAD MODEL
# ----------------------------
@st.cache_resource
def load_model():
    return ViTModel.from_pretrained("facebook/dino-vitb16")

model = load_model()

# ----------------------------
# INPUT IMAGE
# ----------------------------
st.header("📥 Input Satellite Image")

uploaded_file = st.file_uploader("Upload .tif image", type=["tif", "tiff"])

# Decide file source
if uploaded_file is not None:
    st.success("Using uploaded image")
    file_to_read = uploaded_file
else:
    default_path = os.path.join(os.getcwd(), "sample.tif")

    if os.path.exists(default_path):
        st.info("Using default sample image")
        file_to_read = default_path
    else:
        st.error("sample.tif not found. Please upload an image.")
        st.stop()

# ----------------------------
# LOAD IMAGE
# ----------------------------
with rasterio.open(file_to_read) as src:
    img = src.read()

# Normalize
img = img.astype(np.float32)
img = (img - img.min()) / (img.max() - img.min() + 1e-8)

# Ensure 3 channels
if img.shape[0] == 1:
    img = np.repeat(img, 3, axis=0)
elif img.shape[0] == 2:
    img = np.concatenate([img, img[:1]], axis=0)

# Resize
img_tensor = torch.tensor(img).unsqueeze(0)
img_tensor = F.interpolate(img_tensor, size=(224, 224))

# ----------------------------
# MODEL INFERENCE
# ----------------------------
with torch.no_grad():
    output = model(img_tensor).last_hidden_state

# Create flood map
flood_map = output[0, 1:, :].mean(dim=1).reshape(14, 14)

# Normalize (0–1)
flood_map = (flood_map - flood_map.min()) / (flood_map.max() - flood_map.min() + 1e-8)

# ----------------------------
# DISPLAY
# ----------------------------
st.header("🛰️ Satellite Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Satellite Input")
    st.image(img[0], width=350)

with col2:
    st.subheader("Flood Risk Heatmap")
    fig, ax = plt.subplots()
    heat = ax.imshow(flood_map, cmap='viridis')
    plt.colorbar(heat)
    ax.axis('off')
    st.pyplot(fig)

st.markdown("---")

# ----------------------------
# METRICS
# ----------------------------
confidence = flood_map.mean().item()
flood_area = (flood_map > 0.6).float().mean().item()

st.header("📊 Analysis Metrics")

col1, col2 = st.columns(2)
col1.metric("Flood Confidence", f"{confidence:.2f}")
col2.metric("Flood Area", f"{flood_area*100:.2f}%")

st.markdown("---")

# ----------------------------
# DECISION
# ----------------------------
st.header("🚀 Satellite Decision")

if confidence > 0.6:
    st.success(f"High flood probability ({confidence:.2f}). Region prioritized for transmission.")
else:
    st.warning(f"Low flood probability ({confidence:.2f}). Region skipped to conserve bandwidth.")

st.markdown("---")

# ----------------------------
# MAP (FIXED)
# ----------------------------
st.header("🗺️ Flood Region Map")

lat_min, lat_max = 26.0, 27.0
lon_min, lon_max = 91.0, 92.0

if confidence > 0.7:
    color = "red"
elif confidence > 0.4:
    color = "orange"
else:
    color = "blue"

m = folium.Map(
    location=[26.5, 91.5],
    zoom_start=8,
    control_scale=True
)

folium.Rectangle(
    bounds=[[lat_min, lon_min], [lat_max, lon_max]],
    color=color,
    fill=True,
    fill_opacity=0.4
).add_to(m)

# Force correct zoom
m.fit_bounds([[lat_min, lon_min], [lat_max, lon_max]])

st_folium(m, width=700)

st.markdown("---")

# ----------------------------
# TILE GRID
# ----------------------------
st.header("🧩 Tile Grid Visualization")

tiles = []
tile_scores = []

tile_size = 7
idx = 0

for i in range(0, 14, tile_size):
    for j in range(0, 14, tile_size):
        tile = flood_map[i:i+tile_size, j:j+tile_size]
        score = tile.mean().item()
        tiles.append(tile)
        tile_scores.append((idx, score))
        idx += 1

# Sort tiles
tile_scores = sorted(tile_scores, key=lambda x: x[1], reverse=True)

cols = st.columns(4)

tiles_sent = 0  # ✅ FIXED

for rank, (idx, score) in enumerate(tile_scores[:4]):
    with cols[rank]:
        st.write(f"Rank {rank+1} | Tile {idx}")
        st.write(f"Score: {score:.2f}")

        fig, ax = plt.subplots()
        ax.imshow(tiles[idx], cmap='viridis')
        ax.axis('off')
        st.pyplot(fig)

        if score > 0.4:
            st.success("SEND 🚀")
            tiles_sent += 1  # ✅ IMPORTANT FIX
        else:
            st.warning("SKIP ❌")

st.markdown("---")

# ----------------------------
# BANDWIDTH (FIXED)
# ----------------------------
st.header("📡 Bandwidth Optimization")

total_tiles = 4

bandwidth_saved = ((total_tiles - tiles_sent) / total_tiles) * 100

col1, col2, col3 = st.columns(3)
col1.metric("Total Tiles", total_tiles)
col2.metric("Tiles Sent", tiles_sent)
col3.metric("Bandwidth Saved", f"{bandwidth_saved:.2f}%")

# ----------------------------
# EXPLANATION
# ----------------------------
st.header("🧠 AI Decision Explanation")

if confidence > 0.6:
    st.success("High flood probability detected. Critical regions prioritized for response.")
else:
    st.warning("Low flood probability. Data skipped to optimize bandwidth.")