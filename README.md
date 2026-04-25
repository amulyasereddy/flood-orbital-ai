# TerraMind Orbital Flood Intelligence System
**Team 418 | AI/ML in Space Track | TM2Space × Aeon Hackathon**

---

## What problem are we solving?

Monsoon flooding in India (Assam, Kerala, Chennai) destroys crops, displaces millions, and costs insurers billions every year. Emergency responders need flood extent maps within **hours**, not days. The bottleneck today is bandwidth: a raw Sentinel-1 SAR tile is ~300MB; downlinking it from a satellite costs precious time and money.

**Our system runs flood detection directly on the satellite** using TerraMind. The satellite scores each tile and downlinks only the result — a flood mask, an area estimate in km², and a downlink decision. A tile that scores "no flood" is never sent. We estimate **80%+ bandwidth savings** while catching **>85% of actual flood events**.

Customer: NDMA, state disaster response forces, crop insurance companies, satellite operators (TM2Space).

---

## What did we build?

**Architecture:** TerraMind-small (frozen encoder, `S1GRD` modality) + custom `FloodSegHead` decoder fine-tuned on Sen1Floods11.

**Pipeline:**
1. Sentinel-1 SAR tile (2-channel VV/VH) → TerraMind encoder → 196 × 768 patch embeddings
2. FloodSegHead (UNet-style decoder) → 224×224 per-pixel flood probability map
3. **TiM LULC overlay** — TerraMind Thinking-in-Modalities generates synthetic land-use classification to answer: *what type of land got flooded?* (farmland / residential / forest / roads)
4. **Orbital triage** — confidence score gates downlink decision, simulating on-orbit inference

**Dataset:** Sen1Floods11 (446 globally distributed flood event tiles with hand-labeled masks)

**Training:** TerraMind encoder frozen; only head trained. 20 epochs, AdamW, cosine LR, weighted BCE (positive class weight 3.0 for class imbalance).

---

## How did we measure it?

Evaluated on Sen1Floods11 test split:

| Metric     | Baseline (OTSU) | TerraMind (ours) | Delta   |
|------------|-----------------|------------------|---------|
| mIoU       | 0.312           | **0.681**        | +118%   |
| F1         | 0.447           | **0.789**        | +76%    |
| Precision  | —               | 0.812            | —       |
| Recall     | —               | 0.768            | —       |

Orbital triage simulation (20 test tiles, top-20% downlink):
- Downlinked: 4 / 20 tiles (20%)
- Flood events caught: 87%
- Bandwidth saved: **80%**

---

## What's the orbital compute story?

| Component                   | Value                        |
|-----------------------------|------------------------------|
| TerraMind-small encoder     | ~22M parameters              |
| FloodSegHead                | ~1.2M parameters             |
| Total model size            | ~90 MB (ONNX export)         |
| Inference time (V100)       | ~180ms per tile              |
| Target platform             | Nvidia Jetson Orin NX (16GB) |
| Estimated on-orbit latency  | ~2–4s per tile               |

TerraMind-tiny is available for further size reduction (fits Jetson AGX without Orin). The frozen encoder means only the small head needs to run on every new tile post-deployment.

---

## What doesn't work yet?

- **TiM LULC is approximated** — full `terramind_v1_small_tim` fine-tuning requires 6–8h on A100; we use a SAR backscatter proxy for the demo
- **Cloud-free assumption** — SAR is cloud-independent but our LULC proxy degrades over dense urban areas
- **Single-tile inference** — multi-temporal change detection (before/after flood) is not yet implemented
- **Sen1Floods11 distribution bias** — model trained on global data; India-specific AOIs may need additional fine-tuning on regional data

---

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test TerraMind loads correctly
python model_inference.py

# 3. Download Sen1Floods11
#    See notebooks/download_data.ipynb

# 4. Train the flood head
python train.py --data_dir data/sen1floods11 --epochs 20

# 5. Evaluate vs baseline
python eval.py --checkpoint models/flood_head_best.pt

# 6. Run inference on a single tile
python infer.py --input sample.tif

# 7. Launch demo app
streamlit run app.py
```

**Dataset:** [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) — do not commit raw data to repo  
**Weights:** `models/flood_head_best.pt` — hosted externally if >200MB

---

## File structure

```
418_HACKATHON/
├── app.py                ← Streamlit demo (main entry point)
├── flood_model.py        ← TerraMind encoder + FloodSegHead
├── train.py              ← Fine-tuning on Sen1Floods11
├── eval.py               ← mIoU / F1 vs OTSU baseline
├── infer.py              ← CLI inference (submission requirement)
├── preprocess.py         ← SAR loading + TerraMind normalisation
├── triage.py             ← Orbital triage simulator
├── multi_tile.py         ← Batch tile scoring
├── visualize.py          ← Publication figures
├── model_inference.py    ← Pipeline smoke test
├── requirements.txt      ← Pinned dependencies
└── sample.tif            ← Sample SAR tile for testing
```