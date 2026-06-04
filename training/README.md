# Training a custom weapon-detection model

Trains a YOLOv8 model on a multi-class weapon dataset from Roboflow Universe,
using the Mac's Apple-GPU (MPS) acceleration.

## 1. Get a Roboflow API key (free, ~2 min)
1. Sign up / log in at https://roboflow.com
2. Settings -> API -> copy your **Private API Key**
3. Put it in the project `.env` file:
   ```
   ROBOFLOW_API_KEY=your_key_here
   ```

## 2. Download + build the dataset

**Combined (recommended):** downloads both the 28-class and 14-class datasets
and merges them into one unified, de-duplicated multi-class dataset at
`training/datasets/weapons/`:
```bash
cd training && ../.venv/bin/python build_dataset.py && cd ..
```

**Single dataset** instead:
```bash
.venv/bin/python training/download_data.py            # 28-class default
RF_WORKSPACE=rhackathon RF_PROJECT=weapon-detection-aoxpz \
    .venv/bin/python training/download_data.py        # a different one
```

The merge collapses exact duplicate classes (e.g. `Knife`/`knife`) while keeping
granular ones (`ak`, `m16`, `revolver`, `spear`, ...), and re-indexes every
annotation to the unified class list.

## 3. Train
```bash
.venv/bin/python training/train.py                 # quick proof (~minutes)
QUICK=0 EPOCHS=80 .venv/bin/python training/train.py   # full run (hours)

# the trained "coarse" model below used:
QUICK=0 FRACTION=0.6 IMGSZ=448 EPOCHS=12 OPTIMIZER=SGD LR0=0.01 \
    .venv/bin/python training/train.py
```
Best weights are copied to `models/weapon_custom.pt`. The app auto-uses that
file for weapon detection if it exists (falling back to the simple firearm
model + COCO otherwise).

Outputs (curves, confusion matrix, sample predictions) land in
`training/runs/weapons/` (under `runs/detect/...`).

## What actually happened (results + lessons)

- **Specific (15 classes)** rebuilt from both datasets was too hard for a nano
  model: `ak`/`m16`/`rifle`/`semi automatic` look near-identical and each had
  only ~450 images, so recall stayed near zero. Build it with
  `CANON_MODE=specific` if you want to try a bigger model.
- **Coarse (6 classes:** firearm, knife, axe, cleaver, sword, spear**)** is what
  ships. Validation mAP looks low (~0.12) because it is measured at near-zero
  confidence on a noisy community val set, but at the app's operating threshold
  (conf 0.60) the model is a **clean firearm detector**: handgun 0.90, rifle
  0.86, and **zero false positives** across 119 no-weapon pedestrian-video
  frames (max false confidence 0.53).
- **Weakness:** the firearm class dominates (16k vs ~500 per melee class), so
  melee is weak (a knife may be labelled `firearm`). The app keeps COCO on for
  reliable knife / baseball bat / scissors labels alongside the custom firearm
  detector. To improve melee, balance the classes or add melee images.

To revert to the pre-training detector, just delete `models/weapon_custom.pt`.
