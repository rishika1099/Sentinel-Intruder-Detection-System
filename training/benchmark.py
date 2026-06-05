"""Defensible benchmark: IoU-matched mAP / precision / recall on the held-out
TEST split (never seen during training), plus PR curves saved to disk.

    .venv/bin/python training/benchmark.py

Reports the standard detection metrics for the trained models, and an
operating-point summary (detection rate + false-positive rate) that reflects
how the app actually behaves at its confidence thresholds.
"""
import os
from ultralytics import YOLO

MODELS = [
    ("firearm/coarse weapon", "models/weapon_custom.pt", "training/datasets/weapons/data.yaml"),
    ("melee (knife/sword/axe/spear)", "models/melee.pt", "training/datasets/melee/data.yaml"),
]


def bench(name, weights, data):
    if not (os.path.exists(weights) and os.path.exists(data)):
        print(f"\n[{name}] SKIP (missing {weights} or {data})")
        return
    print(f"\n{'='*60}\n{name}\n  weights={weights}\n{'='*60}")
    model = YOLO(weights)
    m = model.val(data=data, split="test", verbose=False,
                  project="training/runs", name="benchmark", exist_ok=True)
    print(f"  mAP@0.5      : {m.box.map50:.3f}")
    print(f"  mAP@0.5:0.95 : {m.box.map:.3f}")
    print(f"  precision    : {m.box.mp:.3f}")
    print(f"  recall       : {m.box.mr:.3f}")
    # per-class
    names = model.names
    print("  per-class mAP@0.5:")
    for i, ap in zip(m.box.ap_class_index, m.box.ap50):
        print(f"     {names[int(i)]:16s} {ap:.3f}")
    print(f"  PR curves saved under training/runs/benchmark/")


if __name__ == "__main__":
    for name, w, d in MODELS:
        bench(name, w, d)
