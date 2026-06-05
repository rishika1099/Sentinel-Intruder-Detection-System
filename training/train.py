"""Train a YOLOv8 weapon-detection model on the M4 GPU (MPS).

Quick proof (default): few epochs on a fraction of the data to validate the
whole pipeline end-to-end in minutes.

    .venv/bin/python training/train.py            # quick proof
    QUICK=0 EPOCHS=80 .venv/bin/python training/train.py   # full run

Env vars:
    DATA_YAML   path to dataset data.yaml (default training/datasets/weapons/data.yaml)
    QUICK       1 = quick proof (default), 0 = full run
    EPOCHS      override epoch count
    IMGSZ       image size (default 512 quick / 640 full)
    BATCH       batch size (default 16)
    DEVICE      mps (Apple GPU, default) | cpu
    BASE_MODEL  starting weights (default yolov8n.pt)

The best weights are copied to models/weapon_custom.pt for the app to use.
"""
import os
import shutil

from ultralytics import YOLO

DATA = os.environ.get("DATA_YAML", "training/datasets/weapons/data.yaml")
QUICK = os.environ.get("QUICK", "1") == "1"
DEVICE = os.environ.get("DEVICE", "mps")
BASE_MODEL = os.environ.get("BASE_MODEL", "yolov8n.pt")
BATCH = int(os.environ.get("BATCH", "16"))

if QUICK:
    EPOCHS = int(os.environ.get("EPOCHS", "10"))
    IMGSZ = int(os.environ.get("IMGSZ", "512"))
    FRACTION = float(os.environ.get("FRACTION", "0.15"))
else:
    EPOCHS = int(os.environ.get("EPOCHS", "80"))
    IMGSZ = int(os.environ.get("IMGSZ", "640"))
    FRACTION = float(os.environ.get("FRACTION", "1.0"))


def main():
    if not os.path.exists(DATA):
        raise SystemExit(f"{DATA} not found. Run training/download_data.py first.")

    print(f"Training {'QUICK PROOF' if QUICK else 'FULL'}: epochs={EPOCHS} "
          f"imgsz={IMGSZ} fraction={FRACTION} device={DEVICE} batch={BATCH}")

    # optional explicit LR / optimizer (auto-LR can be far too low on small data)
    OPTIMIZER = os.environ.get("OPTIMIZER", "auto")
    LR0 = os.environ.get("LR0")
    extra = {"optimizer": OPTIMIZER}
    if LR0:
        extra["lr0"] = float(LR0)

    model = YOLO(BASE_MODEL)
    model.train(
        data=DATA, epochs=EPOCHS, imgsz=IMGSZ, device=DEVICE, batch=BATCH,
        fraction=FRACTION, project="training/runs",
        name=os.environ.get("NAME", "weapons"),
        exist_ok=True, patience=30, plots=True, **extra,
    )

    # locate best.pt robustly via the trainer (save dir varies)
    best = getattr(model.trainer, "best", None)
    dest = os.environ.get("DEST", os.path.join("models", "weapon_custom.pt"))
    if best and os.path.exists(best):
        os.makedirs("models", exist_ok=True)
        shutil.copy(best, dest)
        print(f"\nBest weights copied to {dest}")
        print("Classes:", YOLO(dest).names)
    else:
        print(f"\nWARNING: best.pt not found at {best}")


if __name__ == "__main__":
    main()
