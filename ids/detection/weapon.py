"""Weapon detection.

Prefers a custom multi-class model trained on Roboflow data
(`models/weapon_custom.pt`, e.g. classes Pistol / Rifle / Shotgun / Knife / ...)
when present. Otherwise falls back to a two-source setup:

  1. a small dedicated firearm model (`models/weapon.pt`) -> "firearm", plus
  2. COCO classes from the general model -> knife / baseball bat / scissors

The base COCO YOLO model has no firearm class, which is why a dedicated or
custom model is needed for guns.
"""
import os

from ultralytics import YOLO

_HERE = os.path.dirname(__file__)
CUSTOM_MODEL = os.path.join(_HERE, "..", "..", "models", "weapon_custom.pt")
FIREARM_MODEL = os.path.join(_HERE, "..", "..", "models", "weapon.pt")
MELEE_MODEL = os.path.join(_HERE, "..", "..", "models", "melee.pt")  # dedicated knife/blade
# CCTV-trained single-class "weapon present" detector (guns + knives, robust
# out-of-distribution) - preferred primary when available.
WEAPON_PRESENT_MODEL = os.path.join(_HERE, "..", "..", "models", "weapon_present.pt")

# simple firearm-model raw class -> display label
_FIREARM_LABELS = {"pistol": "firearm", "knife": "knife"}
# COCO class id -> display label (real weapon / threat objects in COCO)
_COCO_WEAPONS = {34: "baseball bat", 43: "knife", 76: "scissors"}
# classes that some weapon datasets include but are NOT weapons -> ignore
_NON_WEAPON = {"hand", "person", "human", "face", "no weapon", "background", "none"}

# Set True once the custom model is balanced enough to trust its type labels
# (firearm vs knife vs sword...). Until then we report a generic "weapon".
TRUST_CUSTOM_CLASSES = False


def _norm(name: str) -> str:
    return name.strip().lower().replace("_", " ")


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


class WeaponDetector:
    def __init__(self, conf: float = 0.60, coco_conf: float = 0.35,
                 melee_conf: float = 0.55, coco_model: str = "yolov8n.pt"):
        self.conf = conf
        self.coco_conf = coco_conf
        self.melee_conf = melee_conf

        # dedicated melee/knife model (trained separately, no firearm class to
        # drown out blades) -> reliable knife / sword / axe / spear labels
        self.melee = YOLO(MELEE_MODEL) if os.path.exists(MELEE_MODEL) else None

        if os.path.exists(WEAPON_PRESENT_MODEL):
            self.model = YOLO(WEAPON_PRESENT_MODEL)
            self.kind = "weapon_present"  # single "weapon" class, CCTV-robust
            self.use_coco = True          # COCO still adds bat/scissors
        elif os.path.exists(CUSTOM_MODEL):
            self.model = YOLO(CUSTOM_MODEL)
            self.kind = "custom"          # use the model's own class names
            # our custom model is strong on firearms but weak on melee (class
            # imbalance), so keep COCO on for reliable knife/bat/scissors labels
            self.use_coco = True
        elif os.path.exists(FIREARM_MODEL):
            self.model = YOLO(FIREARM_MODEL)
            self.kind = "firearm"
            self.use_coco = True
        else:
            self.model = None
            self.kind = "none"
            self.use_coco = True

        self.firearm_available = self.model is not None
        self.coco = YOLO(coco_model) if self.use_coco else None
        self.available = True

    def _label(self, raw: str) -> str | None:
        n = _norm(raw)
        if n in _NON_WEAPON:
            return None
        if self.kind == "firearm":
            return _FIREARM_LABELS.get(n, n)
        # custom model is class-imbalanced (firearm dominates), so it can't be
        # trusted to tell a gun from a knife. Report a generic "weapon" rather
        # than a confident wrong "firearm"; COCO supplies specific melee labels.
        if self.kind == "custom":
            return "weapon" if not TRUST_CUSTOM_CLASSES else n
        return n

    def detect(self, frame):
        """Return list of {box, label, raw, conf, source} for weapons."""
        out = []

        if self.model is not None:
            for r in self.model.predict(frame, conf=self.conf, verbose=False):
                for b in r.boxes:
                    raw = self.model.names[int(b.cls[0])]
                    label = self._label(raw)
                    if label is None:
                        continue
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    out.append({
                        "box": (int(x1), int(y1), int(x2), int(y2)),
                        "label": label, "raw": raw, "conf": float(b.conf[0]),
                        "source": self.kind,
                    })

        if self.melee is not None:
            gun_boxes = [d["box"] for d in out]   # custom-model (firearm) boxes
            for r in self.melee.predict(frame, conf=self.melee_conf, verbose=False):
                for b in r.boxes:
                    raw = self.melee.names[int(b.cls[0])]
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    box = (int(x1), int(y1), int(x2), int(y2))
                    # the melee model never saw guns, so it false-fires on them;
                    # if a "blade" overlaps a detected firearm, it's a gun -> drop
                    if any(_iou(box, gb) > 0.4 for gb in gun_boxes):
                        continue
                    out.append({
                        "box": box, "label": _norm(raw), "raw": raw,
                        "conf": float(b.conf[0]), "source": "melee",
                    })

        if self.coco is not None:
            # the melee model handles knives; let COCO cover only bat/scissors
            coco_classes = {k: v for k, v in _COCO_WEAPONS.items()
                            if not (self.melee is not None and v == "knife")}
            for r in self.coco.predict(frame, conf=self.coco_conf,
                                       classes=list(coco_classes), verbose=False):
                for b in r.boxes:
                    cid = int(b.cls[0])
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    out.append({
                        "box": (int(x1), int(y1), int(x2), int(y2)),
                        "label": _COCO_WEAPONS.get(cid, "weapon"),
                        "raw": self.coco.names[cid], "conf": float(b.conf[0]),
                        "source": "coco",
                    })

        return out
