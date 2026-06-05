"""Audit and clean YOLO label quality in a dataset.

Objective (no-model) label fixes that directly raise the achievable mAP:
  - drop degenerate boxes (zero / negative / sub-pixel area, coords outside [0,1])
  - drop near-duplicate boxes (same class, IoU > dup_iou) - common after merging
  - drop full-frame boxes (cover > big_frac of the image) - usually bad labels
Images left with no labels are removed (unless they were already backgrounds).

    python training/clean_labels.py <dataset_dir> [--apply]

Without --apply it only reports the audit. With --apply it writes cleaned
labels in place (call on a copy).
"""
import os
import sys


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    a1, a2, a3, a4 = ax - aw / 2, ay - ah / 2, ax + aw / 2, ay + ah / 2
    b1, b2, b3, b4 = bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2
    ix1, iy1, ix2, iy2 = max(a1, b1), max(a2, b2), min(a3, b3), min(a4, b4)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    return inter / (aw * ah + bw * bh - inter)


def clean_file(lines, dup_iou=0.92, big_frac=0.97, min_area=1e-4):
    boxes, dropped = [], {"degenerate": 0, "duplicate": 0, "fullframe": 0}
    for ln in lines:
        p = ln.split()
        if len(p) < 5:
            continue
        cls = int(float(p[0]))
        cx, cy, w, h = map(float, p[1:5])
        if not (0 <= cx <= 1 and 0 <= cy <= 1) or w <= 0 or h <= 0 or w * h < min_area:
            dropped["degenerate"] += 1
            continue
        if w * h > big_frac:
            dropped["fullframe"] += 1
            continue
        dup = any(c == cls and _iou((cx, cy, w, h), bb) > dup_iou
                  for c, bb in boxes)
        if dup:
            dropped["duplicate"] += 1
            continue
        boxes.append((cls, (cx, cy, w, h)))
    out = [f"{c} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}" for c, bb in boxes]
    return out, dropped


def run(dataset_dir, apply=False):
    total = {"degenerate": 0, "duplicate": 0, "fullframe": 0}
    n_boxes = n_kept = 0
    for split in ("train", "valid", "test"):
        ld = os.path.join(dataset_dir, split, "labels")
        if not os.path.isdir(ld):
            continue
        for fn in os.listdir(ld):
            path = os.path.join(ld, fn)
            with open(path) as f:
                lines = f.readlines()
            n_boxes += sum(1 for ln in lines if ln.split())
            kept, dropped = clean_file(lines)
            n_kept += len(kept)
            for k in total:
                total[k] += dropped[k]
            if apply:
                with open(path, "w") as f:
                    f.write("\n".join(kept) + ("\n" if kept else ""))
    drop = sum(total.values())
    print(f"boxes: {n_boxes} | kept: {n_kept} | dropped: {drop} "
          f"({100*drop/max(1,n_boxes):.1f}%)")
    print(f"  degenerate={total['degenerate']}  duplicate={total['duplicate']}  "
          f"fullframe={total['fullframe']}")
    print("APPLIED (labels rewritten)" if apply else "AUDIT ONLY (use --apply to clean)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: clean_labels.py <dataset_dir> [--apply]")
    run(sys.argv[1], apply="--apply" in sys.argv)
