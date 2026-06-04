"""Merge several YOLOv8 datasets (different class taxonomies) into one.

Builds a unified class list (case-normalised union, preserving granular names
like 'ak'/'m16'/'revolver' while collapsing exact duplicates such as
'Knife' vs 'knife'), then copies every image and rewrites each label file so the
class indices point at the unified list.

    from merge_datasets import merge
    merge(["training/datasets/ds_28", "training/datasets/ds_14"],
          out="training/datasets/weapons")
"""
import os
import shutil
import sys

import yaml

SPLITS = ["train", "valid", "test"]


def _norm(name: str) -> str:
    return " ".join(name.strip().lower().replace("_", " ").replace("-", " ").split())


def _load_names(ds_dir: str):
    with open(os.path.join(ds_dir, "data.yaml")) as f:
        data = yaml.safe_load(f)
    names = data["names"]
    if isinstance(names, dict):                 # {0: 'a', 1: 'b'}
        names = [names[k] for k in sorted(names, key=int)]
    return names                                # list indexed by local class id


def merge(dirs, out="training/datasets/weapons", canon=None, drop_empty=False):
    """Merge datasets. If `canon` is given (dict: normalised raw name ->
    canonical class name, or None/absent = drop that class), it curates the
    class space; otherwise it just case-normalises and keeps everything.
    If drop_empty=True, images with no labels after curation are skipped (e.g.
    to build a melee-only set without thousands of firearm-only backgrounds)."""
    # 1) build unified class list + per-dataset local-id -> global-id (or None)
    global_ids: dict[str, int] = {}
    display: list[str] = []
    per_ds_map: list[dict[int, int | None]] = []
    dropped: set[str] = set()
    for d in dirs:
        names = _load_names(d)
        m: dict[int, int | None] = {}
        for i, nm in enumerate(names):
            key = _norm(nm)
            if canon is not None:
                target = canon.get(key)
                if not target:                  # None or missing -> drop
                    m[i] = None
                    dropped.add(key)
                    continue
                ckey = _norm(target)
            else:
                ckey = key
            if ckey not in global_ids:
                global_ids[ckey] = len(display)
                display.append(ckey)
            m[i] = global_ids[ckey]
        per_ds_map.append(m)

    print(f"Unified class list ({len(display)}): {display}")
    if dropped:
        print(f"Dropped {len(dropped)} non-weapon/unmapped classes: {sorted(dropped)}")

    # 2) copy images + remap labels
    for split in SPLITS:
        os.makedirs(os.path.join(out, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(out, split, "labels"), exist_ok=True)

    counts = {s: 0 for s in SPLITS}
    for di, d in enumerate(dirs):
        tag = f"ds{di}"
        local_to_global = per_ds_map[di]
        for split in SPLITS:
            img_dir = os.path.join(d, split, "images")
            lbl_dir = os.path.join(d, split, "labels")
            if not os.path.isdir(img_dir):
                continue
            for fn in os.listdir(img_dir):
                stem, ext = os.path.splitext(fn)
                new_stem = f"{tag}_{stem}"
                src_lbl = os.path.join(lbl_dir, stem + ".txt")
                dst_lbl = os.path.join(out, split, "labels", new_stem + ".txt")
                n_boxes = _remap_label(src_lbl, dst_lbl, local_to_global) \
                    if os.path.exists(src_lbl) else 0
                if drop_empty and n_boxes == 0:
                    if os.path.exists(dst_lbl):
                        os.remove(dst_lbl)          # skip firearm-only/background
                    continue
                if not os.path.exists(dst_lbl):
                    open(dst_lbl, "w").close()      # background image
                shutil.copy(os.path.join(img_dir, fn),
                            os.path.join(out, split, "images", new_stem + ext))
                counts[split] += 1

    # 3) write merged data.yaml
    data = {
        "path": os.path.abspath(out),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(display),
        "names": display,
    }
    with open(os.path.join(out, "data.yaml"), "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    print(f"Merged images per split: {counts}")
    print(f"Wrote {os.path.join(out, 'data.yaml')}")
    return os.path.join(out, "data.yaml")


def _remap_label(src, dst, mapping):
    lines = []
    with open(src) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            cls = int(float(parts[0]))
            gid = mapping.get(cls)
            if gid is None:                 # dropped class -> skip this box
                continue
            parts[0] = str(gid)
            lines.append(" ".join(parts))
    with open(dst, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: merge_datasets.py <out_dir> <ds_dir1> <ds_dir2> ...")
    merge(sys.argv[2:], out=sys.argv[1])
