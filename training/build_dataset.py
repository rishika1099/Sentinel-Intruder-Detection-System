"""Download the two weapon datasets from Roboflow Universe and merge them into
one unified multi-class dataset at training/datasets/weapons/.

Needs ROBOFLOW_API_KEY (env or .env). Run once, then train with train.py.
"""
import os
import sys

from dotenv import load_dotenv
from roboflow import Roboflow

from merge_datasets import merge

load_dotenv()
API_KEY = os.environ.get("ROBOFLOW_API_KEY")
if not API_KEY:
    sys.exit("ERROR: set ROBOFLOW_API_KEY (roboflow.com -> Settings -> API key).")

# (workspace, project) pairs -> local download dir
DATASETS = [
    ("yolov7test-u13vc", "weapon-detection-m7qso", "training/datasets/ds_28"),
    ("rhackathon", "weapon-detection-aoxpz", "training/datasets/ds_14"),
]

# Curation map: normalised raw class name -> canonical class (or omit to DROP).
# Cleans dataset 1's messy synonyms + removes non-weapon/scene classes, and
# keeps dataset 2's specific weapon types.
CANON = {
    # dataset 1 (coarse, high-volume) firearm classes -> kept as generic labels
    "gun": "gun",                 # ~2.6k boxes (handguns/pistols)
    "heavy weapon": "heavy weapon",  # ~10.7k boxes (long guns)
    # dataset 2 (specific) firearm classes
    "pistol": "pistol", "pistols": "pistol", "handgun": "pistol",
    "revolver": "revolver", "rifle": "rifle",
    "ak": "ak", "m16": "m16", "semi automatic": "semi automatic",
    "shotgun": "shotgun",
    # bladed / melee
    "knife": "knife", "cleaver": "cleaver", "cutter": "cutter",
    "short sword": "short sword", "long sword": "long sword",
    "spear": "spear", "ax": "ax",
    # everything else (eto junk, plus any scene/non-weapon labels) -> dropped
}

# Coarse map: collapse to a few visually-distinct, learnable classes. Much
# higher accuracy on a nano model than telling ak/m16/rifle apart.
COARSE_CANON = {
    # all firearms -> one class
    "gun": "firearm", "heavy weapon": "firearm", "pistol": "firearm",
    "handgun": "firearm", "revolver": "firearm", "rifle": "firearm",
    "ak": "firearm", "m16": "firearm", "semi automatic": "firearm",
    "shotgun": "firearm", "guns": "firearm", "weapon": "firearm",
    # bladed
    "knife": "knife", "cutter": "knife", "cleaver": "cleaver",
    "short sword": "sword", "long sword": "sword",
    # other melee
    "spear": "spear", "ax": "axe",
}

# Melee-only map for a dedicated blade/knife model (no firearm class to drown
# out knives). Use with drop_empty=True so firearm-only images are excluded.
MELEE_CANON = {
    "knife": "knife", "cutter": "knife", "cleaver": "knife",
    "short sword": "sword", "long sword": "sword",
    "spear": "spear", "ax": "axe",
}

# Single-class "weapon present" map: collapse EVERY weapon type (firearm + melee)
# to one "weapon" class for a high-recall surveillance detector. Drops person,
# hand, phone, ruler, fog, etc. by omission.
WEAPON_PRESENT_CANON = {
    "gun": "weapon", "heavy weapon": "weapon", "pistol": "weapon",
    "handgun": "weapon", "revolver": "weapon", "rifle": "weapon", "ak": "weapon",
    "m16": "weapon", "semi automatic": "weapon", "shotgun": "weapon",
    "guns": "weapon", "weapon": "weapon",
    "knife": "weapon", "cutter": "weapon", "cleaver": "weapon",
    "short sword": "weapon", "long sword": "weapon", "spear": "weapon",
    "ax": "weapon",
}

# pick via env: CANON_MODE=coarse (default) or specific
CANON_BY_MODE = {"coarse": COARSE_CANON, "specific": CANON}


def latest_version(project):
    nums = []
    for v in project.versions():
        try:
            nums.append(int(getattr(v, "version", str(v).rstrip("/").split("/")[-1])))
        except (ValueError, TypeError):
            continue
    return max(nums) if nums else 1


def main():
    rf = Roboflow(api_key=API_KEY)
    dirs = []
    for ws, proj, out in DATASETS:
        print(f"\n=== {ws}/{proj} ===")
        project = rf.workspace(ws).project(proj)
        ver = latest_version(project)
        print(f"latest version: {ver} -> {out}")
        project.version(ver).download("yolov8", location=out, overwrite=True)
        dirs.append(out)

    print("\n=== merging + curating classes ===")
    data_yaml = merge(dirs, out="training/datasets/weapons", canon=CANON)
    print("\nReady to train:", data_yaml)


if __name__ == "__main__":
    main()
