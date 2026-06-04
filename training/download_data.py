"""Download a weapon-detection dataset from Roboflow Universe in YOLOv8 format.

Needs a free Roboflow API key (roboflow.com -> Settings -> API). Provide it via
the ROBOFLOW_API_KEY environment variable or a .env file.

Defaults to the 28-class `yolov7test/weapon-detection-m7qso` dataset (Hand,
Pistol, Knife, Rifle, Shotgun, and more). Override with env vars if you want a
different Universe dataset (copy the slugs from the dataset URL):

    RF_WORKSPACE=rhackathon RF_PROJECT=weapon-detection-aoxpz RF_VERSION=1 \\
        .venv/bin/python training/download_data.py
"""
import os
import sys

from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()

API_KEY = os.environ.get("ROBOFLOW_API_KEY")
WORKSPACE = os.environ.get("RF_WORKSPACE", "yolov7test-u13vc")
PROJECT = os.environ.get("RF_PROJECT", "weapon-detection-m7qso")
VERSION = os.environ.get("RF_VERSION")  # optional; else use latest
OUT = os.environ.get("RF_OUT", os.path.join("training", "datasets", "weapons"))

if not API_KEY:
    sys.exit("ERROR: set ROBOFLOW_API_KEY (roboflow.com -> Settings -> API key).")


def main():
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT)

    if VERSION:
        version_num = int(VERSION)
    else:
        nums = [int(getattr(v, "version", str(v).split("/")[-1]))
                for v in project.versions()]
        version_num = max(nums)
    print(f"Downloading {WORKSPACE}/{PROJECT} v{version_num} -> {OUT}")

    dataset = project.version(version_num).download("yolov8", location=OUT,
                                                    overwrite=True)
    data_yaml = os.path.join(dataset.location, "data.yaml")
    print("\nDONE. data.yaml:", data_yaml)
    with open(data_yaml) as f:
        print(f.read())


if __name__ == "__main__":
    main()
