"""Enrollment helpers: capture face photos into known_faces/<name>/."""
import os
import re

import cv2
import numpy as np

KNOWN_DIR = "known_faces"
_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def _safe_name(name: str) -> str:
    """Turn a display name into a filesystem-safe folder name."""
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_")
    return slug


def person_dir(name: str) -> str:
    return os.path.join(KNOWN_DIR, _safe_name(name))


def list_people() -> dict[str, int]:
    """Return {person_name: photo_count} for everyone already enrolled."""
    people = {}
    if not os.path.isdir(KNOWN_DIR):
        return people
    for entry in sorted(os.listdir(KNOWN_DIR)):
        d = os.path.join(KNOWN_DIR, entry)
        if os.path.isdir(d):
            n = len([f for f in os.listdir(d)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))])
            people[entry] = n
    return people


def count_faces(image_bgr) -> int:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return len(_detector.detectMultiScale(gray, 1.1, 5, minSize=(60, 60)))


def decode_image(file_bytes: bytes):
    """Decode raw image bytes (from st.camera_input/upload) to a BGR frame."""
    arr = np.frombuffer(file_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def save_face(name: str, image_bgr, require_face: bool = True):
    """Save one photo for `name`. Returns (saved: bool, path_or_reason: str)."""
    if not _safe_name(name):
        return False, "enter a valid name first"
    if require_face and count_faces(image_bgr) == 0:
        return False, "no face detected, try again facing the camera"

    d = person_dir(name)
    os.makedirs(d, exist_ok=True)
    idx = 1
    while os.path.exists(os.path.join(d, f"{idx:03d}.jpg")):
        idx += 1
    path = os.path.join(d, f"{idx:03d}.jpg")
    cv2.imwrite(path, image_bgr)
    return True, path
