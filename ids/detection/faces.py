"""Face recognition for known / unknown classification (Stage 2).

Uses OpenCV only (no dlib): a Haar cascade locates faces and an LBPH
recognizer matches them against people enrolled under `known_faces/`.

Enrollment layout:
    known_faces/
        alice/  img1.jpg img2.jpg ...
        bob/    img1.jpg ...

Each subfolder name is the person's label. Call `train()` once after adding
photos. LBPH returns a distance (lower = more similar); a match closer than
`tolerance` is treated as that known person, otherwise "Unknown".

LBPH lives in cv2.face, which ships with opencv-contrib-python. If only the
plain opencv-python build is installed, recognition is disabled gracefully and
every detected face is reported as "Unknown".
"""
import os
import cv2
import numpy as np

_HAS_FACE = hasattr(cv2, "face")


class FaceRecognizer:
    def __init__(self, known_dir: str = "known_faces", tolerance: float = 70.0):
        self.known_dir = known_dir
        self.tolerance = tolerance
        self.detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.recognizer = None
        self.labels: dict[int, str] = {}
        self.trained = False
        self.available = _HAS_FACE

    def train(self) -> bool:
        """Train LBPH from images in `known_dir`. Returns True if trained."""
        if not self.available or not os.path.isdir(self.known_dir):
            return False

        faces, ids, self.labels = [], [], {}
        next_id = 0
        for name in sorted(os.listdir(self.known_dir)):
            person_dir = os.path.join(self.known_dir, name)
            if not os.path.isdir(person_dir):
                continue
            label_id = next_id
            self.labels[label_id] = name
            next_id += 1
            for fn in os.listdir(person_dir):
                img = cv2.imread(os.path.join(person_dir, fn),
                                 cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                for (x, y, w, h) in self.detector.detectMultiScale(img, 1.1, 5):
                    faces.append(cv2.resize(img[y:y + h, x:x + w], (200, 200)))
                    ids.append(label_id)

        if not faces:
            return False

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.train(faces, np.array(ids))
        self.trained = True
        return True

    def identify(self, frame):
        """Return list of {box, name, known, confidence} for visible faces."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out = []
        for (x, y, w, h) in self.detector.detectMultiScale(
                gray, 1.1, 5, minSize=(60, 60)):
            name, known, conf = "Unknown", False, None
            if self.trained:
                roi = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
                label_id, dist = self.recognizer.predict(roi)
                conf = float(dist)
                if dist <= self.tolerance:
                    name, known = self.labels.get(label_id, "Unknown"), True
            out.append({"box": (x, y, x + w, y + h), "name": name,
                        "known": known, "confidence": conf})
        return out
