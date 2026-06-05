"""Face recognition for known / unknown classification (Stage 2).

Primary backend: **ArcFace** (InsightFace `buffalo_l`) - SOTA face embeddings
(~99.x% on LFW). Each enrolled person is represented by the mean of their photo
embeddings; a detected face is matched by cosine similarity, and a similarity
above `sim_threshold` is treated as that known person, else "Unknown".

Fallback backend: OpenCV LBPH (cv2.face) when InsightFace isn't installed.

Enrollment layout:
    known_faces/
        alice/  img1.jpg img2.jpg ...
        bob/    img1.jpg ...
"""
import os
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore")

try:
    from insightface.app import FaceAnalysis
    _HAS_ARCFACE = True
except ImportError:                       # pragma: no cover
    _HAS_ARCFACE = False

_HAS_LBPH = hasattr(cv2, "face")


class FaceRecognizer:
    def __init__(self, known_dir: str = "known_faces", tolerance: float = 70.0,
                 sim_threshold: float = 0.40):
        self.known_dir = known_dir
        self.tolerance = tolerance            # LBPH distance (fallback backend)
        self.sim_threshold = sim_threshold    # ArcFace cosine similarity
        self.labels: dict[int, str] = {}
        self.trained = False
        self.backend = "arcface" if _HAS_ARCFACE else ("lbph" if _HAS_LBPH else "none")
        self.available = self.backend != "none"

        if self.backend == "arcface":
            self.app = FaceAnalysis(name="buffalo_l",
                                    providers=["CPUExecutionProvider"])
            self.app.prepare(ctx_id=-1, det_size=(640, 640))
            self.gallery: dict[str, np.ndarray] = {}   # name -> mean unit embedding
        elif self.backend == "lbph":
            self.detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            self.recognizer = None

    # ---------------------------------------------------------------- train
    def train(self) -> bool:
        if not self.available or not os.path.isdir(self.known_dir):
            return False
        if self.backend == "arcface":
            return self._train_arcface()
        return self._train_lbph()

    def _train_arcface(self) -> bool:
        self.gallery = {}
        for name in sorted(os.listdir(self.known_dir)):
            pdir = os.path.join(self.known_dir, name)
            if not os.path.isdir(pdir):
                continue
            embs = []
            for fn in os.listdir(pdir):
                img = cv2.imread(os.path.join(pdir, fn))
                if img is None:
                    continue
                faces = self.app.get(img)
                if faces:                     # use the most confident face
                    f = max(faces, key=lambda x: x.det_score)
                    embs.append(f.normed_embedding)
            if embs:
                mean = np.mean(embs, axis=0)
                self.gallery[name] = mean / (np.linalg.norm(mean) + 1e-9)
        self.trained = bool(self.gallery)
        return self.trained

    def _train_lbph(self) -> bool:
        faces, ids, self.labels = [], [], {}
        nid = 0
        for name in sorted(os.listdir(self.known_dir)):
            pdir = os.path.join(self.known_dir, name)
            if not os.path.isdir(pdir):
                continue
            self.labels[nid] = name
            for fn in os.listdir(pdir):
                img = cv2.imread(os.path.join(pdir, fn), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                for (x, y, w, h) in self.detector.detectMultiScale(img, 1.1, 5):
                    faces.append(cv2.resize(img[y:y + h, x:x + w], (200, 200)))
                    ids.append(nid)
            nid += 1
        if not faces:
            return False
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.train(faces, np.array(ids))
        self.trained = True
        return True

    # ------------------------------------------------------------- identify
    def identify(self, frame):
        """Return list of {box, name, known, confidence} for visible faces."""
        if self.backend == "arcface":
            return self._identify_arcface(frame)
        if self.backend == "lbph":
            return self._identify_lbph(frame)
        return []

    def _identify_arcface(self, frame):
        out = []
        for f in self.app.get(frame):
            x1, y1, x2, y2 = (int(v) for v in f.bbox)
            name, known, sim = "Unknown", False, 0.0
            if self.trained:
                emb = f.normed_embedding
                best = max(self.gallery.items(),
                           key=lambda kv: float(emb @ kv[1]), default=None)
                if best:
                    sim = float(emb @ best[1])
                    if sim >= self.sim_threshold:
                        name, known = best[0], True
            out.append({"box": (x1, y1, x2, y2), "name": name,
                        "known": known, "confidence": round(sim, 3)})
        return out

    def _identify_lbph(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out = []
        for (x, y, w, h) in self.detector.detectMultiScale(
                gray, 1.1, 5, minSize=(60, 60)):
            name, known, conf = "Unknown", False, None
            if self.trained:
                roi = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
                lid, dist = self.recognizer.predict(roi)
                conf = float(dist)
                if dist <= self.tolerance:
                    name, known = self.labels.get(lid, "Unknown"), True
            out.append({"box": (x, y, x + w, y + h), "name": name,
                        "known": known, "confidence": conf})
        return out
