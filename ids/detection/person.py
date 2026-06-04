"""Person detection with YOLOv8 (class 0 = person)."""
from ultralytics import YOLO


class PersonDetector:
    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.40):
        # yolov8n.pt downloads automatically on first use (~6 MB).
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame):
        """Return a list of {box: (x1,y1,x2,y2), conf: float} for people."""
        results = self.model.predict(frame, conf=self.conf, classes=[0],
                                     verbose=False)
        boxes = []
        for r in results:
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                boxes.append({
                    "box": (int(x1), int(y1), int(x2), int(y2)),
                    "conf": float(b.conf[0]),
                })
        return boxes
