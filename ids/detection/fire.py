"""Color + brightness based fire / flame detection.

Fire is not just "warm colored": real flames are highly saturated AND very
bright, with the red channel clearly dominant over blue, and they form one
sizeable connected region rather than scattered specks. We require all of
those together, which rejects most false positives (skin tones, wood, red
clothing, warm indoor lighting, sunsets).

This is still a heuristic. For production accuracy, drop in a trained YOLO
fire/smoke model behind this same `detect_fire` signature. The simulated
temperature + smoke sensors in the engine add a second line of defense.
"""
import cv2
import numpy as np


def detect_fire(frame, min_area_ratio: float = 0.020,
                max_edge_density: float = 0.18):
    """Return (is_fire, area_ratio, boxes, mask).

    is_fire is True only when a single connected fire-like region covers at
    least `min_area_ratio` of the frame AND that region is a coherent bright
    mass rather than a striated pattern. Real flames form a large solid glow
    with low internal edge density; sunsets and warm textured decor break into
    thin high-frequency bands, which we reject via `max_edge_density`.
    """
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # warm hues with STRONG saturation and HIGH brightness (the two ends of
    # the hue wheel that flames occupy)
    m1 = cv2.inRange(hsv, np.array([0, 150, 200]), np.array([25, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([160, 150, 200]), np.array([180, 255, 255]))
    color = cv2.bitwise_or(m1, m2)

    # red-channel dominance: flames have R > G >= B with a clear R-B gap and a
    # bright red channel. Most warm-but-not-fire objects fail this.
    b, g, r = cv2.split(blur.astype(np.int16))
    rgb_rule = ((r > 180) & (r >= g) & (g >= b) & ((r - b) > 70))
    rgb_rule = (rgb_rule.astype(np.uint8)) * 255

    mask = cv2.bitwise_and(color, rgb_rule)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # drop specks
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # fill the blob

    h, w = frame.shape[:2]
    frame_area = float(h * w)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    boxes, largest = [], 0.0
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        a = cv2.contourArea(c)
        largest = max(largest, a)
        # only box regions that are at least half the detection threshold
        if a / frame_area >= min_area_ratio * 0.5:
            x, y, bw, bh = cv2.boundingRect(c)
            boxes.append((x, y, x + bw, y + bh))

    area_ratio = largest / frame_area          # largest blob, not total pixels

    # texture gate: reject striated warm patterns (sunsets, warm decor). Fire
    # is a coherent bright mass -> low edge density inside its own mask.
    m = mask > 0
    if m.any():
        edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 80, 160)
        edge_density = float((edges[m] > 0).mean())
    else:
        edge_density = 0.0

    is_fire = area_ratio >= min_area_ratio and edge_density <= max_edge_density
    if not is_fire:
        boxes = []                              # don't draw sub-threshold blobs
    return is_fire, area_ratio, boxes, mask
