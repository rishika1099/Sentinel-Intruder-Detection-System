"""Quick offline smoke test: verifies the pipeline without a webcam.

Builds a synthetic frame with a fire-colored blob, runs fire detection and the
fusion engine against simulated sensors, and prints the resulting events.
Run:  python selftest.py
"""
import numpy as np

from ids.config import Settings
from ids.sensors import SimulatedSensors
from ids.detection.fire import detect_fire
from ids.engine import IDSEngine


def main():
    settings = Settings()
    engine = IDSEngine(settings)
    sensors = SimulatedSensors()

    # synthetic 480x640 frame, dark with a bright orange fire blob
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[150:330, 240:420] = (40, 130, 255)  # BGR orange

    fire = detect_fire(frame, settings.fire_min_area_ratio)
    print(f"fire detected={fire[0]}  area_ratio={fire[1]:.4f}  boxes={len(fire[2])}")

    reading = sensors.step("auto", context={"person_present": False,
                                            "fire_vision": fire[0]})
    print(f"sensors: dist={reading.distance_cm:.0f}cm motion={reading.motion} "
          f"smoke={reading.smoke_ppm:.0f}ppm temp={reading.temperature_c:.0f}C")

    events = engine.analyze(reading, persons=[], faces=[], fire=fire)
    print("events:")
    for e in events:
        print(f"  [{e.severity}] {e.type}: {e.message}")

    assert fire[0], "expected fire to be detected in synthetic frame"
    assert any(e.type == "FIRE" for e in events), "expected a FIRE event"
    print("\nOK: pipeline smoke test passed.")


if __name__ == "__main__":
    main()
