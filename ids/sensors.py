"""Simulated hardware sensors (Wokwi / Tinkercad style).

Models four virtual sensors you would normally wire to a microcontroller:
  - HC-SR04 ultrasonic ......... distance to nearest object (cm)
  - PIR ........................ motion present (bool)
  - MQ-2 smoke / gas ........... concentration (ppm)
  - thermistor / DHT ........... temperature (C)

In "auto" mode the readings react to what the vision models see (a detected
person makes the ultrasonic report a near distance, detected fire spikes the
smoke + temperature). In "manual" mode the Streamlit sliders drive them, so
you can demo any scenario without a webcam.
"""
import random
import time
from dataclasses import dataclass, field


@dataclass
class SensorReading:
    distance_cm: float
    motion: bool
    smoke_ppm: float = 20.0
    temperature_c: float = 25.0
    timestamp: float = field(default_factory=time.time)


class SimulatedSensors:
    def __init__(self, max_distance: float = 400.0):
        self.max_distance = max_distance
        self.distance = max_distance
        self.smoke = 20.0
        self.temp = 25.0

    def step(self, mode: str = "auto", manual: dict | None = None,
             context: dict | None = None) -> SensorReading:
        context = context or {}

        if mode == "manual" and manual is not None:
            self.distance = manual.get("distance_cm", self.distance)
            self.smoke = manual.get("smoke_ppm", self.smoke)
            self.temp = manual.get("temperature_c", self.temp)
            return SensorReading(self.distance, bool(manual.get("motion", False)),
                                 self.smoke, self.temp)

        # --- auto mode: readings track the vision results ---
        person = context.get("person_present", False)
        box = context.get("person_box")
        frame_w = context.get("frame_w")
        fire = context.get("fire_vision", False)

        if person:
            target = 80.0
            if box and frame_w:                      # bigger bounding box => closer
                ratio = (box[2] - box[0]) / frame_w
                target = max(25.0, min(320.0, 320.0 * (1 - ratio)))
            self.distance += (target - self.distance) * 0.4
            motion = True
        else:
            self.distance += (self.max_distance - self.distance) * 0.2
            motion = False

        target_smoke = 600.0 if fire else 20.0
        target_temp = 85.0 if fire else 25.0
        self.smoke += (target_smoke - self.smoke) * 0.3
        self.temp += (target_temp - self.temp) * 0.3

        return SensorReading(
            distance_cm=max(2.0, self.distance + random.uniform(-3, 3)),
            motion=motion,
            smoke_ppm=max(0.0, self.smoke + random.uniform(-5, 5)),
            temperature_c=self.temp + random.uniform(-1, 1),
        )
