"""Fusion engine: combine sensor readings + vision detections into events."""
import time
from dataclasses import dataclass, field

# severity ordering used for alert thresholds and overlay colors
SEV = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


@dataclass
class Event:
    type: str          # INTRUDER | KNOWN_PERSON | FIRE | HAZARD | MOTION
    severity: str      # info | low | medium | high | critical
    message: str
    timestamp: float = field(default_factory=time.time)


class IDSEngine:
    def __init__(self, settings):
        self.s = settings

    def analyze(self, reading, persons, faces, fire, weapons=None) -> list[Event]:
        """reading: SensorReading, persons/faces/weapons: lists, fire: detect_fire tuple."""
        events: list[Event] = []
        weapons = weapons or []
        weapon_labels = ", ".join(sorted({w["label"] for w in weapons}))

        # --- fire / disaster fusion (vision + temp + smoke) ---
        fire_vision = fire[0]
        temp_hot = reading.temperature_c >= self.s.temp_alarm_c
        smoke_high = reading.smoke_ppm >= self.s.smoke_alarm_ppm

        if fire_vision or (temp_hot and smoke_high):
            events.append(Event(
                "FIRE", "critical",
                f"FIRE detected (vision={fire_vision}, "
                f"temp={reading.temperature_c:.0f}C, smoke={reading.smoke_ppm:.0f}ppm)"))
        elif temp_hot or smoke_high:
            events.append(Event(
                "HAZARD", "high",
                f"Possible hazard: temp={reading.temperature_c:.0f}C, "
                f"smoke={reading.smoke_ppm:.0f}ppm"))

        # --- intruder logic (person + face id + distance zone) ---
        in_zone = reading.distance_cm <= self.s.intruder_distance_cm
        if persons:
            known = [f["name"] for f in faces if f["known"]]
            unknown_face = any(not f["known"] for f in faces)
            is_known = bool(faces and known and not unknown_face)

            if weapons:
                who = f"known person ({', '.join(known)})" if is_known \
                    else "unidentified person"
                events.append(Event(
                    "ARMED_INTRUDER", "critical",
                    f"ARMED {who}: {weapon_labels} at {reading.distance_cm:.0f}cm"))
            elif is_known:
                events.append(Event("KNOWN_PERSON", "info",
                                    f"Known person: {', '.join(known)}"))
            else:
                who = "Unknown face" if faces else "Unidentified person"
                sev = "high" if in_zone else "medium"
                events.append(Event(
                    "INTRUDER", sev,
                    f"{who} detected at {reading.distance_cm:.0f}cm "
                    f"({'inside zone' if in_zone else 'approaching'})"))
        elif weapons:
            events.append(Event("WEAPON", "critical",
                                f"Weapon detected ({weapon_labels}), no person in view"))
        elif reading.motion:
            events.append(Event(
                "MOTION", "low",
                f"PIR motion with no person in view ({reading.distance_cm:.0f}cm)"))

        return events
