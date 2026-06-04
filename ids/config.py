"""Central settings, loaded from environment / .env with sensible defaults."""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    # --- Detection thresholds ---
    person_conf: float = 0.40          # YOLO confidence for "person"
    weapon_conf: float = 0.60          # firearm confidence (high to avoid false alarms)
    fire_min_area_ratio: float = 0.020  # largest fire-like blob must cover this fraction of the frame
    face_tolerance: float = 70.0        # LBPH distance; lower = stricter match
    intruder_distance_cm: float = 200.0  # person within this range = inside the zone

    # --- Disaster fusion (simulated sensors) ---
    temp_alarm_c: float = 55.0          # temperature considered dangerous
    smoke_alarm_ppm: float = 300.0      # smoke/gas considered dangerous

    # --- Alerting ---
    alert_cooldown_s: float = 60.0      # min seconds between alerts of the same type

    # --- Email (overridable from the Streamlit sidebar) ---
    smtp_host: str = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    smtp_user: str = os.getenv("SMTP_USER", "")
    smtp_password: str = os.getenv("SMTP_PASSWORD", "")
    alert_from: str = os.getenv("ALERT_FROM", os.getenv("SMTP_USER", ""))
    alert_to: str = os.getenv("ALERT_TO", "")
