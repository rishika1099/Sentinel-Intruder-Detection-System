"""Sentinel IDS: Streamlit dashboard tying sensors + vision + alerts together."""
import time
from collections import deque

import cv2
import streamlit as st

from ids.config import Settings
from ids.sensors import SimulatedSensors
from ids.sensors_mqtt import MqttSensors
from ids.video import open_source
from ids.detection.person import PersonDetector
from ids.detection.weapon import WeaponDetector
from ids.detection.fire import detect_fire
from ids.detection.faces import FaceRecognizer
from ids.alerts import AlertManager
from ids.engine import IDSEngine, SEV
from ids.activity import IncidentTracker
from ids.llm import ThreatNarrator
from ids import enroll

st.set_page_config(page_title="Sentinel IDS", page_icon="🛡️", layout="wide")

SEV_COLOR = {                       # BGR for OpenCV overlays
    "info": (0, 180, 0), "low": (0, 200, 200), "medium": (0, 140, 255),
    "high": (0, 80, 255), "critical": (0, 0, 255),
}
SEV_EMOJI = {"info": "🟢", "low": "🟡", "medium": "🟠",
             "high": "🔴", "critical": "🚨"}


@st.cache_resource(show_spinner="Loading YOLOv8 person detector...")
def get_person_detector(conf):
    return PersonDetector(conf=conf)


@st.cache_resource(show_spinner="Loading weapon detector...")
def get_weapon_detector(conf):
    return WeaponDetector(conf=conf)


@st.cache_resource(show_spinner="Loading ArcFace face recognizer...")
def get_face_recognizer(sim_threshold):
    fr = FaceRecognizer(sim_threshold=sim_threshold)
    fr.train()
    return fr


@st.cache_resource(show_spinner="Connecting to Wokwi / MQTT broker...")
def get_mqtt_sensors(broker, port, topic):
    return MqttSensors(broker, port, topic)


@st.cache_resource(show_spinner="Connecting to Claude API...")
def get_narrator():
    return ThreatNarrator()


def draw_overlay(frame, persons, faces, weapons, fire_boxes, reading, events):
    top_sev = max((e.severity for e in events), key=lambda s: SEV[s], default="info")
    pcolor = SEV_COLOR.get(top_sev, (0, 180, 0)) if persons else (0, 180, 0)

    for p in persons:
        x1, y1, x2, y2 = p["box"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), pcolor, 2)
        cv2.putText(frame, f"person {p['conf']:.2f}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, pcolor, 2)

    for f in faces:
        x1, y1, x2, y2 = f["box"]
        c = (0, 180, 0) if f["known"] else (0, 0, 255)
        label = f["name"] if f["known"] else "UNKNOWN"
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
        cv2.putText(frame, label, (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)

    for w in weapons:
        x1, y1, x2, y2 = w["box"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(frame, f"WEAPON: {w['label']} {w['conf']:.2f}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    for (x1, y1, x2, y2) in fire_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, "FIRE", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    hud = (f"dist {reading.distance_cm:5.0f}cm | motion {int(reading.motion)} | "
           f"smoke {reading.smoke_ppm:4.0f}ppm | temp {reading.temperature_c:4.0f}C")
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(frame, hud, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1)
    return frame


# ------------------------------------------------------------------ sidebar
st.sidebar.title("⚙️ Controls")
settings = Settings()

# main header + mode selector (kept in the main area, not the left sidebar)
st.title("🛡️ Sentinel IDS")
st.caption("Sensor + computer-vision intruder, weapon and fire detection")
app_mode = st.radio("Mode", ["Monitor", "Enroll faces"], horizontal=True)

# ------------------------------------------------------------- enroll page
if app_mode == "Enroll faces":
    st.header("📸 Enroll a known face")
    st.caption("Capture photos from your webcam (or upload some) to teach the "
               "system who is allowed. Anyone who doesn't match becomes UNKNOWN.")

    people = enroll.list_people()
    if people:
        st.markdown("**Already enrolled:** "
                    + ", ".join(f"{n} ({c})" for n, c in people.items()))

    name = st.text_input("Person's name", placeholder="e.g. Alice")
    require_face = st.checkbox("Require a detectable face in the shot", True)

    cap_col, up_col = st.columns(2)
    with cap_col:
        st.subheader("Webcam")
        shot = st.camera_input("Take a photo", label_visibility="collapsed")
        if shot is not None:
            if not name.strip():
                st.warning("Enter a name above before saving.")
            else:
                img = enroll.decode_image(shot.getvalue())
                ok, info = enroll.save_face(name, img, require_face)
                if ok:
                    st.success(f"Saved {info}")
                    get_face_recognizer.clear()   # retrain on next Monitor run
                else:
                    st.error(info)

    with up_col:
        st.subheader("Or upload images")
        ups = st.file_uploader("Upload face photos", type=["jpg", "jpeg", "png"],
                               accept_multiple_files=True,
                               label_visibility="collapsed")
        if ups and st.button("Save uploaded photos"):
            if not name.strip():
                st.warning("Enter a name above before saving.")
            else:
                saved = 0
                for u in ups:
                    img = enroll.decode_image(u.getvalue())
                    ok, _ = enroll.save_face(name, img, require_face)
                    saved += int(ok)
                st.success(f"Saved {saved}/{len(ups)} photo(s).")
                get_face_recognizer.clear()

    st.info("After enrolling, switch **Mode** back to *Monitor* and enable "
            "**Face recognition (Stage 2)** in Detection. Aim for 3-10 varied "
            "photos per person.")
    st.stop()

with st.sidebar.expander("🎯 Detection", expanded=True):
    settings.person_conf = st.slider("Person confidence", 0.1, 0.9, 0.40, 0.05)
    settings.intruder_distance_cm = st.slider("Intruder zone (cm)", 30, 400, 200, 10)
    enable_weapon = st.checkbox("Weapon detection (gun / knife)", True)
    settings.weapon_conf = st.slider("Weapon confidence", 0.2, 0.9, 0.60, 0.05)
    enable_fire = st.checkbox("Fire / disaster detection", True)
    settings.fire_min_area_ratio = st.slider(
        "Fire sensitivity (lower = more sensitive)", 0.005, 0.10, 0.020, 0.005,
        help="Largest fire-colored blob must cover at least this fraction of "
             "the frame. Raise it if you get false fire alarms.")
    enable_faces = st.checkbox("Face recognition (Stage 2, ArcFace)", False)
    face_sim = st.slider("Face match strictness", 0.20, 0.60, 0.40, 0.05,
                         help="Higher = stricter (ArcFace cosine similarity)")
    enable_ai = st.checkbox("🤖 AI threat descriptions (Claude)", False,
                            help="Writes a natural-language report on each alert. "
                                 "Needs ANTHROPIC_API_KEY in .env.")

with st.sidebar.expander("📡 Sensors"):
    sensor_backend = st.radio("Source", ["Simulated", "Wokwi hardware (MQTT)"])
    manual = None
    mqtt_cfg = None
    if sensor_backend == "Wokwi hardware (MQTT)":
        mode = "hardware"
        mqtt_cfg = {
            "broker": st.text_input("MQTT broker", "broker.hivemq.com"),
            "port": int(st.number_input("Port", value=1883)),
            "topic": st.text_input("Topic", "sentinel-ids/demo/sensors"),
        }
        st.caption("Run the Wokwi project in wokwi/ and match this topic.")
    else:
        sensor_mode = st.radio("Mode", ["Auto (react to vision)", "Manual"])
        if sensor_mode == "Manual":
            manual = {
                "distance_cm": st.slider("Distance (cm)", 2, 400, 150),
                "motion": st.checkbox("PIR motion", True),
                "smoke_ppm": st.slider("Smoke (ppm)", 0, 1000, 20),
                "temperature_c": st.slider("Temperature (C)", 0, 120, 25),
            }
        mode = "manual" if sensor_mode == "Manual" else "auto"

with st.sidebar.expander("✉️ Email alerts"):
    enable_alerts = st.checkbox("Send email on high/critical", False)
    settings.smtp_host = st.text_input("SMTP host", settings.smtp_host)
    settings.smtp_port = int(st.number_input("SMTP port", value=settings.smtp_port))
    settings.smtp_user = st.text_input("SMTP user", settings.smtp_user)
    settings.smtp_password = st.text_input("SMTP password", settings.smtp_password,
                                           type="password")
    settings.alert_to = st.text_input("Alert to", settings.alert_to)
    settings.alert_from = settings.alert_from or settings.smtp_user

max_w = 960
if "running" not in st.session_state:
    st.session_state.running = False

# -------------------------------------- video source + run control (main area)
ctl = st.columns([1.4, 2.4, 1.2], gap="medium")
with ctl[0]:
    src_kind = st.radio("Video source", ["Webcam", "File / URL / YouTube"])
with ctl[1]:
    if src_kind == "Webcam":
        source = str(st.number_input("Webcam index", 0, 10, 0))
    else:
        source = st.text_input(
            "Path or link",
            placeholder="https://youtube.com/watch?v=...  or  /path/clip.mp4")
with ctl[2]:
    st.write("")
    st.write("")
    if st.session_state.running:
        if st.button("⏹  Stop", use_container_width=True):
            st.session_state.running = False
            st.rerun()
    else:
        if st.button("▶  Start", type="primary", use_container_width=True):
            st.session_state.running = True
            st.rerun()
running = st.session_state.running

# placeholders laid out top-to-bottom so content spreads across the page
banner_ph = st.empty()                       # big all-clear / threat banner
metrics_ph = st.empty()                       # row of live sensor cards
col_video, col_right = st.columns([3, 2], gap="large")
video_ph = col_video.empty()
events_ph = col_right.empty()
log_ph = col_right.empty()

if not running:
    banner_ph.info("Pick a **Video source** above and press **▶ Start** to begin. "
                   "Detection / sensor / email options are in the sidebar.")
    # AI Q&A over the incidents from the last run
    past = st.session_state.get("incident_log", [])
    if enable_ai and past:
        st.subheader("🤖 Ask the AI about what happened")
        st.caption(f"{len(past)} incident(s) logged in the last session.")
        q = st.chat_input("e.g. Was anyone armed? What was the closest approach?")
        if q:
            with st.spinner("Thinking..."):
                ans = get_narrator().ask(q, past)
            st.markdown(f"**Q:** {q}")
            st.markdown(f"**A:** {ans}")
        with st.expander("Show raw incident log"):
            st.markdown("\n".join(f"- {x}" for x in past))
    st.stop()

detector = get_person_detector(settings.person_conf)
detector.conf = settings.person_conf
weapon_det = None
if enable_weapon:
    weapon_det = get_weapon_detector(settings.weapon_conf)
    weapon_det.conf = settings.weapon_conf
    if not weapon_det.firearm_available:
        st.warning("Firearm model not found at models/weapon.pt - guns won't be "
                   "detected (knife/bat/scissors still are). See README to add it.")
face_rec = get_face_recognizer(face_sim) if enable_faces else None
engine = IDSEngine(settings)
alerts = AlertManager(settings)
if mqtt_cfg:
    sensors = get_mqtt_sensors(mqtt_cfg["broker"], mqtt_cfg["port"], mqtt_cfg["topic"])
    st.caption(f"📡 MQTT `{mqtt_cfg['topic']}` @ {mqtt_cfg['broker']} - "
               + ("🟢 connected" if sensors.connected else "🔴 connecting...")
               + (" · ⚠️ no data yet (start the Wokwi sim)" if sensors.stale else ""))
else:
    sensors = SimulatedSensors()
tracker = IncidentTracker()
weapon_window = deque(maxlen=3)   # temporal confirmation against single-frame blips

narrator = get_narrator() if enable_ai else None
if enable_ai and (narrator is None or not narrator.available):
    st.warning("AI descriptions need ANTHROPIC_API_KEY in .env (and the anthropic "
               "package). Running without AI.")
    narrator = None
ai_text, ai_last = "", 0.0       # latest AI description + its timestamp
AI_COOLDOWN = 20.0               # seconds between AI calls (they cost money + time)

cap = open_source(source)
if cap is None or not cap.isOpened():
    video_ph.error(f"Could not open source: {source!r}")
    st.stop()

incident_log = []   # one entry per finished incident (behaviour summary)
try:
    while running:
        ok, frame = cap.read()
        if not ok:
            video_ph.warning("Stream ended or no frame. Press Stop, then Start "
                             "to restart.")
            break

        h, w = frame.shape[:2]
        if w > max_w:
            frame = cv2.resize(frame, (max_w, int(h * max_w / w)))

        persons = detector.detect(frame)
        # weapons must show up in >=2 of the last 3 frames to count (kills blips)
        weapons_raw = weapon_det.detect(frame) if weapon_det else []
        weapon_window.append(bool(weapons_raw))
        weapons = weapons_raw if sum(weapon_window) >= 2 else []
        faces = face_rec.identify(frame) if face_rec else []
        fire = detect_fire(frame, settings.fire_min_area_ratio) if enable_fire \
            else (False, 0.0, [], None)

        person_box = persons[0]["box"] if persons else None
        reading = sensors.step(mode, manual, {
            "person_present": bool(persons), "person_box": person_box,
            "frame_w": frame.shape[1], "fire_vision": fire[0],
        })

        events = engine.analyze(reading, persons, faces, fire, weapons)
        annotated = draw_overlay(frame, persons, faces, weapons, fire[2],
                                 reading, events)
        video_ph.image(annotated, channels="BGR", use_container_width=True)

        # behaviour tracking: build a narrative incident, not per-frame spam
        tracker.update(events, persons, faces, weapons, reading, frame.shape[1])
        closed = tracker.maybe_close()
        if closed:
            incident_log.insert(0, closed)
            incident_log = incident_log[:30]
        live = tracker.live_summary()

        # top threat banner
        top_sev = max((SEV[e.severity] for e in events), default=0)
        if top_sev >= SEV["high"]:
            banner_ph.error("🚨 THREAT — " + "  ·  ".join(
                e.type for e in events if SEV[e.severity] >= SEV["high"]))
        elif events:
            banner_ph.warning("⚠️ " + "  ·  ".join(e.type for e in events))
        else:
            banner_ph.success("✅ All clear")

        # live sensor cards spread across the page
        with metrics_ph.container():
            mc = st.columns(6)
            mc[0].metric("📏 Distance", f"{reading.distance_cm:.0f} cm")
            mc[1].metric("🏃 Motion", "YES" if reading.motion else "—")
            mc[2].metric("💨 Smoke", f"{reading.smoke_ppm:.0f} ppm")
            mc[3].metric("🌡️ Temp", f"{reading.temperature_c:.0f} °C")
            mc[4].metric("🧍 People", len(persons))
            mc[5].metric("🔫 Weapons", len(weapons))

        # AI threat description: only on a high/critical event, rate-limited
        if narrator and top_sev >= SEV["high"] and (time.time() - ai_last) > AI_COOLDOWN:
            ok_jpg, buf = cv2.imencode(".jpg", annotated)
            ai_text = narrator.describe_alert(
                buf.tobytes() if ok_jpg else None,
                {"reading": reading, "people": len(persons), "weapons": weapons,
                 "faces": faces, "fire": fire[0], "events": events, "activity": live})
            ai_last = time.time()

        # current events + activity + AI report (right column)
        ev_lines = ["#### Current events"]
        ev_lines += [f"{SEV_EMOJI[e.severity]} **{e.type}** — {e.message}"
                     for e in events] or ["✅ All clear"]
        if live:
            ev_lines += ["#### 🎬 Activity", f"_{live}_"]
        if ai_text:
            ev_lines += ["#### 🤖 AI assessment", f"> {ai_text}"]
        events_ph.markdown("\n".join(ev_lines))

        # alerts on high/critical with snapshot (rate-limited per type)
        for e in events:
            if enable_alerts and SEV[e.severity] >= SEV["high"]:
                body = f"{e.message}\n\nTime: {time.ctime(e.timestamp)}"
                if ai_text:
                    body += f"\n\nAI assessment:\n{ai_text}"
                alerts.send(
                    subject=f"[Sentinel IDS] {e.type} ({e.severity})",
                    body=body, frame=annotated, key=e.type)

        # log shows the ongoing incident (live) + finished-incident summaries
        rows = []
        if live:
            rows.append(f"🔴 **ONGOING** - {live}")
        rows += [f"- {x}" for x in incident_log]
        log_md = "### 🧾 Incident log (what happened)\n"
        log_md += "\n".join(rows) if rows else "_no incidents yet_"
        log_ph.markdown(log_md)
finally:
    last = tracker.force_close()        # flush any incident in progress
    if last:
        incident_log.insert(0, last)
        log_ph.markdown("### 🧾 Incident log (what happened)\n"
                        + "\n".join(f"- {x}" for x in incident_log))
    st.session_state["incident_log"] = incident_log   # keep for AI Q&A
    cap.release()
