# 🛡️ Sentinel IDS

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=rishika1099/Sentinel-Intruder-Detection-System&branch=main&mainModule=app.py)

An **Intruder Detection System** that fuses **simulated / Wokwi hardware sensors**
with **computer-vision models**. It detects people and whether they are known or
unknown, estimates distance from an ultrasonic sensor, flags weapons (custom
trained YOLO models), spots fire / disaster conditions, narrates what an intruder
was doing, and **emails the owner** (with a snapshot) when something serious
happens. Runs on your Mac/laptop through a Streamlit dashboard; sensors can be
pure-Python simulated or fed live from a Wokwi-simulated ESP32 over MQTT.

> **Live demo / deploy:** click the **Open in Streamlit** badge above to deploy
> your own copy on Streamlit Community Cloud. Note: the hosted version can only
> use **File / URL video** (cloud servers have no webcam), and heavy models run
> slower there, the full experience (webcam, real-time) is local via
> `streamlit run app.py`.

## What it does

| Capability | How |
|---|---|
| Presence + distance | Simulated HC-SR04 ultrasonic + PIR motion (Wokwi/Tinkercad style), fused with vision |
| Person detection | YOLOv8 (`yolov8n`) |
| Weapon detection | Firearm model (`models/weapon.pt`) + COCO classes -> firearm, knife, baseball bat, scissors |
| Known vs unknown | OpenCV LBPH face recognition against enrolled faces (Stage 2) |
| Fire / disaster | Color-based fire detection + simulated smoke (MQ-2) & temperature sensors |
| Alerts | Email to owner with snapshot, per-event cooldown (owner then calls 911) |
| Video input | Webcam, local file, direct URL, or a pasted **YouTube link** |

> Note on 911: emergency services cannot be dialed through an API by an
> individual app. Sentinel alerts the **owner**, who makes the 911 call. The
> alert text and email recipient are configurable.

## Setup

```bash
cd ~/Desktop/IDS
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
# ultralytics pulls in opencv-python, which clashes with the contrib build
# that provides face recognition. Force the contrib build to win:
pip uninstall -y opencv-python opencv-python-headless
pip install opencv-contrib-python
```

(Optional) email alerts:

```bash
cp .env.example .env
# edit .env with your SMTP details (Gmail: use an App Password)
```

## Run

```bash
source .venv/bin/activate
streamlit run app.py
```

Then in the sidebar:
1. Pick a **video source**: Webcam, or paste a file path / direct URL / YouTube link.
2. Tune **Detection** (person confidence, intruder zone, fire on/off, faces on/off).
3. Choose **Sensors** mode: *Auto* (readings react to the video) or *Manual* (drive the sliders to stage any scenario).
4. (Optional) fill in **Email alerts** and tick "Send email on high/critical".
5. Flip **Run system** on.

Offline sanity check (no webcam, no models needed beyond install):

```bash
python selftest.py
```

## Deploy to Streamlit Community Cloud

1. Push your fork to GitHub (this repo already is).
2. Click the **Open in Streamlit** badge at the top, sign in with GitHub, set
   the main file to `app.py`, and **Deploy**. You get a public `*.streamlit.app` URL.
3. On the hosted app, pick **Demo clip** (or paste a video URL), the cloud has
   no webcam, then press **Start**.

**Free-tier caveat (important):** the free tier gives ~1 GB RAM, and this app
loads PyTorch + several CV models. With everything on it can run out of memory.
If the app crashes or shows the "could not load detection models" message:

- Keep **Face recognition OFF** (ArcFace/InsightFace is the heaviest piece).
- Turn off **High-accuracy weapon mode (TTA)** for lower CPU.
- For the full experience, run **locally** (`streamlit run app.py`) or use a
  larger instance (paid Streamlit tier, or Hugging Face Spaces with more RAM).

`packages.txt` (system libs for OpenCV) and `requirements.txt` are already set
up for the cloud build.

## Known vs unknown faces (Stage 2)

Face recognition is staged separately so you can start with plain person
detection. To enable it:

1. Add photos under `known_faces/<person_name>/` (see `known_faces/README.md`).
2. Restart the app and tick **Face recognition (Stage 2)**.

Matched people show as a green box with their name; everyone else is **UNKNOWN**
and triggers an intruder event when inside the zone.

## How the fusion works

```
 video frame ─┬─> YOLOv8 person detector ──┐
              ├─> LBPH face recognizer ─────┤
              └─> color fire detector ──────┤
                                            ├─> IDSEngine.analyze() ─> events ─> overlay + email
 simulated sensors (auto/manual) ───────────┘
   distance / motion / smoke / temperature
```

Event rules (`ids/engine.py`):
- **ARMED_INTRUDER** (critical): a person AND a weapon (firearm/knife) in frame.
  Still critical even if the person's face is known; the message notes who.
- **WEAPON** (critical): a weapon detected with no person visible.
- **FIRE** (critical): vision sees fire, *or* simulated temp **and** smoke are both high.
- **HAZARD** (high): temp or smoke high on their own (e.g. gas leak / overheating).
- **INTRUDER** (high if inside the distance zone, else medium): a person who is
  unknown / unidentified.
- **KNOWN_PERSON** (info): a recognized enrolled person.
- **MOTION** (low): PIR motion but no person visible.

### Weapon detection (multi-source)

The base COCO YOLO model has **no firearm class**, so guns are invisible to it,
but COCO *does* include real threat objects (knife, baseball bat, scissors).
Weapon detection combines two sources for broad coverage with real type labels:

1. **Firearm model** (`models/weapon.pt`) -> **firearm** (catches pistols and
   rifles). Run at high confidence (0.60 default) because it can false-fire on
   grainy/distant people at lower thresholds.
2. **COCO model** -> **knife**, **baseball bat**, **scissors**.

On top of that, a weapon must appear in **at least 2 of the last 3 frames**
before it counts, so a single-frame blip can't raise a false "armed" alert.

There is no good multi-class real-world weapon model publicly available (the
broad ones collapse to a single "weapon" label; the multi-class ones are
trained on video-game art), which is why this two-source approach is used.

If `models/weapon.pt` is missing, firearm detection switches off with a warning
(knife/bat/scissors still work). To (re)download it:

```bash
mkdir -p models
curl -L -o models/weapon.pt \
  https://huggingface.co/Hadi959/weapon-detection-yolov8/resolve/main/best.pt
```

Source: [Hadi959/weapon-detection-yolov8](https://huggingface.co/Hadi959/weapon-detection-yolov8).
Like the fire detector, this is a community model: for higher accuracy or to
distinguish rifles from handguns, swap in a model with more weapon classes
behind the same `WeaponDetector` interface.

High/critical events trigger an email (subject + body + snapshot), rate-limited
per event type by `alert_cooldown_s`.

### Incident log (what the intruder was doing)

Instead of logging an identical line every frame, `ids/activity.py` groups a
continuous stretch of activity into one **incident** and narrates the behaviour:
duration, whether the subject approached or moved away (and from/to what
distance), which way they crossed the view, whether they loitered, whether they
were armed and with what, and known vs unidentified. The panel shows the live
activity; the log keeps one summary line per finished incident, e.g.

```
02:22:03-02:22:21 (18s): armed person; approached (360->120cm); crossed view L->R; carrying firearm
```

## Project layout

```
app.py                  Streamlit dashboard + capture loop
selftest.py             offline pipeline smoke test
ids/
  config.py             thresholds + email settings
  sensors.py            simulated ultrasonic / PIR / smoke / temperature
  video.py              webcam / file / URL / YouTube source resolver
  alerts.py             email alerting with cooldown + snapshot
  engine.py             sensor + vision fusion -> events
  activity.py           incident tracker -> behaviour summaries
  detection/
    person.py           YOLOv8 person detection
    weapon.py           YOLOv8 firearm/knife detection
    fire.py             color-based fire detection
    faces.py            LBPH known/unknown face recognition
models/weapon.pt        weapon detection weights (downloaded, see below)
known_faces/            enrolled face photos (Stage 2)
```

## Limitations & where to extend

- **Fire detection is a color heuristic.** It can false-positive on sunsets or
  bright red objects. Swap in a trained YOLO fire/smoke model behind the same
  `detect_fire()` signature for production accuracy; the temp+smoke fusion
  already helps suppress false alarms.
- **Sensors are simulated.** To go to real hardware, replace `SimulatedSensors`
  with a class that reads a microcontroller (e.g. a Raspberry Pi GPIO HC-SR04 /
  PIR / MQ-2, or an Arduino over serial). The rest of the pipeline is unchanged.
- **Manual sensor sliders restart the stream** when changed (Streamlit reruns on
  widget interaction). Auto mode is the smoother live demo.
- LBPH recognition is lightweight; for higher accuracy use an embedding model
  (InsightFace / `face_recognition`) behind the same `FaceRecognizer` interface.
```
