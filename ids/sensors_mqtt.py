"""Hardware sensor backend: read live readings from a Wokwi-simulated ESP32
(or any real ESP32/Arduino) over MQTT.

The microcontroller publishes JSON like
    {"distance_cm": 142.0, "motion": 1, "smoke_ppm": 80.0, "temperature_c": 26.5}
to an MQTT topic; this class subscribes and exposes the latest reading through
the same `.step()` interface as SimulatedSensors, so the rest of the app does
not care whether sensors are simulated in Python or coming from the board.
"""
import json
import threading
import time

import paho.mqtt.client as mqtt

from .sensors import SensorReading


class MqttSensors:
    def __init__(self, broker: str = "broker.hivemq.com", port: int = 1883,
                 topic: str = "sentinel-ids/demo/sensors",
                 command_topic: str = "sentinel-ids/demo/commands",
                 stale_after: float = 6.0):
        self.broker, self.port, self.topic = broker, port, topic
        self.command_topic = command_topic
        self.stale_after = stale_after
        self._lock = threading.Lock()
        self._latest: dict | None = None
        self._last_ts = 0.0
        self.error: str | None = None

        try:
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        except (AttributeError, TypeError):          # paho < 2.0
            self.client = mqtt.Client()
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        try:
            self.client.connect(broker, port, keepalive=30)
            self.client.loop_start()
        except Exception as e:                       # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"

    # ------------------------------------------------------------- callbacks
    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        client.subscribe(self.topic)

    def _on_message(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode())
            with self._lock:
                self._latest = data
                self._last_ts = time.time()
        except (ValueError, UnicodeDecodeError):
            pass

    # ------------------------------------------------------------- interface
    @property
    def connected(self) -> bool:
        return self.client.is_connected()

    @property
    def stale(self) -> bool:
        with self._lock:
            return self._latest is None or \
                (time.time() - self._last_ts) > self.stale_after

    def step(self, mode=None, manual=None, context=None) -> SensorReading:
        """Return the latest hardware reading (args ignored; kept for interface
        parity with SimulatedSensors)."""
        with self._lock:
            d = dict(self._latest) if self._latest else None
        if not d:
            # nothing received yet -> benign all-clear defaults
            return SensorReading(distance_cm=400.0, motion=False,
                                 smoke_ppm=0.0, temperature_c=25.0)
        return SensorReading(
            distance_cm=float(d.get("distance_cm", 400.0)),
            motion=bool(d.get("motion", False)),
            smoke_ppm=float(d.get("smoke_ppm", 0.0)),
            temperature_c=float(d.get("temperature_c", 25.0)),
        )

    def send_command(self, payload: str) -> bool:
        """Publish a raw command string to the board's command topic
        (app -> hardware). Returns True if handed to the client."""
        try:
            self.client.publish(self.command_topic, payload)
            return True
        except Exception:                            # noqa: BLE001
            return False

    def set_alarm(self, on: bool) -> bool:
        """Turn the board's alarm (buzzer + LED beacon) ON or OFF."""
        return self.send_command("ON" if on else "OFF")

    def close(self):
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception:                            # noqa: BLE001
            pass
