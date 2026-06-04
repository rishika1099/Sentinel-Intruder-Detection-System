# Wokwi hardware integration (simulated ESP32 -> MQTT -> Sentinel IDS)

A simulated **ESP32** in Wokwi reads four sensors and publishes their values
over **MQTT**; the Sentinel IDS app subscribes and uses them as the live
sensor feed (instead of the Python simulation).

```
 Wokwi ESP32  ──WiFi──>  MQTT broker  ──>  Sentinel IDS app
 HC-SR04 (distance)      broker.hivemq.com   (Sensors = Wokwi hardware)
 PIR     (motion)        topic:
 DHT22   (temperature)   sentinel-ids/demo/sensors
 pot     (gas/smoke)
```

## Run it

1. Go to **https://wokwi.com** and create a new **ESP32** project.
2. Replace `sketch.ino` and `diagram.json` with the files in this folder.
3. Open the **Library Manager** and add (see `libraries.txt`):
   - `PubSubClient`
   - `DHT sensor library`
   - `Adafruit Unified Sensor`
4. Press **▶ Start the simulation**. The serial monitor should print lines like:
   ```
   {"distance_cm":142,"motion":0,"smoke_ppm":80,"temperature_c":26.5}
   ```
5. In the Sentinel IDS app sidebar, open **📡 Sensors**, choose
   **Wokwi hardware (MQTT)**, and make sure the broker + topic match
   (`broker.hivemq.com` / `sentinel-ids/demo/sensors`). Press **▶ Start**.

The live metric cards (Distance / Motion / Smoke / Temp) now come from the
board. Drag the potentiometer or trigger the PIR in Wokwi and watch the app
react, fire/hazard and intruder-distance logic all run off the real readings.

## Notes

- `Wokwi-GUEST` is Wokwi's built-in open WiFi; it only works inside Wokwi.
- The public broker is shared, so pick a **unique topic** (e.g.
  `sentinel-ids/yourname-7421/sensors`) and set the same value in both the
  sketch (`MQTT_TOPIC`) and the app sidebar to avoid crosstalk.
- The potentiometer stands in for an MQ-2 gas/smoke sensor (Wokwi has no MQ-2
  part); turning it up raises `smoke_ppm` and can trigger a HAZARD/FIRE event.
- This same sketch runs on a **real ESP32** unchanged (use your WiFi SSID/pass).
