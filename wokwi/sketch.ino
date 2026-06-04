/*
 * Sentinel IDS - Wokwi ESP32 sensor node
 * Reads HC-SR04 (distance), PIR (motion), DHT22 (temperature) and a
 * potentiometer standing in for an MQ-2 gas/smoke sensor, then publishes a
 * JSON reading over MQTT once a second. The Sentinel IDS Streamlit app
 * subscribes to the same broker/topic (Sensors -> "Wokwi hardware (MQTT)").
 *
 * Required Wokwi libraries (add via the "Library Manager" panel):
 *   - PubSubClient  (by Nick O'Leary)
 *   - DHT sensor library (by Adafruit) + Adafruit Unified Sensor
 */
#include <WiFi.h>
#include <PubSubClient.h>
#include <DHT.h>

// ---------- config (must match the app sidebar) ----------
const char* WIFI_SSID   = "Wokwi-GUEST";
const char* WIFI_PASS   = "";
const char* MQTT_BROKER = "broker.hivemq.com";
const int   MQTT_PORT   = 1883;
const char* MQTT_TOPIC  = "sentinel-ids/demo/sensors";

// ---------- pins ----------
#define TRIG_PIN 5
#define ECHO_PIN 18
#define PIR_PIN  19
#define DHT_PIN  4
#define GAS_PIN  34          // potentiometer = simulated MQ-2 gas/smoke

DHT dht(DHT_PIN, DHT22);
WiFiClient wifiClient;
PubSubClient mqtt(wifiClient);

long readDistanceCm() {
  digitalWrite(TRIG_PIN, LOW);  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH); delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  long dur = pulseIn(ECHO_PIN, HIGH, 30000);   // ~5 m timeout
  if (dur == 0) return 400;                    // no echo -> far
  return dur * 0.0343 / 2;
}

void connectWifi() {
  WiFi.begin(WIFI_SSID, WIFI_PASS, 6);         // channel 6 = fast in Wokwi
  while (WiFi.status() != WL_CONNECTED) delay(200);
}

void connectMqtt() {
  while (!mqtt.connected()) {
    String cid = "sentinel-esp-" + String(random(0xffff), HEX);
    if (mqtt.connect(cid.c_str())) break;
    delay(500);
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(PIR_PIN, INPUT);
  dht.begin();
  connectWifi();
  mqtt.setServer(MQTT_BROKER, MQTT_PORT);
  connectMqtt();
}

void loop() {
  if (!mqtt.connected()) connectMqtt();
  mqtt.loop();

  long  dist   = readDistanceCm();
  int   motion = digitalRead(PIR_PIN);
  float temp   = dht.readTemperature();
  if (isnan(temp)) temp = 25.0;
  int   gasRaw = analogRead(GAS_PIN);          // 0..4095
  float smoke  = gasRaw * (1000.0 / 4095.0);   // 0..1000 ppm

  char payload[160];
  snprintf(payload, sizeof(payload),
    "{\"distance_cm\":%ld,\"motion\":%d,\"smoke_ppm\":%.0f,\"temperature_c\":%.1f}",
    dist, motion, smoke, temp);
  mqtt.publish(MQTT_TOPIC, payload);
  Serial.println(payload);
  delay(1000);
}
