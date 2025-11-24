# stream_consumer_mqtt.py
import time, json, uuid
import threading
import requests
import cv2
from ultralytics import YOLO
import paho.mqtt.client as mqtt

# ---------- CONFIG ----------
ESP32_STREAM = "http://10.29.25.251/stream"     # ganti sesuai IP ESP32
MODEL_PATH = "yolo.pt"                            
MQTT_BROKER = "ad9fd300499b4d29ae0706343566b619.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_USER = "tiarts"
MQTT_PASS = "Lalilulelo1."
TOPIC_DETECTED = "prs/detected"
TOPIC_SENSORS = "prs/sensors"
LARAVEL_API = "http://192.168.0.8:8000/api/reports"  
WAIT_SENSOR_SECONDS = 6
# ----------------------------

# shared last sensor reading (no device_id)
last_sensor = None
sensor_lock = threading.Lock()

# MQTT callbacks
def on_connect(client, userdata, flags, rc):
    print("[MQTT] connected rc:", rc)
    client.subscribe(TOPIC_SENSORS, qos=1)

def on_message(client, userdata, msg):
    global last_sensor
    try:
        payload = json.loads(msg.payload.decode())
    except Exception:
        # ignore malformed
        return
    with sensor_lock:
        last_sensor = payload
    print("[MQTT] sensor received:", payload)

# start mqtt client
mqttc = mqtt.Client()
mqttc.username_pw_set(MQTT_USER, MQTT_PASS)
mqttc.tls_set()            # enable TLS
mqttc.on_connect = on_connect
mqttc.on_message = on_message
mqttc.connect(MQTT_BROKER, MQTT_PORT)
mqttc.loop_start()

# load model
model = YOLO(MODEL_PATH)

def determine_level(depth_cm):
    try:
        d = float(depth_cm)
    except:
        return "ringan"
    if d < 3:
        return "ringan"
    elif d < 6:
        return "sedang"
    else:
        return "berat"

def send_report_to_laravel(lat, lon, depth, level, photo_bytes):
    data = {
        "latitude": lat or "",
        "longitude": lon or "",
        "depth": depth or "",
        "level": level
    }
    files = None
    if photo_bytes:
        files = {"photo": ("pothole.jpg", photo_bytes, "image/jpeg")}
    try:
        r = requests.post(LARAVEL_API, data=data, files=files, timeout=10)
        print("[HTTP] laravel:", r.status_code, r.text)
    except Exception as e:
        print("[ERROR] sending to laravel:", e)

def main():
    cap = cv2.VideoCapture(ESP32_STREAM)
    if not cap.isOpened():
        print("[ERROR] cannot open stream:", ESP32_STREAM)
        return

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] failed read frame, reconnecting...")
                time.sleep(1)
                cap.release()
                cap = cv2.VideoCapture(ESP32_STREAM)
                continue

            frame_idx += 1
            small = cv2.resize(frame, (320, 320))
            results = model(small)
            r = results[0]

            if hasattr(r, "boxes") and len(r.boxes) > 0:
                # threshold optional: pick boxes with conf > 0.35
                box = r.boxes[0]
                conf = float(box.conf)
                if conf < 0.35:
                    # ignore low confidence
                    pass
                else:
                    print("[INFO] detection conf=", conf, "frame=", frame_idx)
                    # publish detected message
                    mqttc.publish(TOPIC_DETECTED, json.dumps({"detected": True}), qos=1)
                    # keep full-res photo
                    ok, jpg = cv2.imencode('.jpg', frame)
                    photo_bytes = jpg.tobytes() if ok else None

                    # wait for sensor reading (latest) up to WAIT_SENSOR_SECONDS
                    sensor = None
                    waited = 0.0
                    while waited < WAIT_SENSOR_SECONDS:
                        with sensor_lock:
                            if last_sensor is not None:
                                sensor = last_sensor
                                # after using it, clear last_sensor to avoid reuse
                                last_sensor = None
                                break
                        time.sleep(0.5)
                        waited += 0.5

                    if sensor:
                        lat = sensor.get("lat")
                        lon = sensor.get("lon")
                        depth = sensor.get("depth")
                    else:
                        lat = lon = depth = None

                    level = determine_level(depth if depth is not None else 0)
                    print("[INFO] sending report -> lat:", lat, "lon:", lon, "depth:", depth, "level:", level)
                    send_report_to_laravel(lat, lon, depth, level, photo_bytes)

            # optional show annotated
            annotated = r.plot()
            cv2.imshow("YOLO", annotated)
            if cv2.waitKey(1) == 27:
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        mqttc.loop_stop()
        mqttc.disconnect()

if __name__ == "__main__":
    main()
