import time
import json
import threading
import requests
import cv2
import numpy as np
from ultralytics import YOLO
import paho.mqtt.client as mqtt


# 1. KONFIGURASI (SUDAH DIUPDATE)

CAMERA_INDEX = 1

MQTT_BROKER      = "ad9fd300499b4d29ae0706343566b619.s1.eu.hivemq.cloud"
MQTT_PORT        = 8883
MQTT_USER        = "tiarts"
MQTT_PASS        = "Lalilulelo1."
TOPIC_SENSOR     = "prs/sensors"

# Gunakan Localhost agar stabil saat demo
LARAVEL_API_URL  = "http://127.0.0.1:8081/api/reports"
IOT_API_KEY      = "MY_SECRET_IOT_KEY_123"

YOLO_MODEL_PATH  = "yolo.pt"
AI_FRAME_SIZE    = 320
CONFIDENCE_MIN   = 0.50

# --- PARAMETER BARU (0 - 10 CM) ---
# Jarak sensor ke permukaan rata (jarak udara saat tidak ada lubang)
SENSOR_HEIGHT_NORMAL = 4.0  

# Batas minimal (agar lubang 1 cm pun terdeteksi)
DEPTH_THRESHOLD      = 1.0  

WAIT_TIMEOUT     = 4.0


# 2. GLOBAL STATE

current_sensor_data = {"lat": None, "lon": None, "depth": 0}
monitoring_active = False
monitoring_start_time = 0
saved_photo_bytes = None
yolo_boxes_cache = []


# 3. FUNGSI MQTT

def on_connect(client, userdata, flags, reason_code, properties):
    print(f"[MQTT] Terhubung (code: {reason_code})")
    if reason_code == 0:
        client.subscribe(TOPIC_SENSOR)
        print(f"[MQTT] Subscribe: {TOPIC_SENSOR}")

def on_message(client, userdata, msg):
    global current_sensor_data
    try:
        current_sensor_data = json.loads(msg.payload.decode())
    except:
        print("[MQTT] JSON rusak")

mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqttc.username_pw_set(MQTT_USER, MQTT_PASS)
mqttc.tls_set()
mqttc.on_connect = on_connect
mqttc.on_message = on_message

# 4. LOGIKA LEVEL

def determine_level(depth_cm):
    try:
        d = float(depth_cm)
        
        # Rentang 0 - 10 cm:
        if d <= 4.0: 
            return "ringan"    # 1.0 - 4.0 cm
        elif d <= 7.0: 
            return "sedang"    # 4.1 - 7.0 cm
        else: 
            return "berat"     # 7.1 - 10.0 cm
            
    except:
        return "ringan"

def send_report_to_laravel(photo_bytes, lat, lon, depth, level):
    print(f"\n[UPLOAD] Mengirim laporan... (Depth: {depth:.1f} cm | Level: {level})")
    final_lat = lat if lat not in ["null", None] else 0
    final_lon = lon if lon not in ["null", None] else 0

    try:
        files = {'photo': ('pothole.jpg', photo_bytes, 'image/jpeg')}
        data = {
            "latitude": final_lat,
            "longitude": final_lon,
            "depth": depth,
            "level": level
        }
        headers = {"X-Api-Key": IOT_API_KEY}

        res = requests.post(LARAVEL_API_URL, data=data, files=files, headers=headers, timeout=10)
        print("[LARAVEL Response]", res.status_code, res.text)

    except Exception as e:
        print("[ERROR] Gagal upload:", e)


# 5. MAIN PROGRAM

def main():
    global monitoring_active, monitoring_start_time, saved_photo_bytes, yolo_boxes_cache

    print("[INIT] Koneksi MQTT...")
    mqttc.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqttc.loop_start()

    print("[INIT] Load YOLO...")
    model = YOLO(YOLO_MODEL_PATH)

    print(f"[INIT] Buka webcam index {CAMERA_INDEX}")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(3, 640)
    cap.set(4, 480)

    if not cap.isOpened():
        print("[FATAL] Webcam gagal dibuka.")
        return

    print("\n=== SISTEM SIAP: SCANNING JALAN (MAX 10 CM) ===\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Webcam lost, reconnecting...")
            time.sleep(1)
            cap = cv2.VideoCapture(CAMERA_INDEX)
            continue

        # --- HITUNG KEDALAMAN ---
        raw_distance = float(current_sensor_data.get("depth", 0))
        
        # Rumus: Jarak Sensor Sekarang - Jarak Sensor Normal
        calculated_depth = raw_distance - SENSOR_HEIGHT_NORMAL
        
        # Filter nilai minus (noise)
        if calculated_depth < 0: calculated_depth = 0
        curr_depth = calculated_depth

        # ---------------------------------------------------------
        # MODE SENSOR (Wait & Confirm)
        # ---------------------------------------------------------
        if monitoring_active:
            elapsed = time.time() - monitoring_start_time

            # Tetap jalankan YOLO (Visual Only)
            try:
                frame_ai = cv2.resize(frame, (AI_FRAME_SIZE, AI_FRAME_SIZE))
                results = model(frame_ai, verbose=False)
                if results:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        scale_x = frame.shape[1] / AI_FRAME_SIZE
                        scale_y = frame.shape[0] / AI_FRAME_SIZE
                        x1, y1, x2, y2 = int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            except: pass

            # Debug di Layar
            cv2.putText(frame, f"WAITING... {elapsed:.1f}s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            # Tampilkan hitungan agar jelas
            cv2.putText(frame, f"Raw: {raw_distance:.1f}cm | Depth: {curr_depth:.1f}cm", 
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            # --- DEBUG DI TERMINAL ---
            print(f"\r[Sensor Check] Raw: {raw_distance:.1f} - Norm: {SENSOR_HEIGHT_NORMAL} = {curr_depth:.1f} cm", end="")

            # --- LOGIKA PENENTUAN ---
            if curr_depth >= DEPTH_THRESHOLD:
                print(f"\n\n>>> VALID! Lubang sedalam {curr_depth} cm terdeteksi.")
                
                level = determine_level(curr_depth)

                threading.Thread(target=send_report_to_laravel, args=(
                    saved_photo_bytes,
                    current_sensor_data.get("lat"),
                    current_sensor_data.get("lon"),
                    curr_depth,
                    level
                )).start()

                monitoring_active = False
                saved_photo_bytes = None
                print(">>> Data dikirim. Kembali SCANNING.\n")

            elif elapsed > WAIT_TIMEOUT:
                print(f"\n[TIMEOUT] Kedalaman tidak cukup (Max: {curr_depth:.1f} cm).")
                monitoring_active = False
                saved_photo_bytes = None

        # ---------------------------------------------------------
        # MODE SCANNING
        # ---------------------------------------------------------
        else:
            frame_ai = cv2.resize(frame, (AI_FRAME_SIZE, AI_FRAME_SIZE))
            results = model(frame_ai, verbose=False)
            deteksi_visual = False
            
            # Info Depth realtime
            cv2.putText(frame, f"Normal: {SENSOR_HEIGHT_NORMAL}cm | Depth: {curr_depth:.1f}cm", 
                        (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            for box in results[0].boxes:
                conf = float(box.conf[0])
                if conf > CONFIDENCE_MIN:
                    deteksi_visual = True
                    x1, y1, x2, y2 = box.xyxy[0]
                    scale_x = frame.shape[1] / AI_FRAME_SIZE
                    scale_y = frame.shape[0] / AI_FRAME_SIZE
                    x1, y1, x2, y2 = int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, f"{conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            if deteksi_visual:
                print("\n>>> VISUAL DETECTED! Memeriksa kedalaman...")
                _, img_encoded = cv2.imencode('.jpg', frame)
                saved_photo_bytes = img_encoded.tobytes()
                monitoring_active = True
                monitoring_start_time = time.time()

        cv2.imshow("SISTEM DETEKSI 10CM", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    mqttc.loop_stop()

if __name__ == "__main__":
    main()