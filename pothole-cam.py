import time
import json
import threading
import requests
import cv2
import numpy as np
from ultralytics import YOLO
import paho.mqtt.client as mqtt

# ==========================================
# 1. KONFIGURASI (WAJIB DIGANTI SESUAI ALAT)
# ==========================================
# GANTI IP INI DENGAN YANG MUNCUL DI SERIAL MONITOR ARDUINO!
ESP32_STREAM_URL = "http://192.168.0.20:81/stream"  # <--- GANTI INI
# (Contoh: http://192.168.1.5/stream atau http://192.168.43.100/stream)

# MQTT Config
MQTT_BROKER      = "ad9fd300499b4d29ae0706343566b619.s1.eu.hivemq.cloud"
MQTT_PORT        = 8883
MQTT_USER        = "tiarts"
MQTT_PASS        = "Lalilulelo1."
TOPIC_SENSOR     = "prs/sensors"

# Backend Laravel
LARAVEL_API_URL  = "http://192.168.0.21:8000/api/reports" 
IOT_API_KEY      = "MY_SECRET_IOT_KEY_123"

# Logika Deteksi
YOLO_MODEL_PATH  = "yolo.pt"
AI_FRAME_SIZE    = 320   
CONFIDENCE_MIN   = 0.50  
DEPTH_THRESHOLD  = 5.0   
WAIT_TIMEOUT     = 4.0   
SENSOR_HEIGHT_NORMAL = 20.0 # <--- Pastikan ini sudah diukur

# ==========================================
# 2. GLOBAL STATE
# ==========================================
current_sensor_data = {"lat": None, "lon": None, "depth": 0}
monitoring_active = False
monitoring_start_time = 0
saved_photo_bytes = None 

# ==========================================
# 3. FUNGSI MQTT (FIXED FOR PAHO V2)
# ==========================================
def on_connect(client, userdata, flags, reason_code, properties):
    print(f"[MQTT] Terhubung ke Broker! (Code: {reason_code})")
    if reason_code == 0:
        client.subscribe(TOPIC_SENSOR)
        print(f"[MQTT] Mendengarkan topik: {TOPIC_SENSOR}")
    else:
        print(f"[MQTT] Gagal konek, return code: {reason_code}")

def on_message(client, userdata, msg):
    global current_sensor_data
    try:
        payload = json.loads(msg.payload.decode())
        current_sensor_data = payload
        # print(f"[DEBUG] Data Masuk: {payload}") 
    except Exception as e:
        print(f"[MQTT Error] JSON rusak: {e}")

# UPDATE PENTING UNTUK PAHO v2.0+
mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2) 
mqttc.username_pw_set(MQTT_USER, MQTT_PASS)
mqttc.tls_set() 
mqttc.on_connect = on_connect
mqttc.on_message = on_message

# ==========================================
# 4. FUNGSI UTILITY & API
# ==========================================
def determine_level(depth_cm):
    try:
        d = float(depth_cm)
        if d <= 5: return "ringan"    
        elif d <= 10: return "sedang" 
        else: return "berat"          
    except:
        return "ringan"

def send_report_to_laravel(photo_bytes, lat, lon, depth, level):
    print("\n[UPLOAD] Mengirim Laporan ke Laravel...")
    
    final_lat = lat if lat != "null" and lat is not None else 0
    final_lon = lon if lon != "null" and lon is not None else 0
    
    try:
        files = {'photo': ('pothole_evidence.jpg', photo_bytes, 'image/jpeg')}
        data = {
            'latitude': final_lat,
            'longitude': final_lon,
            'depth': depth,
            'level': level
        }
        headers = { "X-Api-Key": IOT_API_KEY }

        response = requests.post(LARAVEL_API_URL, data=data, files=files, headers=headers, timeout=10)
        
        if response.status_code == 201: 
            print(f"[SUCCESS] Laporan Terkirim! ID: {response.json().get('data', {}).get('id')}")
        elif response.status_code == 401:
            print("[ERROR] API Key Ditolak! Cek .env Laravel.")
        else:
            print(f"[FAIL] Server Error: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"[ERROR] Gagal koneksi ke Laravel: {e}")

# ==========================================
# 5. MAIN PROGRAM
# ==========================================
def main():
    global monitoring_active, monitoring_start_time, saved_photo_bytes

    # A. Konek MQTT Dulu sebelum Loop
    print("[INIT] Menghubungkan ke MQTT Broker...")
    try:
        mqttc.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqttc.loop_start()
    except Exception as e:
        print(f"[FATAL] Gagal konek MQTT: {e}")
        return

    # B. Load Model YOLO
    print(f"[INIT] Memuat Model YOLO ({YOLO_MODEL_PATH})...")
    try:
        model = YOLO(YOLO_MODEL_PATH)
    except:
        print("[FATAL] File yolo.pt tidak ditemukan!")
        return

    # C. Buka Kamera (Dengan Error Handling Lebih Baik)
    print(f"[INIT] Membuka Stream Kamera: {ESP32_STREAM_URL}")
    cap = cv2.VideoCapture(ESP32_STREAM_URL)
    
    # Tunggu sebentar untuk memastikan koneksi
    if not cap.isOpened():
        print("-----------------------------------------------------")
        print(f"[FATAL] Gagal membuka stream: {ESP32_STREAM_URL}")
        print("SARAN PERBAIKAN:")
        print("1. Pastikan ESP32 sudah menyala.")
        print("2. Pastikan Laptop & ESP32 di WiFi/Hotspot yang SAMA.")
        print("3. Cek Serial Monitor Arduino untuk lihat IP terbaru.")
        print("-----------------------------------------------------")
        mqttc.loop_stop()
        return

    print("\n=== SISTEM START: SCANNING JALAN ===")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Stream putus. Mencoba reconnect...")
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(ESP32_STREAM_URL)
            continue

        # Hitung kedalaman terkoreksi
        raw_distance = float(current_sensor_data.get("depth", 0))
        # Rumus: Bacaan Sensor - Tinggi Sensor Normal
        calculated_depth = raw_distance - SENSOR_HEIGHT_NORMAL
        if calculated_depth < 0: calculated_depth = 0
        
        curr_depth = calculated_depth
        
        # ---------------------------------------------------------
        # MODE 1: MONITORING
        # ---------------------------------------------------------
        if monitoring_active:
            elapsed = time.time() - monitoring_start_time
            
            cv2.putText(frame, f"WAITING SENSOR... {elapsed:.1f}s", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"DEPTH: {curr_depth:.1f} cm", (20, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if curr_depth > DEPTH_THRESHOLD:
                print(f"\n>>> KONFIRMASI: Sensor mendeteksi kedalaman {curr_depth} cm!")
                level_kerusakan = determine_level(curr_depth)
                
                t = threading.Thread(target=send_report_to_laravel, args=(
                    saved_photo_bytes, 
                    current_sensor_data.get("lat"), 
                    current_sensor_data.get("lon"), 
                    curr_depth, 
                    level_kerusakan
                ))
                t.start()
                
                monitoring_active = False
                saved_photo_bytes = None
                print(">>> Kembali ke mode Scanning...\n")

            elif elapsed > WAIT_TIMEOUT:
                print(f"[TIMEOUT] Sensor tidak mendeteksi lubang. Batal lapor.")
                monitoring_active = False
                saved_photo_bytes = None

        # ---------------------------------------------------------
        # MODE 2: SCANNING
        # ---------------------------------------------------------
        else:
            frame_ai = cv2.resize(frame, (AI_FRAME_SIZE, AI_FRAME_SIZE))
            results = model(frame_ai, verbose=False)
            deteksi_visual = False

            for box in results[0].boxes:
                conf = float(box.conf[0])
                if conf > CONFIDENCE_MIN:
                    deteksi_visual = True
                    break
            
            if deteksi_visual:
                print("\n>>> VISUAL: Kamera melihat lubang! Menunggu konfirmasi sensor...")
                _, img_encoded = cv2.imencode('.jpg', frame)
                saved_photo_bytes = img_encoded.tobytes()
                monitoring_active = True
                monitoring_start_time = time.time()

        cv2.imshow("Sistem Deteksi Pothole", frame)

        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    mqttc.loop_stop()
    print("[EXIT] Sistem Berhenti.")

if __name__ == "__main__":
    main()