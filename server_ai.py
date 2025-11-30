# File: ai_server.py (Versi Ultimate)
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
import cv2
import uvicorn
import json
import paho.mqtt.client as mqtt

# ================= CONFIG =================
MQTT_BROKER      = "ad9fd300499b4d29ae0706343566b619.s1.eu.hivemq.cloud"
MQTT_PORT        = 8883
MQTT_USER        = "tiarts"
MQTT_PASS        = "Lalilulelo1."
TOPIC_SENSOR     = "prs/sensors"

# TINGGI SENSOR (KALIBRASI) - Sesuaikan alat!
SENSOR_HEIGHT_NORMAL = 20.0 
# ==========================================

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State untuk Sensor
current_sensor_data = {
    "raw_depth": 0,
    "real_depth": 0,
    "lat": 0, 
    "lon": 0
}

# --- 1. SETUP MQTT (BACKGROUND) ---
def on_connect(client, userdata, flags, reason_code, properties):
    print(f"[MQTT] Terhubung ke Broker! Code: {reason_code}")
    if reason_code == 0:
        client.subscribe(TOPIC_SENSOR)

def on_message(client, userdata, msg):
    global current_sensor_data
    try:
        # Terima: {"lat": -8.1, "lon": 113.1, "depth": 20.5}
        payload = json.loads(msg.payload.decode())
        
        raw_depth = float(payload.get("depth", 0))
        lat = payload.get("lat")
        lon = payload.get("lon")

        # Hitung Kedalaman Asli (Koreksi Tinggi)
        real_depth = raw_depth - SENSOR_HEIGHT_NORMAL
        if real_depth < 0: real_depth = 0

        current_sensor_data = {
            "raw_depth": raw_depth,
            "real_depth": round(real_depth, 2), # Dibulatkan 2 desimal
            "lat": lat if lat != "null" else 0,
            "lon": lon if lon != "null" else 0
        }
    except Exception as e:
        print(f"[MQTT Error] {e}")

# Jalankan MQTT Client
mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqttc.username_pw_set(MQTT_USER, MQTT_PASS)
mqttc.tls_set()
mqttc.on_connect = on_connect
mqttc.on_message = on_message
mqttc.connect(MQTT_BROKER, MQTT_PORT, 60)
mqttc.loop_start() # Jalan di background thread

# --- 2. SETUP YOLO ---
print("[INIT] Memuat Model YOLO...")
model = YOLO("yolo.pt")

# --- 3. API ENDPOINTS ---

# Endpoint Baru: Web minta data sensor lewat sini
@app.get("/sensor-data")
def get_sensor_data():
    return current_sensor_data

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img_resized = cv2.resize(img, (320, 320)) 
    results = model(img_resized)

    detections = []
    for box in results[0].boxes:
        conf = float(box.conf[0])
        if conf > 0.40:
            coords = box.xyxyn.tolist()[0] 
            
            # Tentukan level (bisa pakai data sensor real-time juga lho!)
            # Disini kita gabungkan: Kalau sensor deteksi dalam, level ikut sensor
            level = "ringan"
            
            # Cek data sensor saat ini
            depth_now = current_sensor_data["real_depth"]
            
            # Logika Hybrid (Visual + Sensor)
            if depth_now > 10: level = "berat"
            elif depth_now > 5: level = "sedang"
            elif conf > 0.7: level = "berat" # Fallback visual
            
            detections.append({
                "bbox": coords,
                "conf": conf,
                "label": "Pothole",
                "level": level
            })

    return {"results": detections, "sensor_context": current_sensor_data}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)