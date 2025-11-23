from fastapi import FastAPI, UploadFile
from ultralytics import YOLO
import numpy as np
import cv2

app = FastAPI()

# Load model YOLO kamu
model = YOLO("best.pt")

@app.post("/predict")
async def predict(file: UploadFile):
    # Baca gambar dari ESP32-CAM
    img_bytes = await file.read()
    img = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(img, cv2.IMREAD_COLOR)

    # Jalankan deteksi YOLO
    results = model(frame)
    
    detections = []
    for box in results[0].boxes:
        detections.append({
            "class": int(box.cls),
            "confidence": float(box.conf),
            "bbox": box.xyxy.tolist()[0]
        })

    return {
        "total_detections": len(detections),
        "results": detections
    }
