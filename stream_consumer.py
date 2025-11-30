# stream_consumer.py
from ultralytics import YOLO
import cv2
import time
import requests
import json

ESP32_STREAM = "http://192.168.0.21:81/stream"  # ganti sesuai IP ESP32
MODEL_PATH = "yolo.pt"
POST_RESULTS = False 
POST_URL = "http://localhost:8000/report" 

def send_results_to_server(payload):
    try:
        res = requests.post(POST_URL, json=payload, timeout=2)
        # print(res.status_code, res.text)
    except Exception as e:
        print("Gagal kirim hasil:", e)

def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(ESP32_STREAM)
    if not cap.isOpened():
        print("Gagal membuka stream:", ESP32_STREAM)
        return

    fps_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal baca frame, mencoba reconnect...")
            time.sleep(1)
            cap.release()
            cap = cv2.VideoCapture(ESP32_STREAM)
            continue

        frame_count += 1

        # jalankan inference (untuk speed, bisa set imgsz atau device jika perlu)
        results = model(frame)  # ultralytics mengembalikan object results

        # buat output sederhana
        detections = []
        r = results[0]
        if hasattr(r, "boxes") and r.boxes is not None:
            for b in r.boxes:
                # setiap b mempunyai .xyxy, .conf, .cls
                xy = b.xyxy.tolist()[0] if hasattr(b.xyxy, "tolist") else b.xyxy
                detections.append({
                    "bbox": [float(x) for x in xy],  # [x1,y1,x2,y2]
                    "confidence": float(b.conf),
                    "class": int(b.cls)
                })

        # Annotate frame (visual)
        annotated = r.plot()  # ultralytics helper

        # Show
        cv2.imshow("YOLO - ESP32 Stream", annotated)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

        # optional: kirim hasil deteksi sebagai JSON ke server
        if POST_RESULTS:
            payload = {
                "timestamp": int(time.time()),
                "detections": detections,
                "frame_index": frame_count
            }
            send_results_to_server(payload)

        # fps printing setiap 5 detik
        if time.time() - fps_time > 5:
            fps = frame_count / (time.time() - fps_time)
            print(f"[INFO] fps ~ {fps:.2f} | detections: {len(detections)}")
            fps_time = time.time()
            frame_count = 0

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
