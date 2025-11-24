# server.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import tempfile
import cv2
import os
from ultralytics import YOLO

app = FastAPI()


model = YOLO("yolo.pt")  

@app.post("/predict_video")
async def predict_video(file: UploadFile = File(...)):
    # Simpan video upload ke file sementara
    temp_input = tempfile.mktemp(suffix=".mp4")
    with open(temp_input, "wb") as f:
        f.write(await file.read())

    # Output video file
    temp_output = tempfile.mktemp(suffix=".mp4")

    cap = cv2.VideoCapture(temp_input)

    # Ambil detail frame
    fps     = int(cap.get(cv2.CAP_PROP_FPS))
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Writer video output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO detection
        results = model(frame)[0]

        # Gambar bounding box
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.putText(frame, f"{model.names[cls]} {conf:.2f}",
                        (int(x1), int(y1)-5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,255,0), 2)

        out.write(frame)

    cap.release()
    out.release()
    os.remove(temp_input)

    # Kirim file video hasil detection
    return FileResponse(temp_output, media_type="video/mp4", filename="result.mp4")
