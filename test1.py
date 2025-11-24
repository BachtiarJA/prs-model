import requests

video = open("0081.mp4", "rb")

res = requests.post("http://localhost:8000/predict_video",
                    files={"file": ("pothole.mp4", video, "video/mp4")})

# Simpan hasilnya
with open("result.mp4", "wb") as f:
    f.write(res.content)

print("Video hasil tersimpan sebagai result.mp4")
