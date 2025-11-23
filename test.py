import requests

img = open("pothole.jpg", "rb")
res = requests.post("http://localhost:8000/predict", files={"file": img})
print(res.json())
