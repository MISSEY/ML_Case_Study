import requests

data = {
    "timestamp": "2024-02-04T10:30:00",
    "temperature": 23.5,
    "humidity": 45.2,
    "light": 450,
    "co2": 800,
    "humidity_ratio": 0.005
}

response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())