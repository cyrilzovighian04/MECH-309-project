import requests
import pandas as pd
import numpy as np
import os

URL = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": 45.5017,
    "longitude": -73.5673,
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "hourly": "temperature_2m,wind_speed_10m",
    "timezone": "America/Toronto",
}

response = requests.get(URL, params=params)
response.raise_for_status()
data = response.json()

hourly = data["hourly"]
df = pd.DataFrame({
    "time": hourly["time"],
    "temperature_2m": np.array(hourly["temperature_2m"], dtype=float),
    "wind_speed_10m": np.array(hourly["wind_speed_10m"], dtype=float),
})

os.makedirs("data", exist_ok=True)
df.to_csv("data/montreal_weather.csv", index=False)
print(f"Saved {len(df)} rows to data/montreal_weather.csv")
