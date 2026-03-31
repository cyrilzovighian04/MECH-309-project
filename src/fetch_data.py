import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from GetWeatherData import fetch_open_meteo_hourly, preprocess, MONTREAL

df_raw = fetch_open_meteo_hourly(
    start_date="2024-01-01",
    end_date="2024-12-31",
    location=MONTREAL,
)

df = preprocess(df_raw)

os.makedirs(os.path.join(os.path.dirname(__file__), "..", "data"), exist_ok=True)
out_path = os.path.join(os.path.dirname(__file__), "..", "data", "montreal_weather.csv")
df.to_csv(out_path)
print(f"Saved {len(df)} rows to {os.path.normpath(out_path)}")
print(f"Columns: {list(df.columns)}")
