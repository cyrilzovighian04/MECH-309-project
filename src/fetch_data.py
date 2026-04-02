import sys
import os

# Allow imports from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from GetWeatherData import fetch_open_meteo_hourly, preprocess, MONTREAL

# Fetch and preprocess hourly weather data for Montreal, full year 2024
df_raw = fetch_open_meteo_hourly(
    start_date="2024-01-01",
    end_date="2024-12-31",
    location=MONTREAL,
)
df = preprocess(df_raw)

# Save to CSV
out_path = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "montreal_weather.csv")
)
os.makedirs(os.path.dirname(out_path), exist_ok=True)
df.to_csv(out_path)

print(f"Saved {len(df)} rows to {out_path}")
print(f"Columns: {list(df.columns)}")