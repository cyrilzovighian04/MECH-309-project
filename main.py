import os
import sys
import subprocess

import pandas as pd

# ---------------------------------------------------------------------------
# Step 1: ensure data exists
# ---------------------------------------------------------------------------

CSV_PATH = os.path.join("data", "montreal_weather.csv")

if not os.path.exists(CSV_PATH):
    print("data/montreal_weather.csv not found — fetching from Open-Meteo...")
    result = subprocess.run(
        [sys.executable, os.path.join("src", "fetch_data.py")],
        check=True,
    )
    print()
else:
    print(f"Found {CSV_PATH}")

# ---------------------------------------------------------------------------
# Step 2: load data
# ---------------------------------------------------------------------------

print("Loading data...")
df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
print(f"  {len(df)} rows, columns: {list(df.columns)}")
print()

# ---------------------------------------------------------------------------
# Step 3: evaluate all horizons and print results table
# ---------------------------------------------------------------------------

from src.validate import evaluate_all_horizons

print("Evaluating all horizons (h = 1, 3, 6, 12, 24, 48)...")
results = evaluate_all_horizons(df)

header = f"{'h':>4} | {'Model RMSE':>10} {'Model MAE':>10} | {'Persist RMSE':>12} {'Persist MAE':>11}"
sep = "-" * len(header)
print(sep)
print(header)
print(sep)
for h, scores in results.items():
    print(
        f"{h:>4} | {scores['model_rmse']:>10.3f} {scores['model_mae']:>10.3f} "
        f"| {scores['persist_rmse']:>12.3f} {scores['persist_mae']:>11.3f}"
    )
print(sep)
print()

# ---------------------------------------------------------------------------
# Step 4: prediction plots for summer and winter
# ---------------------------------------------------------------------------

from src.plot import plot_predictions, plot_error_table

os.makedirs("figures", exist_ok=True)

for h in [1, 24]:
    for season in ["summer", "winter"]:
        print(f"Plotting predictions  h={h}h  {season}...")
        plot_predictions(df, horizon=h, season=season)
print()

# ---------------------------------------------------------------------------
# Step 5: error table figure
# ---------------------------------------------------------------------------

print("Generating error table figure...")
plot_error_table(results)
print()

print("All done. Figures saved to figures/")
