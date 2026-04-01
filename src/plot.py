from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from GetWeatherData import add_lags
from src.model import fit_model, predict
from src.validate import evaluate_all_horizons

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SEASON_MONTHS = {
    "summer": [6, 7, 8],
    "winter": [12, 1, 2],
}


N_LAGS = 4


def _build_seasonal_split(
    df: pd.DataFrame, horizon: int, season: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Filter df to season, build lag features, and do an 80/20 train/val split.

    Returns X_train, y_train, X_val, y_val, val_index.
    """
    months = SEASON_MONTHS[season]
    df_s = df[df.index.month.isin(months)].copy()

    df_s["__target"] = df_s["T"].shift(-horizon)
    df_s = add_lags(df_s, "T", list(range(1, N_LAGS + 1)))
    df_s = add_lags(df_s, "W", list(range(1, N_LAGS + 1)))
    df_s = df_s.dropna()

    lag_T = [f"T_lag{L}" for L in range(1, N_LAGS + 1)]
    lag_W = [f"W_lag{L}" for L in range(1, N_LAGS + 1)]
    feature_cols = lag_T + lag_W + ["sin_day", "cos_day"]

    X = df_s[feature_cols].to_numpy(dtype=float)
    X = np.hstack([X, np.ones((len(X), 1), dtype=float)])
    y = df_s["__target"].to_numpy(dtype=float)
    idx = df_s.index

    n_train = int(len(X) * 0.8)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:], idx[n_train:]


def plot_predictions(df: pd.DataFrame, horizon: int, season: str) -> None:
    """Plot observed vs predicted temperature for a season using an 80/20 split
    within that season's data.

    Parameters
    ----------
    df      : preprocessed DataFrame with at least columns T, W, sin_day, cos_day
    horizon : forecast horizon in hours
    season  : 'summer' (Jun-Aug) or 'winter' (Dec-Feb)
    """
    if season not in SEASON_MONTHS:
        raise ValueError(f"season must be 'summer' or 'winter', got {season!r}")

    X_train, y_train, X_val, y_val, val_idx = _build_seasonal_split(df, horizon, season)

    if len(X_val) == 0:
        print(f"  No data for {season} at h={horizon}h — skipping.")
        return

    theta = fit_model(X_train, y_train)
    y_pred = predict(X_val, theta)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(val_idx, y_val,  label="Observed",
            linewidth=0.9, color="steelblue")
    ax.plot(val_idx, y_pred, label=f"Predicted (h={horizon}h)",
            linewidth=0.9, color="tomato", alpha=0.85)
    ax.set_title(f"Temperature forecast — h={horizon}h  |  {season.capitalize()}")
    ax.set_xlabel("Date")
    ax.set_ylabel("T [°C]")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.autofmt_xdate()
    fig.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    out = os.path.join(FIGURES_DIR, f"pred_h{horizon}_{season}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


def plot_error_table(results_dict: dict) -> None:
    """Render a styled matplotlib table of RMSE / MAE for all horizons.

    Parameters
    ----------
    results_dict : output of evaluate_all_horizons — keys are horizons (int),
                   values are dicts with model_rmse, model_mae,
                   persist_rmse, persist_mae.
    """
    horizons = sorted(results_dict.keys())

    col_labels = ["h (hours)", "Model RMSE", "Model MAE", "Persist RMSE", "Persist MAE"]
    rows = []
    for h in horizons:
        s = results_dict[h]
        rows.append([
            str(h),
            f"{s['model_rmse']:.3f}",
            f"{s['model_mae']:.3f}",
            f"{s['persist_rmse']:.3f}",
            f"{s['persist_mae']:.3f}",
        ])

    fig, ax = plt.subplots(figsize=(9, 0.6 + 0.45 * len(rows)))
    ax.axis("off")

    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.5)

    # Header styling
    header_color = "#2c4770"
    for j in range(len(col_labels)):
        cell = tbl[0, j]
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", fontweight="bold")

    # Alternating row colours
    for i in range(1, len(rows) + 1):
        row_color = "#eaf0fb" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(row_color)

    ax.set_title("Forecast Error Summary", fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    out = os.path.join(FIGURES_DIR, "error_table.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "montreal_weather.csv")
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    for h in [1, 24]:
        for season in ["summer", "winter"]:
            print(f"Plotting h={h}h, {season}...")
            plot_predictions(df, horizon=h, season=season)

    print("Generating error table...")
    results = evaluate_all_horizons(df)
    plot_error_table(results)
    print("Done.")
