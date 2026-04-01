from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from src.preprocess import build_design_matrix
from src.model import fit_model, predict


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


# ---------------------------------------------------------------------------
# Persistence baseline
# ---------------------------------------------------------------------------

def persistence_baseline(y_series: np.ndarray, h: int) -> np.ndarray:
    """Return the persistence forecast T_{k+h} ≈ T_k.

    The output is aligned with y_series: element k of the returned array is
    the persistence prediction for the target at position k, i.e. y_series[k-h]
    shifted forward by h.  The first h elements (which have no look-back) are
    filled with y_series[0].
    """
    n = len(y_series)
    pred = np.empty(n, dtype=float)
    pred[:h] = y_series[0]
    pred[h:] = y_series[:n - h]
    return pred


# ---------------------------------------------------------------------------
# Evaluate across horizons
# ---------------------------------------------------------------------------

def evaluate_all_horizons(
    df: pd.DataFrame,
    horizons: list[int] | None = None,
) -> dict[int, dict[str, float]]:
    """Evaluate the linear regression model and persistence baseline for each horizon.

    Parameters
    ----------
    df       : preprocessed DataFrame (output of preprocess.preprocess)
    horizons : list of forecast horizons in hours

    Returns
    -------
    results : dict  {h: {"model_rmse", "model_mae", "persist_rmse", "persist_mae"}}
    """
    if horizons is None:
        horizons = [1, 3, 6, 12, 24, 48]

    results = {}
    for h in horizons:
        X_train, y_train, X_val, y_val = build_design_matrix(df, h=h)

        # Fit and predict
        theta = fit_model(X_train, y_train)
        y_pred = predict(X_val, theta)

        # Persistence: use the last column of T lags (lag-1 of T) as T_k proxy
        # T_lag1 is the first feature column in X_val
        t_k = X_val[:, 0]  # T_lag1 — best available "current" T on the val set
        y_persist = persistence_baseline(t_k, h=0)  # already aligned (T_k predicts T_{k+h})

        results[h] = {
            "model_rmse":   rmse(y_val, y_pred),
            "model_mae":    mae(y_val, y_pred),
            "persist_rmse": rmse(y_val, y_persist),
            "persist_mae":  mae(y_val, y_persist),
        }

    return results


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "montreal_weather.csv")
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

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
