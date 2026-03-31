import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from GetWeatherData import add_lags, split_train_val

VAL_HOURS = 2196


def build_design_matrix(df: pd.DataFrame, h: int, n_lags: int = 4):
    """
    Build design matrix X and target vector y for a forecast horizon of h hours.

    Parameters
    ----------
    df      : preprocessed DataFrame with at least columns T, W, sin_day, cos_day
    h       : forecast horizon in hours (y[k] = T[k+h])
    n_lags  : number of lag steps to add for T and W (lags 1..n_lags)

    Returns
    -------
    X_train, y_train, X_val, y_val : numpy arrays
    """
    df = df.copy()

    # Target: T shifted forward by h (so row k holds the value h steps ahead)
    df["target"] = df["T"].shift(-h)

    # Add lagged features for T and W
    df = add_lags(df, "T", list(range(1, n_lags + 1)))
    df = add_lags(df, "W", list(range(1, n_lags + 1)))

    # Drop rows that have NaN from shifting / lagging
    df = df.dropna()

    # Feature columns: T_lag1..n, W_lag1..n, sin_day, cos_day, bias
    lag_T_cols = [f"T_lag{L}" for L in range(1, n_lags + 1)]
    lag_W_cols = [f"W_lag{L}" for L in range(1, n_lags + 1)]
    feature_cols = lag_T_cols + lag_W_cols + ["sin_day", "cos_day"]

    X = df[feature_cols].to_numpy(dtype=float)
    bias = np.ones((len(X), 1), dtype=float)
    X = np.hstack([X, bias])

    y = df["target"].to_numpy(dtype=float)

    # Split into train / validation using the professor's helper
    df_features = pd.DataFrame(X, index=df.index)
    df_features["__y__"] = y

    train_df, val_df = split_train_val(df_features, val_hours=VAL_HOURS)

    X_train = train_df.drop(columns="__y__").to_numpy(dtype=float)
    y_train = train_df["__y__"].to_numpy(dtype=float)
    X_val   = val_df.drop(columns="__y__").to_numpy(dtype=float)
    y_val   = val_df["__y__"].to_numpy(dtype=float)

    return X_train, y_train, X_val, y_val


if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "montreal_weather.csv")
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    for h in [1, 6, 24]:
        X_train, y_train, X_val, y_val = build_design_matrix(df, h=h)
        print(
            f"h={h:>2}h | X_train {X_train.shape} y_train {y_train.shape} "
            f"| X_val {X_val.shape} y_val {y_val.shape}"
        )
