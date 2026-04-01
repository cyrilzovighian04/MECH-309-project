from __future__ import annotations

import numpy as np


def fit_model(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """Fit a linear model via the normal equations.

    Solves (X^T X) theta = X^T y directly with np.linalg.solve.

    Parameters
    ----------
    X_train : np.ndarray, shape (n_samples, n_features)
    y_train : np.ndarray, shape (n_samples,)

    Returns
    -------
    theta : np.ndarray, shape (n_features,)
    """
    return np.linalg.solve(X_train.T @ X_train, X_train.T @ y_train)


def predict(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Predict targets for feature matrix X given parameter vector theta.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    theta : np.ndarray, shape (n_features,)

    Returns
    -------
    np.ndarray, shape (n_samples,)
    """
    return X @ theta


if __name__ == "__main__":
    import os
    import sys

    import pandas as pd

    # Ensure project root is on the path when running this file directly.
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from src.preprocess import build_design_matrix

    df = pd.read_csv("data/montreal_weather.csv", index_col=0, parse_dates=True)

    X_train, y_train, X_val, y_val = build_design_matrix(df, h=1)
    theta = fit_model(X_train, y_train)

    print(f"theta shape : {theta.shape}")
    print(f"First 5 predictions : {predict(X_train[:5], theta)}")
