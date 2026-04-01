# MECH 309 — Montreal Temperature Forecasting

A data-driven temperature forecasting model for Montreal built for MECH 309 (Numerical Methods) at McGill University. The project fetches real hourly weather data from the Open-Meteo archive API, fits a linear regression model using the normal equations, and evaluates forecast accuracy across six time horizons.

---

## Project description

The model predicts the 2 m air temperature in Montreal up to 48 hours ahead. For each forecast horizon `h`, a separate linear model is trained on lagged temperature and wind-speed features together with sine/cosine encodings of the hour of day. Coefficients are solved analytically via the normal equations (X^T X θ = X^T y). Performance is compared against a persistence baseline (i.e. "the temperature h hours from now equals the temperature now").

Forecast horizons evaluated: **1, 3, 6, 12, 24, and 48 hours**.

---

## Repository structure

```
MECH-309-project/
├── main.py               # End-to-end entry point (fetch → evaluate → plot)
├── GetWeatherData.py     # Provided course module: API fetch, preprocessing, lag/split helpers
├── requirements.txt      # Python dependencies
├── data/
│   └── montreal_weather.csv   # Cached dataset (auto-fetched if missing)
├── figures/              # Output plots (created automatically)
└── src/
    ├── fetch_data.py     # Downloads and caches the dataset
    ├── preprocess.py     # Builds the design matrix and train/val split
    ├── model.py          # Normal-equations solver and predict function
    ├── validate.py       # RMSE/MAE metrics and multi-horizon evaluation
    └── plot.py           # Prediction time-series and error-summary table plots
```

---

## Installation

Python 3.10+ is required.

```bash
# 1. Clone the repository
git clone <repo-url>
cd MECH-309-project

# 2a. Using a virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2b. Or with conda
conda create -n mech309 python=3.11 numpy pandas matplotlib requests
conda activate mech309
```

---

## Running the code

Run the full pipeline from the project root:

```bash
python main.py
```

This will:
1. Fetch and cache weather data (if `data/montreal_weather.csv` is missing)
2. Evaluate the linear model and persistence baseline for all six horizons and print a results table
3. Save prediction time-series plots for h = 1 h and h = 24 h (summer and winter) to `figures/`
4. Save an error-summary table image to `figures/`

Individual modules can also be run in isolation for development and debugging:

```bash
python src/fetch_data.py     # re-download the dataset
python src/preprocess.py     # inspect design-matrix shapes
python src/model.py          # quick model smoke test
python src/validate.py       # print the full metrics table
python src/plot.py           # regenerate all figures
```

---

## Source files

| File | Description |
|---|---|
| `src/fetch_data.py` | Calls the Open-Meteo archive API via `GetWeatherData.py` to download 2024 hourly weather data for Montreal and saves the preprocessed result to `data/montreal_weather.csv`. |
| `src/preprocess.py` | Builds the design matrix `X` for a given forecast horizon `h`. Features are lagged temperature (T_lag1–4), lagged wind speed (W_lag1–4), and sine/cosine hour-of-day encodings, plus a bias column. Uses `split_train_val` from the course module to hold out the last ~3 months as the validation set. |
| `src/model.py` | Fits a linear model by solving the normal equations with `numpy.linalg.solve`. Also exposes a `predict` function (matrix–vector product). No external ML libraries are used. |
| `src/validate.py` | Computes RMSE and MAE for both the linear model and the persistence baseline across all six forecast horizons. Returns results as a plain dictionary. |
| `src/plot.py` | Generates two types of figures: time-series plots of observed vs predicted temperature (split by season and horizon) and a styled error-summary table. All figures are written to `figures/`. |

---

## Data

Weather data is fetched automatically from the [Open-Meteo Historical Weather API](https://open-meteo.com/) covering **2024-01-01 to 2024-12-31** for Montreal (45.50 °N, 73.57 °W). Variables retrieved: 2 m temperature, 10 m wind speed and direction, relative humidity, surface pressure, precipitation, and cloud cover.

If `data/montreal_weather.csv` is already present, `main.py` skips the fetch step and loads it directly. Delete the file to force a fresh download.
