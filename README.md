# MECH 309 — Montreal Temperature Forecasting

Linear regression model that forecasts hourly air temperature in Montreal 
up to 48 hours ahead. Built for MECH 309 at McGill, Winter 2026.

## How it works

For each forecast horizon (1, 3, 6, 12, 24, 48 hours), a separate linear model 
is trained on lagged temperature and wind speed values plus sine/cosine terms 
for the hour of day. Parameters are solved directly using the normal equations. 
Results are compared against a persistence baseline.

## Setup
```bash
pip install -r requirements.txt
```

## Run
```bash
python main.py
```

This fetches the data if needed, prints the RMSE/MAE table for all horizons, 
and saves plots to `figures/`.

## Project structure
```
├── main.py              # run this
├── GetWeatherData.py    # provided by Prof. Nicolai
├── src/
│   ├── fetch_data.py    # pulls 2024 Montreal weather from Open-Meteo
│   ├── preprocess.py    # builds design matrix with lags and diurnal terms
│   ├── model.py         # normal equations solver
│   ├── validate.py      # RMSE/MAE evaluation across all horizons
│   └── plot.py          # generates prediction plots and error table
├── data/                # CSV auto-generated on first run
└── figures/             # output plots
```

## Authors
Cyril Zovighian & Aldric Tascher
