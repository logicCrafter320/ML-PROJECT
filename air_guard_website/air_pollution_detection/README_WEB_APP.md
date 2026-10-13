# Air Guard — Web Dashboard

A Streamlit website built around the existing DBSCAN + Isolation Forest project.

## Features
- Upload a CSV file from the sidebar.
- Includes the bundled sample dataset by default.
- Cleans numeric data, imputes missing values with medians, and standardizes the eight pollution/weather features.
- Runs DBSCAN and Isolation Forest with adjustable settings.
- Interactive charts for distributions, feature comparisons, anomaly scores, and model comparisons.
- Shows evaluation metrics when a binary `IsAnomaly` ground-truth column is present.
- Filters results and downloads analyzed or filtered data as CSV.

## Required CSV columns
`PM2.5`, `PM10`, `CO`, `NO2`, `SO2`, `O3`, `Temperature`, `Humidity`

Optional evaluation column: `IsAnomaly` (`0` = normal, `1` = anomaly). Common variants such as `pm25`, `is_anomaly`, `anomaly`, `temp`, and `relative humidity` are recognized.

## Run in Windows / VS Code
1. Extract the ZIP file.
2. Open the extracted `air_pollution_detection` folder in VS Code (the folder containing `app.py` and `requirements.txt`).
3. Open **Terminal → New Terminal**.
4. (Recommended) Create a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks activation, use the install/run commands below without activating, or run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal and activate again.
5. Install packages:
   ```powershell
   python -m pip install -r requirements.txt
   ```
6. Launch the website:
   ```powershell
   python -m streamlit run app.py
   ```
7. Streamlit prints a local URL, usually `http://localhost:8501`. Open it in your browser.

## Troubleshooting
- If `streamlit` isn't found, use `python -m streamlit run app.py`.
- If a package is missing, rerun `python -m pip install -r requirements.txt`.
- If columns are missing, check spelling against the required column list above.
- Upload a CSV with numeric values for all eight features. Missing values are allowed; entirely empty feature rows are skipped.
- Model flags are anomaly-detection outputs, not an official health or regulatory judgement.
