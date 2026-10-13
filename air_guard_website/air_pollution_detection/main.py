import pandas as pd

from preprocessing import load_and_preprocess
from dbscan_model import run_dbscan
from isolation_forest_model import run_isolation_forest
from evaluation import evaluate_model
from evaluation import calculate_dbscan_silhouette


# ============================================================
# PROJECT HEADER
# ============================================================

print("\n==============================================")
print("   AIR POLLUTION ANOMALY DETECTION")
print("   DBSCAN + ISOLATION FOREST")
print("==============================================\n")


# ============================================================
# 1. DATA PREPROCESSING
# ============================================================

print("STEP 1: DATA PREPROCESSING")

df, X, y_true, scaler = load_and_preprocess(
    "data/air_pollution_dataset.csv"
)


# ============================================================
# 2. DBSCAN
# ============================================================

print("\nSTEP 2: DBSCAN")

dbscan_labels, dbscan_anomalies = run_dbscan(X)


# ============================================================
# 3. ISOLATION FOREST
# ============================================================

print("\nSTEP 3: ISOLATION FOREST")

if_predictions, if_anomalies, if_scores = \
    run_isolation_forest(X)


# ============================================================
# 4. M5 — EVALUATION
# ============================================================

print("\nSTEP 4: MODEL EVALUATION")

dbscan_results = evaluate_model(
    "DBSCAN",
    y_true,
    dbscan_anomalies
)

isolation_results = evaluate_model(
    "Isolation Forest",
    y_true,
    if_anomalies
)


# ============================================================
# 5. DBSCAN SILHOUETTE SCORE
# ============================================================

print("\nSTEP 5: DBSCAN CLUSTERING EVALUATION")

silhouette_score = calculate_dbscan_silhouette(
    X.values,
    dbscan_labels
)


# ============================================================
# 6. CREATE RESULT DATASET
# ============================================================

results = df.copy()

results["DBSCAN_Label"] = dbscan_labels

results["DBSCAN_Anomaly"] = dbscan_anomalies

results["IsolationForest_Anomaly"] = if_anomalies

results["IsolationForest_Score"] = if_scores


# ============================================================
# 7. SAVE ANOMALY RESULTS
# ============================================================

results.to_csv(
    "anomaly_results.csv",
    index=False
)


# ============================================================
# 8. CREATE MODEL COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame([
    dbscan_results,
    isolation_results
])

comparison.to_csv(
    "model_comparison.csv",
    index=False
)


# ============================================================
# 9. FINAL SUMMARY
# ============================================================

print("\n==============================================")
print("             FINAL PROJECT SUMMARY")
print("==============================================")

print("\nDataset:")
print("Total records:", len(df))

print("Ground-truth anomalies:", int(y_true.sum()))

print("\nDBSCAN:")
print("Detected anomalies:", int(dbscan_anomalies.sum()))

print("\nIsolation Forest:")
print("Detected anomalies:", int(if_anomalies.sum()))

print("\nDBSCAN Silhouette Score:",
      silhouette_score)

print("\nModel Comparison:")
print(comparison.to_string(index=False))

print("\n==============================================")
print("RESULT FILES CREATED")
print("==============================================")

print("1. anomaly_results.csv")
print("2. model_comparison.csv")

print("\nML PROJECT EXECUTION COMPLETED!")