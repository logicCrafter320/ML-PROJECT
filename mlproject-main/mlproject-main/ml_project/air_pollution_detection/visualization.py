import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# LOAD RESULTS
# ============================================================

df = pd.read_csv("anomaly_results.csv")


print("==========================================")
print("AIR POLLUTION VISUALIZATION")
print("==========================================")


# ============================================================
# GRAPH 1 — PM2.5 ANOMALIES USING DBSCAN
# ============================================================

plt.figure(figsize=(12, 6))

normal = df["DBSCAN_Anomaly"] == 0
anomaly = df["DBSCAN_Anomaly"] == 1

plt.scatter(
    df.index[normal],
    df.loc[normal, "PM2.5"],
    label="Normal",
    alpha=0.6
)

plt.scatter(
    df.index[anomaly],
    df.loc[anomaly, "PM2.5"],
    label="DBSCAN Anomaly",
    marker="x",
    s=60
)

plt.xlabel("Observation")
plt.ylabel("PM2.5")
plt.title("DBSCAN Detection of PM2.5 Anomalies")
plt.legend()
plt.grid(alpha=0.2)

plt.savefig(
    "dbscan_pm25_anomalies.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# GRAPH 2 — ISOLATION FOREST ANOMALY SCORES
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    df.index,
    df["IsolationForest_Score"],
    linewidth=1
)

plt.xlabel("Observation")
plt.ylabel("Anomaly Score")
plt.title("Isolation Forest Anomaly Scores")
plt.grid(alpha=0.2)

plt.savefig(
    "isolation_forest_scores.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# GRAPH 3 — MODEL COMPARISON
# ============================================================

actual_count = df["IsAnomaly"].sum()
dbscan_count = df["DBSCAN_Anomaly"].sum()
isolation_count = df["IsolationForest_Anomaly"].sum()

models = [
    "Actual",
    "DBSCAN",
    "Isolation Forest"
]

counts = [
    actual_count,
    dbscan_count,
    isolation_count
]

plt.figure(figsize=(9, 6))

bars = plt.bar(
    models,
    counts
)

plt.xlabel("Method")
plt.ylabel("Number of Anomalies")
plt.title("Anomaly Detection Comparison")

# Add values above bars
for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        str(int(height)),
        ha="center",
        va="bottom"
    )

plt.grid(
    axis="y",
    alpha=0.2
)

plt.savefig(
    "anomaly_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# GRAPH 4 — ACTUAL VS DETECTED ANOMALIES
# ============================================================

plt.figure(figsize=(12, 6))

plt.scatter(
    df.index,
    df["PM2.5"],
    c=df["IsAnomaly"],
    alpha=0.7
)

plt.xlabel("Observation")
plt.ylabel("PM2.5")
plt.title("Ground-Truth Air Pollution Anomalies")

plt.grid(alpha=0.2)

plt.savefig(
    "ground_truth_anomalies.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print("\nVisualization completed!")

print("\nGenerated files:")

print("1. dbscan_pm25_anomalies.png")
print("2. isolation_forest_scores.png")
print("3. anomaly_comparison.png")
print("4. ground_truth_anomalies.png")