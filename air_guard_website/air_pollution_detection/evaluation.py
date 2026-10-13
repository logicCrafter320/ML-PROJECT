import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    silhouette_score
)

from preprocessing import load_and_preprocess
from dbscan_model import run_dbscan
from isolation_forest_model import run_isolation_forest


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(model_name, y_true, y_pred):

    print("\n==========================================")
    print(model_name)
    print("==========================================")

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["Normal", "Anomaly"],
            zero_division=0
        )
    )

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }


# ============================================================
# DBSCAN SILHOUETTE SCORE
# ============================================================

def calculate_dbscan_silhouette(X, labels):

    print("\n==========================================")
    print("DBSCAN SILHOUETTE SCORE")
    print("==========================================")

    # Remove DBSCAN noise points (-1)
    mask = labels != -1

    X_clustered = X[mask]
    cluster_labels = labels[mask]

    # Silhouette requires at least 2 clusters
    number_of_clusters = len(set(cluster_labels))

    if number_of_clusters >= 2:

        score = silhouette_score(
            X_clustered,
            cluster_labels
        )

        print(
            "Silhouette Score:",
            round(score, 4)
        )

        return score

    else:

        print(
            "Silhouette score cannot be calculated."
        )

        return np.nan


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n==========================================")
    print("M5 MODEL EVALUATION")
    print("==========================================")

    # Load and preprocess data
    df, X, y_true, scaler = load_and_preprocess(
        "data/air_pollution_dataset.csv"
    )

    # --------------------------------------------------------
    # DBSCAN
    # --------------------------------------------------------

    dbscan_labels, dbscan_anomalies = run_dbscan(X)

    # --------------------------------------------------------
    # Isolation Forest
    # --------------------------------------------------------

    if_predictions, if_anomalies, if_scores = \
        run_isolation_forest(X)

    # --------------------------------------------------------
    # Evaluate DBSCAN
    # --------------------------------------------------------

    dbscan_results = evaluate_model(
        "DBSCAN",
        y_true,
        dbscan_anomalies
    )

    # --------------------------------------------------------
    # Evaluate Isolation Forest
    # --------------------------------------------------------

    isolation_results = evaluate_model(
        "Isolation Forest",
        y_true,
        if_anomalies
    )

    # --------------------------------------------------------
    # DBSCAN clustering evaluation
    # --------------------------------------------------------

    silhouette = calculate_dbscan_silhouette(
        X.values,
        dbscan_labels
    )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    print("\n==========================================")
    print("MODEL COMPARISON")
    print("==========================================")

    print("\nDBSCAN:")
    print(dbscan_results)

    print("\nIsolation Forest:")
    print(isolation_results)

    print("\nDBSCAN Silhouette Score:")
    print(silhouette)