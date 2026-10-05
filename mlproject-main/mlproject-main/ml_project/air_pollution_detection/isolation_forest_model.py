from sklearn.ensemble import IsolationForest

from preprocessing import load_and_preprocess


def run_isolation_forest(X):

    print("\n==========================================")
    print("ISOLATION FOREST ANOMALY DETECTION")
    print("==========================================")

    # Create Isolation Forest model
    model = IsolationForest(
        n_estimators=200,
        contamination=0.07,
        random_state=42,
        n_jobs=-1
    )

    # Train model and predict
    predictions = model.fit_predict(X)

    # Isolation Forest:
    # +1 = Normal
    # -1 = Anomaly

    anomaly_labels = (predictions == -1).astype(int)

    # Calculate anomaly scores
    anomaly_scores = -model.decision_function(X)

    number_of_anomalies = anomaly_labels.sum()

    print("Isolation Forest completed successfully!")

    print("Number of anomalies:", number_of_anomalies)

    print("\nPrediction distribution:")
    print({
        "Normal": int((anomaly_labels == 0).sum()),
        "Anomaly": int((anomaly_labels == 1).sum())
    })

    return predictions, anomaly_labels, anomaly_scores


if __name__ == "__main__":

    # Load preprocessed data
    df, X, y_true, scaler = load_and_preprocess(
        "data/air_pollution_dataset.csv"
    )

    # Run Isolation Forest
    predictions, anomaly_labels, anomaly_scores = \
        run_isolation_forest(X)

    print("\nFirst 20 predictions:")
    print(predictions[:20])

    print("\nFirst 20 anomaly labels:")
    print(anomaly_labels[:20])

    print("\nFirst 20 anomaly scores:")
    print(anomaly_scores[:20])