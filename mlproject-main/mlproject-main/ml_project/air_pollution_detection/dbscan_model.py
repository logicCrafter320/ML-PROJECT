from sklearn.cluster import DBSCAN

from preprocessing import load_and_preprocess


def run_dbscan(X):

    print("\n==========================================")
    print("DBSCAN ANOMALY DETECTION")
    print("==========================================")

    # Create DBSCAN model
    dbscan = DBSCAN(
        eps=0.8,
        min_samples=10
    )

    # Train and predict clusters
    labels = dbscan.fit_predict(X)

    # DBSCAN uses -1 to represent noise
    # Noise points are treated as anomalies
    anomaly_labels = (labels == -1).astype(int)

    # Count clusters
    unique_labels = set(labels)

    number_of_clusters = len(unique_labels)

    if -1 in unique_labels:
        number_of_clusters -= 1

    number_of_anomalies = anomaly_labels.sum()

    print("DBSCAN completed successfully!")

    print("Number of clusters:", number_of_clusters)
    print("Number of anomalies:", number_of_anomalies)

    print("\nCluster distribution:")
    print(
        {
            label: list(labels).count(label)
            for label in sorted(unique_labels)
        }
    )

    return labels, anomaly_labels


if __name__ == "__main__":

    # Load preprocessed data
    df, X, y_true, scaler = load_and_preprocess(
        "data/air_pollution_dataset.csv"
    )

    # Run DBSCAN
    labels, anomaly_labels = run_dbscan(X)

    print("\nFirst 20 DBSCAN labels:")
    print(labels[:20])

    print("\nFirst 20 anomaly labels:")
    print(anomaly_labels[:20])