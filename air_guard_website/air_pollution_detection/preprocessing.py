import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "PM2.5",
    "PM10",
    "CO",
    "NO2",
    "SO2",
    "O3",
    "Temperature",
    "Humidity"
]


def load_and_preprocess(filepath):
    print("Loading dataset...")

    df = pd.read_csv(filepath)

    print("Dataset shape:", df.shape)

    # Keep ground-truth labels separately for M5 evaluation
    y_true = df["IsAnomaly"].copy()

    # Select ML features
    X = df[FEATURE_COLUMNS].copy()

    # Handle missing values using median
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    X_processed = pd.DataFrame(
        X_scaled,
        columns=FEATURE_COLUMNS
    )

    print("\nPreprocessing completed!")
    print("Features:", len(FEATURE_COLUMNS))
    print("Missing values after imputation:", X_processed.isnull().sum().sum())

    return df, X_processed, y_true, scaler


if __name__ == "__main__":

    df, X_processed, y_true, scaler = load_and_preprocess(
        "data/air_pollution_dataset.csv"
    )

    print("\nProcessed data:")
    print(X_processed.head())