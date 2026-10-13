# AirGuard AI

## Air Pollution Anomaly Detection Using DBSCAN and Isolation Forest

AirGuard AI is a machine learning project designed to identify unusual patterns in air pollution data using unsupervised anomaly detection algorithms. The system analyses environmental parameters, including particulate matter, gaseous pollutants, temperature, and humidity, to identify observations that differ from normal patterns.

The project uses two machine learning algorithms, **DBSCAN** and **Isolation Forest**, to detect potential anomalies and compare their results. Data visualizations and evaluation metrics help users understand the model outputs.

---

## 1. Project Overview

Air pollution is a major environmental concern that can affect human health, ecosystems, and overall environmental quality. Monitoring pollution measurements and identifying unusual changes can help highlight observations that require further investigation.

Traditional monitoring systems often rely on predefined thresholds. However, unusual pollution patterns may involve combinations of multiple environmental parameters rather than a single measurement exceeding a fixed limit.

AirGuard AI explores how unsupervised machine learning can identify such patterns by analysing environmental measurements together.

The system processes air pollution data, applies anomaly detection algorithms, compares model predictions, and generates visualizations to support analysis.

**Project Title:** Air Pollution Anomaly Detection Using DBSCAN and Isolation Forest

**Project Name:** AirGuard AI

**Domain:** Machine Learning, Environmental Monitoring, Data Analytics

**Approach:** Unsupervised Anomaly Detection

---

## 2. Problem Statement

Air pollution datasets contain measurements of several pollutants and environmental conditions. Some observations may differ significantly from the usual patterns in the data.

Identifying these unusual observations manually can be difficult, particularly when multiple parameters must be considered simultaneously.

The objective of this project is to develop a machine learning-based system that analyses air pollution measurements and identifies potential anomalies using density-based clustering and isolation-based detection techniques.

---

## 3. Project Objectives

The main objectives of AirGuard AI are:

* Analyse environmental data containing multiple air pollution parameters.
* Preprocess numerical data before applying machine learning algorithms.
* Detect unusual observations using DBSCAN.
* Detect potential anomalies using Isolation Forest.
* Compare the results produced by both algorithms.
* Evaluate model predictions using appropriate metrics when reliable reference labels are available.
* Generate visualizations to help interpret the detected anomalies.
* Demonstrate how unsupervised machine learning can support environmental data analysis.

---

## 4. Dataset Description

The project uses a structured CSV dataset containing environmental measurements.

The previously inspected dataset was recorded as containing approximately 1,800 observations, eight environmental features, and a ground-truth anomaly label. These figures should be verified against the current dataset before being reported as final statistics.

### Environmental Features

| Feature     | Full Form              | Description                                                                                           |
| ----------- | ---------------------- | ----------------------------------------------------------------------------------------------------- |
| PM2.5       | Particulate Matter 2.5 | Measures fine airborne particles approximately 2.5 micrometres or smaller in diameter.                |
| PM10        | Particulate Matter 10  | Measures airborne particles approximately 10 micrometres or smaller in diameter.                      |
| CO          | Carbon Monoxide        | Represents the measured concentration of carbon monoxide.                                             |
| NO2         | Nitrogen Dioxide       | Represents the measured concentration of nitrogen dioxide.                                            |
| SO2         | Sulphur Dioxide        | Represents the measured concentration of sulphur dioxide.                                             |
| O3          | Ozone                  | Represents the measured concentration of ozone, typically ground-level ozone in air quality datasets. |
| Temperature | Temperature            | Represents the surrounding atmospheric temperature.                                                   |
| Humidity    | Humidity               | Represents the moisture level in the air, commonly expressed as relative humidity.                    |

The measurement units depend on how the dataset was created. They should be confirmed from the data source or generation script.

### Ground-Truth Label

If present, the `IsAnomaly` column provides reference labels for evaluating predictions.

A common encoding is:

* `0` — Normal observation.
* `1` — Anomalous observation.

The actual column name and encoding must be verified in the CSV.

The label should be excluded from the input features when fitting the unsupervised models. It can be used afterward to evaluate their predictions.

### Dataset File

Expected path:

```text
data/air_pollution_dataset.csv
```

---

## 5. Machine Learning Algorithms

### 5.1 DBSCAN

DBSCAN stands for **Density-Based Spatial Clustering of Applications with Noise**.

It is an unsupervised clustering algorithm that groups observations according to the density of nearby data points.

Observations that do not belong to sufficiently dense regions may be classified as noise.

In AirGuard AI, DBSCAN analyses the environmental features to identify observations that differ from the density patterns found in the dataset.

**Main characteristics:**

* Groups observations based on density.
* Does not require the number of clusters to be specified in advance.
* Can identify noise points.
* Can help detect unusual combinations of environmental measurements.

DBSCAN commonly assigns the label `-1` to observations identified as noise.

A noise observation is a potential anomaly, but it is not automatically proof of dangerous pollution.

### 5.2 Isolation Forest

Isolation Forest is an unsupervised anomaly detection algorithm.

It works by repeatedly dividing the feature space using randomly selected features and split values. Observations that are easier to isolate tend to be considered more anomalous.

In AirGuard AI, Isolation Forest examines the environmental measurements and identifies observations that differ from the broader data distribution.

**Main characteristics:**

* Detects unusual observations using random isolation trees.
* Can analyse multiple numerical features together.
* Produces anomaly scores and predictions.
* Provides an approach to anomaly detection that differs from density-based clustering.

The model's contamination parameter helps establish the decision threshold. It represents an assumed proportion of outliers, not a guarantee about the true anomaly rate.

### 5.3 Why Use Both Algorithms?

DBSCAN and Isolation Forest use different approaches.

| DBSCAN                                                    | Isolation Forest                                                     |
| --------------------------------------------------------- | -------------------------------------------------------------------- |
| Density-based method                                      | Isolation-based method                                               |
| Groups observations into clusters and identifies noise    | Identifies observations that are relatively easy to isolate          |
| Sensitive to neighbourhood parameters and feature scaling | Uses random trees and an anomaly decision threshold                  |
| Useful for identifying observations outside dense regions | Useful for identifying unusual observations across multiple features |

Comparing both algorithms helps us understand how different anomaly detection techniques behave on the same dataset.

---

## 6. Data Preprocessing

Data preprocessing prepares the dataset for machine learning.

### 6.1 Loading the Dataset

The CSV file is loaded into a data-processing environment for inspection and analysis.

### 6.2 Handling Missing Values

Missing numerical measurements can interfere with machine learning algorithms.

Median imputation replaces missing values with the median of the corresponding feature.

The preprocessing method used in the current code should be verified before describing it as a confirmed implementation detail.

### 6.3 Feature Scaling

Environmental features may have different numerical ranges.

For example, PM2.5 may have values in the tens or hundreds, while CO may use a much smaller numerical range.

StandardScaler transforms each feature using:

`z = (x - mean) / standard_deviation`

This transformation is particularly important for DBSCAN because it uses distances between observations.

Isolation Forest does not depend on feature scales in the same way as a distance-based algorithm.

### 6.4 Feature Selection

The model input consists of the eight environmental features:

* PM2.5
* PM10
* CO
* NO2
* SO2
* O3
* Temperature
* Humidity

If `IsAnomaly` is present, it should be kept separate from the model input and used only for evaluation.

---

## 7. System Workflow

The overall workflow of AirGuard AI is:

1. Load the air pollution dataset.
2. Inspect the data structure and environmental features.
3. Handle missing values.
4. Scale the numerical features when required by the algorithm.
5. Apply DBSCAN to identify clusters and potential noise observations.
6. Apply Isolation Forest to identify potential anomalies.
7. Compare the predictions generated by both algorithms.
8. Evaluate predictions against reference labels when available.
9. Generate visualizations.
10. Interpret the results to understand unusual environmental patterns.

### Workflow Diagram

```text
       Air Pollution Dataset
                 |
                 v
         Data Preprocessing
                 |
                 v
          Feature Selection
                 |
          +------+------+
          |             |
          v             v
        DBSCAN     Isolation Forest
          |             |
          v             v
    Noise Detection  Anomaly Detection
          |             |
          +------+------+
                 |
                 v
       Compare Predictions
                 |
                 v
      Evaluation and Analysis
                 |
                 v
          Visualizations
```

---

## 8. Evaluation Metrics

When reliable ground-truth labels are available, model predictions can be evaluated using a confusion matrix and standard classification metrics.

### Accuracy

Accuracy measures the proportion of correctly classified observations.

`Accuracy = (TP + TN) / (TP + TN + FP + FN)`

### Precision

Precision measures the proportion of observations predicted as anomalous that are actually anomalous.

`Precision = TP / (TP + FP)`

### Recall

Recall measures the proportion of actual anomalies correctly detected by the model.

`Recall = TP / (TP + FN)`

### F1-Score

The F1-score is the harmonic mean of precision and recall.

`F1-Score = 2 * (Precision * Recall) / (Precision + Recall)`

### Confusion Matrix Terms

* **True Positive (TP):** An anomalous observation correctly identified as anomalous.
* **False Positive (FP):** A normal observation incorrectly identified as anomalous.
* **True Negative (TN):** A normal observation correctly identified as normal.
* **False Negative (FN):** An anomalous observation incorrectly identified as normal.

Accuracy alone may be misleading when anomalies are relatively rare. Precision, recall, and F1-score provide additional information about false alarms and missed anomalies.

These metrics require appropriate reference labels. If the dataset does not contain reliable ground-truth labels, the results should not be presented as verified classification performance.

---

## 9. Data Visualization

Visualizations help interpret the results produced by the anomaly detection algorithms.

The project's previously recorded output files include:

| File                          | Purpose                                                                   |
| ----------------------------- | ------------------------------------------------------------------------- |
| `dbscan_pm25_anomalies.png`   | Recorded as a visualization of DBSCAN results against PM2.5 measurements. |
| `isolation_forest_scores.png` | Recorded as a visualization of Isolation Forest scores.                   |
| `anomaly_comparison.png`      | Recorded as a comparison of anomaly detection results.                    |
| `ground_truth_anomalies.png`  | Recorded as a visualization involving reference anomaly labels.           |

These filenames should be checked against the current project directory and visualization code.

The visualizations can help answer questions such as:

* Which observations were identified as potential anomalies?
* How do DBSCAN and Isolation Forest differ?
* Do the algorithms identify the same observations?
* How do detected anomalies relate to the environmental measurements?
* How do predictions compare with reference labels?

Visualization supports interpretation but does not independently establish the cause of an anomaly.

---

## 10. Technologies Used

The project is based on Python and machine learning tools.

| Technology            | Purpose                                                             |
| --------------------- | ------------------------------------------------------------------- |
| Python                | Main programming language for data processing and machine learning. |
| Pandas                | Loading and analysing structured datasets.                          |
| NumPy                 | Numerical operations.                                               |
| Scikit-learn          | Machine learning algorithms and preprocessing utilities.            |
| Matplotlib            | Creating data visualizations.                                       |
| Seaborn               | Statistical visualizations, if used by the implementation.          |
| Gradio                | Web interface, if included in the current project.                  |
| HTML, CSS, JavaScript | Frontend interface, if included in the current project.             |

The exact dependencies should be confirmed against the project's Python files and dependency list.

---

## 11. Project Structure

The following is an illustrative project structure. Actual filenames and folders may differ.

```text
AirGuard-AI/
│
├── data/
│   └── air_pollution_dataset.csv
│
├── preprocessing.py
├── visualization.py
│
├── app.py
├── requirements.txt
├── README.md
│
└── outputs/
    ├── anomaly_results.csv
    ├── model_comparison.csv
    ├── dbscan_pm25_anomalies.png
    ├── isolation_forest_scores.png
    ├── anomaly_comparison.png
    └── ground_truth_anomalies.png
```

**Note:** Keep only the files and folders that actually exist in your project. The `app.py` and `outputs/` paths above are examples and must not be treated as confirmed paths.

---

## 12. Installation and Setup

### Prerequisites

Install the following:

* Python 3.10 or a compatible version supported by the project's dependencies.
* Visual Studio Code or another Python development environment.
* pip, the Python package installer.

### Step 1: Open the Project

Open the AirGuard AI project folder in Visual Studio Code.

Open the integrated terminal.

### Step 2: Create a Virtual Environment

Run:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

### Step 3: Install Dependencies

If the project contains a `requirements.txt` file, run:

```bash
python -m pip install -r requirements.txt
```

If the file does not exist, install the packages required by the actual implementation.

For a basic Python data-analysis pipeline using the libraries described above:

```bash
python -m pip install pandas numpy scikit-learn matplotlib seaborn
```

Install Gradio only if the project uses it:

```bash
python -m pip install gradio
```

---

## 13. Running the Project

The exact commands depend on the files and execution methods in the current project.

### Inspect the Dataset

If the dataset is stored at `data/air_pollution_dataset.csv`, run:

```bash
python -c "import pandas as pd; df=pd.read_csv('data/air_pollution_dataset.csv'); print('Shape:', df.shape); print(df.head()); print(df.dtypes); print(df.isnull().sum()); print(df.describe().T)"
```

This displays:

* Number of records and columns.
* The first five observations.
* Data types.
* Missing-value counts.
* Summary statistics.

### Generate Visualizations

If `visualization.py` is an executable Python script that generates the plots, run:

```bash
python visualization.py
```

The script must contain the required plotting logic and use the correct dataset and output paths.

If the visualization code is instead called by another module, use the execution command documented in that module.

### Run the Web Application

If the project contains a Gradio application named `app.py`, run:

```bash
python app.py
```

If the frontend uses another framework or entry-point file, follow the corresponding startup instructions.

**Important:** These are conditional execution examples. Confirm the actual script names and commands in your project before using them in a final demonstration.

---

## 14. Expected Results

When executed correctly, the system should:

* Load and process the environmental dataset.
* Analyse the eight environmental features.
* Produce DBSCAN cluster and noise labels.
* Produce Isolation Forest anomaly predictions or scores.
* Compare predictions when the relevant outputs are available.
* Generate the configured visualizations.
* Calculate evaluation metrics if suitable ground-truth labels are available.

The exact number of detected anomalies and the evaluation scores depend on the dataset, preprocessing, model parameters, and decision thresholds.

Results should be measured from the actual execution rather than assumed in advance.

---

## 15. Applications

AirGuard AI demonstrates techniques that may be useful in:

* Air quality data analysis.
* Environmental monitoring research.
* Anomaly detection in sensor measurements.
* Exploratory analysis of pollution patterns.
* Comparing unsupervised machine learning algorithms.
* Educational demonstrations of machine learning.

The project is a prototype for analysing environmental data. It is not, by itself, a certified air quality monitoring or public health warning system.

---

## 16. Limitations

The project has several limitations:

1. **Dataset dependency:** Results depend on the quality, size, and representativeness of the dataset.
2. **Synthetic data:** If the dataset is generated artificially, model performance may not generalize to real-world air quality data.
3. **Parameter sensitivity:** DBSCAN results depend on `eps`, `min_samples`, and feature scaling.
4. **Threshold sensitivity:** Isolation Forest results depend on its configuration and anomaly decision threshold.
5. **Reference labels:** Accuracy, precision, recall, and F1-score require reliable ground-truth labels.
6. **No automatic causal explanation:** Detecting an unusual observation does not identify its source.
7. **No guaranteed health assessment:** An anomaly prediction does not establish whether an observation exceeds a regulatory standard or poses a specific health risk.
8. **Monitoring limitations:** A real deployment would require reliable sensor measurements, suitable calibration, and appropriate validation.

---

## 17. Future Enhancements

Possible future improvements include:

* Integrating real-time air quality sensor data.
* Using verified data from environmental monitoring stations.
* Adding timestamps and geographical information where available.
* Providing configurable visualizations and filtering.
* Comparing additional anomaly detection algorithms.
* Improving parameter selection and model evaluation.
* Adding historical trend analysis.
* Developing a dashboard with more detailed reporting.
* Evaluating models against independently verified reference labels.
* Investigating model explanations and potential causes of detected anomalies.

These are proposed enhancements and should not be considered implemented features unless they exist in the current code.

---

## 18. Conclusion

AirGuard AI demonstrates how unsupervised machine learning can be applied to environmental data to identify unusual air pollution patterns.

By analysing PM2.5, PM10, CO, NO2, SO2, O3, temperature, and humidity, the system examines environmental measurements from multiple dimensions.

DBSCAN identifies observations that differ from dense clusters, while Isolation Forest identifies observations that are relatively easy to isolate. Comparing these approaches provides insight into different methods of anomaly detection.

Data preprocessing, evaluation metrics, and visualization support the analysis and interpretation of the results.

Overall, AirGuard AI serves as an educational prototype for exploring machine learning techniques in air pollution analysis and environmental monitoring.

---

## 19. Project Summary

| Item               | Description                                                                            |
| ------------------ | -------------------------------------------------------------------------------------- |
| Project Name       | AirGuard AI                                                                            |
| Project Title      | Air Pollution Anomaly Detection Using DBSCAN and Isolation Forest                      |
| Domain             | Machine Learning and Environmental Monitoring                                          |
| Input              | Structured air pollution dataset                                                       |
| Features           | PM2.5, PM10, CO, NO2, SO2, O3, Temperature, Humidity                                   |
| Primary Algorithms | DBSCAN and Isolation Forest                                                            |
| Learning Approach  | Unsupervised anomaly detection                                                         |
| Preprocessing      | Missing-value handling and feature scaling, as implemented                             |
| Evaluation         | Accuracy, Precision, Recall, and F1-score when reliable reference labels are available |
| Outputs            | Anomaly predictions, evaluation results, and visualizations                            |
| Main Objective     | Identify unusual patterns in air pollution measurements                                |

