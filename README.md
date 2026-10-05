# Air Pollution Anomaly Detection Using DBSCAN and Isolation Forest

## Project Overview

This project detects unusual air pollution observations using unsupervised machine learning techniques.

The project uses two anomaly detection algorithms:

1. DBSCAN
2. Isolation Forest

The system works with air-quality and environmental parameters such as PM2.5, PM10, CO, NO2, SO2, O3, temperature, and humidity.

---

## Machine Learning Workflow

``text
Air Pollution Dataset
        |
        v
Data Loading
        |
        v
Data Preprocessing
        |
        +-- Missing Value Handling
        |
        +-- Feature Selection
        |
        +-- Standardization
        |
        v
+-----------------------+
|                       |
v                       v
DBSCAN          Isolation Forest
|                       |
v                       v
Anomaly Detection       Anomaly Detection
|                       |
+-----------+-----------+
            |
            v
       M5 Evaluation
            |
            v
   Model Comparison
            |
            v
     Visualization
``
