import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

FEATURE_COLUMNS = ["PM2.5", "PM10", "CO", "NO2", "SO2", "O3", "Temperature", "Humidity"]
ALIASES = {
    "PM2.5": ["pm2.5", "pm25", "pm_2_5", "pm 2.5"],
    "PM10": ["pm10", "pm_10", "pm 10"],
    "CO": ["co", "carbon monoxide"],
    "NO2": ["no2", "no₂", "nitrogen dioxide"],
    "SO2": ["so2", "so₂", "sulfur dioxide", "sulphur dioxide"],
    "O3": ["o3", "o₃", "ozone"],
    "Temperature": ["temperature", "temp", "temperature (c)", "temperature_c"],
    "Humidity": ["humidity", "relative humidity", "humidity (%)"],
    "IsAnomaly": ["isanomaly", "is_anomaly", "anomaly", "ground_truth", "label"]
}

st.set_page_config(page_title="Air Guard", page_icon="🌫️", layout="wide")

st.markdown("""
<style>
.main {background-color: #f7f9fc;}
.block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
.hero {padding: 1.35rem 1.5rem; border-radius: 16px; background: linear-gradient(120deg,#102a43,#176b87); color: white; margin-bottom: 1rem;}
.hero h1 {color:white; margin:0; font-size:2rem;}
.hero p {color:#e5f4fa; margin:.45rem 0 0 0;}
div[data-testid="stMetric"] {background:white; border:1px solid #e6edf3; padding:15px; border-radius:12px;}
</style>
<div class="hero">
<h1>🌫️ Air Guard</h1>
<p>Upload air-quality data, detect unusual readings with DBSCAN and Isolation Forest, and explore interactive charts.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    st.caption("Required columns: PM2.5, PM10, CO, NO2, SO2, O3, Temperature, Humidity. IsAnomaly is optional and enables evaluation.")
    st.divider()
    st.subheader("Model settings")
    eps = st.slider("DBSCAN eps", min_value=0.1, max_value=3.0, value=0.8, step=0.1)
    min_samples = st.slider("DBSCAN min samples", min_value=2, max_value=30, value=10)
    contamination = st.slider("Isolation Forest contamination", min_value=0.01, max_value=0.25, value=0.07, step=0.01)
    st.caption("Settings apply when you click Analyze data.")

def normalize_columns(df):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    rename = {}
    for canonical, aliases in ALIASES.items():
        for alias in aliases:
            if alias.lower() in lookup:
                rename[lookup[alias.lower()]] = canonical
                break
    return df.rename(columns=rename)

def analyze_data(raw, eps_value, min_samples_value, contamination_value):
    df = normalize_columns(raw.copy())
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing required feature columns: " + ", ".join(missing))
    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Rows with all feature values missing cannot be analyzed.
    df = df.dropna(subset=FEATURE_COLUMNS, how="all").reset_index(drop=True)
    if len(df) < 3:
        raise ValueError("Please upload at least 3 rows with numeric pollution/weather measurements.")
    imputer = SimpleImputer(strategy="median")
    scaled = imputer.fit_transform(df[FEATURE_COLUMNS])
    scaler = StandardScaler()
    X = scaler.fit_transform(scaled)
    db_labels = DBSCAN(eps=eps_value, min_samples=min_samples_value).fit_predict(X)
    db_anomaly = (db_labels == -1).astype(int)
    iso = IsolationForest(n_estimators=200, contamination=contamination_value, random_state=42, n_jobs=-1)
    iso_pred = iso.fit_predict(X)
    iso_anomaly = (iso_pred == -1).astype(int)
    scores = -iso.decision_function(X)
    out = df.copy()
    out["DBSCAN_Label"] = db_labels
    out["DBSCAN_Anomaly"] = db_anomaly
    out["IsolationForest_Anomaly"] = iso_anomaly
    out["IsolationForest_Score"] = scores
    if "IsAnomaly" in out.columns:
        labels = pd.to_numeric(out["IsAnomaly"], errors="coerce")
        # Only evaluate if all labels can be interpreted as 0/1.
        if labels.notna().all() and set(labels.unique()).issubset({0, 1}):
            out["IsAnomaly"] = labels.astype(int)
    return out

if uploaded is not None:
    try:
        raw_df = pd.read_csv(uploaded)
        source_name = uploaded.name
    except Exception as exc:
        st.error(f"Could not read this CSV: {exc}")
        st.stop()
else:
    default_path = os.path.join(os.path.dirname(__file__), "data", "air_pollution_dataset.csv")
    try:
        raw_df = pd.read_csv(default_path)
        source_name = "Built-in sample dataset"
    except Exception:
        raw_df = None
        source_name = ""

if raw_df is None:
    st.info("Upload a CSV file from the sidebar to get started.")
    st.stop()

with st.expander(f"Preview source data — {source_name}", expanded=False):
    st.dataframe(raw_df.head(20), use_container_width=True)
    st.caption(f"{len(raw_df):,} rows · {len(raw_df.columns)} columns")

if st.button("Analyze data", type="primary", use_container_width=False):
    st.session_state["analyze_now"] = True

if "analyze_now" not in st.session_state:
    st.session_state["analyze_now"] = uploaded is None

if st.session_state["analyze_now"]:
    try:
        with st.spinner("Cleaning data and running anomaly detection models..."):
            results = analyze_data(raw_df, eps, min_samples, contamination)
        st.session_state["results"] = results
        st.session_state["analyzed_source"] = source_name
        st.session_state["analyzed_settings"] = (eps, min_samples, contamination)
    except Exception as exc:
        st.error(str(exc))
        st.stop()
else:
    st.info("Click **Analyze data** to process the current CSV.")
    st.stop()

df = st.session_state["results"]
st.success(f"Analysis complete for **{st.session_state['analyzed_source']}**.")

total = len(df)
db_count = int(df["DBSCAN_Anomaly"].sum())
if_count = int(df["IsolationForest_Anomaly"].sum())
normal_count = total - if_count
m1, m2, m3, m4 = st.columns(4)
m1.metric("Records analyzed", f"{total:,}")
m2.metric("DBSCAN anomalies", f"{db_count:,}", f"{db_count/total:.1%} of records")
m3.metric("Isolation Forest anomalies", f"{if_count:,}", f"{if_count/total:.1%} of records")
m4.metric("Isolation Forest normal", f"{normal_count:,}")

tab_overview, tab_charts, tab_models, tab_data = st.tabs(["Overview", "Interactive charts", "Model evaluation", "Results table"])

with tab_overview:
    st.subheader("Analysis summary")
    c1, c2 = st.columns(2)
    with c1:
        counts = pd.DataFrame({"Model": ["DBSCAN", "Isolation Forest"], "Anomalies": [db_count, if_count], "Normal": [total-db_count, total-if_count]})
        fig = px.bar(counts, x="Model", y=["Anomalies", "Normal"], barmode="group", title="Normal readings vs detected anomalies", color_discrete_sequence=["#e76f51", "#2a9d8f"])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        feature = st.selectbox("Feature distribution", FEATURE_COLUMNS, index=0, key="overview_feature")
        fig = px.histogram(df, x=feature, nbins=35, title=f"Distribution of {feature}", marginal="box")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("**How to interpret the results**")
    st.write("- **DBSCAN anomaly:** a reading that DBSCAN identifies as noise relative to nearby readings.")
    st.write("- **Isolation Forest anomaly:** a reading that the Isolation Forest model separates as unusual.")
    st.write("- Model outputs are data-driven flags, not regulatory determinations of air safety.")

with tab_charts:
    st.subheader("Explore the dataset")
    c1, c2 = st.columns(2)
    with c1:
        x_feature = st.selectbox("X-axis feature", FEATURE_COLUMNS, index=0, key="x_feature")
    with c2:
        y_feature = st.selectbox("Y-axis feature", FEATURE_COLUMNS, index=1, key="y_feature")
    color_by = st.selectbox("Color points by", ["DBSCAN_Anomaly", "IsolationForest_Anomaly"] + (["IsAnomaly"] if "IsAnomaly" in df.columns else []))
    scatter = px.scatter(df, x=x_feature, y=y_feature, color=color_by, hover_data=[c for c in ["DBSCAN_Label", "IsolationForest_Score"] if c in df.columns], title=f"{y_feature} vs {x_feature}")
    st.plotly_chart(scatter, use_container_width=True)
    c3, c4 = st.columns(2)
    with c3:
        selected_feature = st.selectbox("Feature to compare by detection", FEATURE_COLUMNS, key="box_feature")
        box = px.box(df, x="IsolationForest_Anomaly", y=selected_feature, color="IsolationForest_Anomaly", title=f"{selected_feature}: normal vs anomaly")
        st.plotly_chart(box, use_container_width=True)
    with c4:
        score_fig = px.line(df.reset_index().rename(columns={"index":"Observation"}), x="Observation", y="IsolationForest_Score", title="Isolation Forest anomaly score by observation")
        score_fig.add_hline(y=0, line_dash="dash", annotation_text="Decision boundary")
        st.plotly_chart(score_fig, use_container_width=True)
    if "IsAnomaly" in df.columns:
        st.plotly_chart(px.scatter(df, x=df.index, y="PM2.5", color="IsAnomaly", title="Ground-truth anomalies in PM2.5"), use_container_width=True)

with tab_models:
    st.subheader("Model comparison")
    if "IsAnomaly" not in df.columns or not pd.to_numeric(df["IsAnomaly"], errors="coerce").notna().all() or not set(pd.to_numeric(df["IsAnomaly"], errors="coerce").unique()).issubset({0,1}):
        st.info("Your CSV does not contain a valid binary IsAnomaly column (0 = normal, 1 = anomaly). Detection results are available, but accuracy/precision/recall/F1 cannot be calculated without ground-truth labels.")
    else:
        y_true = df["IsAnomaly"].astype(int)
        metric_rows = []
        for name, pred_col in [("DBSCAN", "DBSCAN_Anomaly"), ("Isolation Forest", "IsolationForest_Anomaly")]:
            y_pred = df[pred_col].astype(int)
            metric_rows.append({"Model": name, "Accuracy": accuracy_score(y_true, y_pred), "Precision": precision_score(y_true, y_pred, zero_division=0), "Recall": recall_score(y_true, y_pred, zero_division=0), "F1 Score": f1_score(y_true, y_pred, zero_division=0)})
        metrics = pd.DataFrame(metric_rows)
        st.dataframe(metrics.style.format({c:"{:.3f}" for c in ["Accuracy","Precision","Recall","F1 Score"]}), use_container_width=True, hide_index=True)
        fig = px.bar(metrics.melt(id_vars="Model", var_name="Metric", value_name="Score"), x="Metric", y="Score", color="Model", barmode="group", title="Evaluation metrics", range_y=[0,1])
        st.plotly_chart(fig, use_container_width=True)
        chosen = st.selectbox("Confusion matrix model", ["DBSCAN", "Isolation Forest"])
        pred_col = "DBSCAN_Anomaly" if chosen == "DBSCAN" else "IsolationForest_Anomaly"
        cm = confusion_matrix(y_true, df[pred_col].astype(int), labels=[0,1])
        cm_fig = px.imshow(cm, text_auto=True, x=["Predicted normal", "Predicted anomaly"], y=["Actual normal", "Actual anomaly"], color_continuous_scale="Blues", title=f"{chosen} confusion matrix")
        st.plotly_chart(cm_fig, use_container_width=True)

with tab_data:
    st.subheader("Analyzed records")
    filter_mode = st.selectbox("Filter rows", ["All records", "DBSCAN anomalies", "Isolation Forest anomalies", "Both models flag anomaly", "Either model flags anomaly"])
    filtered = df
    if filter_mode == "DBSCAN anomalies":
        filtered = df[df["DBSCAN_Anomaly"] == 1]
    elif filter_mode == "Isolation Forest anomalies":
        filtered = df[df["IsolationForest_Anomaly"] == 1]
    elif filter_mode == "Both models flag anomaly":
        filtered = df[(df["DBSCAN_Anomaly"] == 1) & (df["IsolationForest_Anomaly"] == 1)]
    elif filter_mode == "Either model flags anomaly":
        filtered = df[(df["DBSCAN_Anomaly"] == 1) | (df["IsolationForest_Anomaly"] == 1)]
    st.caption(f"Showing {len(filtered):,} of {total:,} records")
    st.dataframe(filtered, use_container_width=True, height=430)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download all analyzed results (CSV)", data=csv_bytes, file_name="air_pollution_anomaly_results.csv", mime="text/csv")
    st.download_button("Download filtered results (CSV)", data=filtered.to_csv(index=False).encode("utf-8"), file_name="filtered_air_pollution_results.csv", mime="text/csv")

st.divider()
st.caption("Air Guard · DBSCAN + Isolation Forest · For exploratory analysis; not a substitute for official air-quality monitoring.")
