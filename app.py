import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from metrics_utils import compute_metrics

TARGET = "Target"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="Student Outcome Predictor", page_icon="🎓", layout="wide")
st.title("🎓 Student Dropout & Academic Success – Model Evaluator")
st.caption("Upload the test CSV, pick a trained model, and inspect its performance. "
           "Models are pre-trained and loaded from disk (no retraining).")


@st.cache_resource
def load_artifacts():
    models = {n: joblib.load(os.path.join("model", f)) for n, f in MODEL_FILES.items()}
    le = joblib.load("model/label_encoder.joblib")
    cols = joblib.load("model/feature_columns.joblib")
    return models, le, cols


models, le, feature_cols = load_artifacts()

# ---------------- Sidebar ----------------
st.sidebar.header("Controls")
uploaded = st.sidebar.file_uploader("Upload test CSV (features + 'Target')", type=["csv"])
model_name = st.sidebar.selectbox("Select model", list(MODEL_FILES.keys()))
show_all = st.sidebar.checkbox("Also compare all 5 models", value=True)

if uploaded is None:
    st.info("👈 Upload `test_data.csv` from the repository to begin.")
    if os.path.exists("model/results_table.csv"):
        st.subheader("Reference results (from training run)")
        st.dataframe(pd.read_csv("model/results_table.csv"), use_container_width=True)
    st.stop()

df = pd.read_csv(uploaded)
missing = [c for c in feature_cols + [TARGET] if c not in df.columns]
if missing:
    st.error(f"Uploaded file is missing required columns: {missing}")
    st.stop()

st.subheader("Preview of uploaded data")
st.write(f"Rows: **{len(df)}**, Columns: **{df.shape[1]}**")
st.dataframe(df.head(), use_container_width=True)

unknown = set(df[TARGET].unique()) - set(le.classes_)
if unknown:
    st.error(f"Unknown labels in '{TARGET}': {unknown}. Expected: {list(le.classes_)}")
    st.stop()

X = df[feature_cols]
y_true = le.transform(df[TARGET])


def evaluate(name):
    pipe = models[name]
    pred = pipe.predict(X)
    proba = pipe.predict_proba(X)
    return pred, compute_metrics(y_true, pred, proba)


pred, metrics = evaluate(model_name)

st.subheader(f"Results – {model_name}")
cols = st.columns(6)
for c, (k, v) in zip(cols, metrics.items()):
    c.metric(k, f"{v:.4f}")

left, right = st.columns(2)
with left:
    st.markdown("**Confusion matrix**")
    cm = confusion_matrix(y_true, pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
with right:
    st.markdown("**Classification report**")
    rep = classification_report(y_true, pred, target_names=le.classes_, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(rep).T.round(3), use_container_width=True)

if show_all:
    st.subheader("Comparison of all models on uploaded data")
    rows = [{"ML Model Name": n, **evaluate(n)[1]} for n in MODEL_FILES]
    comp = pd.DataFrame(rows).set_index("ML Model Name").round(4)
    st.dataframe(comp.style.highlight_max(axis=0, color="#c6f6d5"), use_container_width=True)
    st.bar_chart(comp[["Accuracy", "F1", "MCC"]])
