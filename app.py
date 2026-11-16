
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="FinGuard | Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

ROOT = Path(__file__).parent

@st.cache_resource
def load_model():
    model = joblib.load(ROOT / "rf_fraud_model.pkl")
    scaler = joblib.load(ROOT / "scaler.pkl")

    with open(ROOT / "model_metadata.json") as f:
        metadata = json.load(f)

    return model, scaler, metadata

model, scaler, metadata = load_model()

FEATURES = (
    ["Time"] +
    [f"V{i}" for i in range(1, 29)] +
    ["Amount"]
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#07111f,#10233a);
    color: white;
}
[data-testid="stSidebar"] {
    background: #0b1728;
}
h1,h2,h3 {
    color: #ffffff;
}
.metric-card {
    background: #142941;
    padding: 22px;
    border-radius: 15px;
    border: 1px solid #29425d;
}
.subtitle {
    color: #9bb5cc;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🛡️ FinGuard")
st.sidebar.caption("CREDIT CARD FRAUD INTELLIGENCE")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Fraud Predictor", "Batch Detection", "Model Analytics"]
)

threshold = st.sidebar.slider(
    "Fraud Threshold",
    0.05, 0.95, 0.30, 0.05
)

def prepare_data(data):
    missing = [c for c in FEATURES if c not in data.columns]

    if missing:
        raise ValueError(f"Missing features: {missing}")

    X = data[FEATURES].copy()

    X["Time"] = pd.to_numeric(X["Time"], errors="raise")
    X["Amount"] = pd.to_numeric(X["Amount"], errors="raise")

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Input contains invalid numeric values.")

    X[["Time", "Amount"]] = scaler.transform(
        X[["Time", "Amount"]]
    )

    return X

st.title("🛡️ FinGuard")
st.markdown(
    '<p class="subtitle">Intelligent Credit Card Fraud Detection System</p>',
    unsafe_allow_html=True
)

if page == "Dashboard":

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("ML MODEL", "Random Forest")
    c2.metric("FEATURES", "30")
    c3.metric("ROC AUC", f"{metadata['roc_auc']:.4f}")
    c4.metric("F1 SCORE", f"{metadata['f1_score']:.4f}")

    st.divider()

    st.subheader("System Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Detection Workflow")
        st.write("""
        1. Transaction feature input
        2. Data preprocessing
        3. Feature scaling
        4. Random Forest prediction
        5. Fraud probability calculation
        6. Risk classification
        """)

    with col2:
        st.markdown("### Model Performance")
        metrics = pd.DataFrame({
            "Metric": ["Precision", "Recall", "F1 Score", "ROC AUC", "PR AUC"],
            "Score": [
                metadata["precision"],
                metadata["recall"],
                metadata["f1_score"],
                metadata["roc_auc"],
                metadata["pr_auc"]
            ]
        })

        fig = px.bar(
            metrics,
            x="Metric",
            y="Score",
            title="Random Forest Performance"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#10233a",
            plot_bgcolor="#10233a"
        )

        st.plotly_chart(fig, use_container_width=True)

elif page == "Fraud Predictor":

    st.subheader("Single Transaction Analysis")

    with st.form("fraud_form"):

        col1, col2 = st.columns(2)

        with col1:
            time = st.number_input(
                "Transaction Time",
                min_value=0.0,
                value=80000.0
            )

        with col2:
            amount = st.number_input(
                "Transaction Amount",
                min_value=0.0,
                value=125.50
            )

        st.markdown("### Anonymized PCA Features")

        values = {}

        columns = st.columns(4)

        for i in range(1, 29):
            with columns[(i-1) % 4]:
                values[f"V{i}"] = st.number_input(
                    f"V{i}",
                    value=0.0,
                    format="%.5f",
                    key=f"v{i}"
                )

        submit = st.form_submit_button(
            "🔍 Analyze Transaction",
            type="primary",
            use_container_width=True
        )

    if submit:

        row = {
            "Time": time,
            **values,
            "Amount": amount
        }

        try:
            X = prepare_data(pd.DataFrame([row]))

            probability = model.predict_proba(X)[0, 1]

            st.divider()
            st.subheader("Prediction Result")

            st.metric(
                "Fraud Probability",
                f"{probability:.2%}"
            )

            st.progress(float(probability))

            if probability >= threshold:
                st.error("⚠️ HIGH RISK — Manual review required")
            else:
                st.success("LOWER RISK — No alert at current threshold")

            st.caption(
                f"Classification threshold: {threshold:.0%}. "
                "Prediction is not proof of fraud."
            )

        except Exception as e:
            st.error(str(e))

elif page == "Batch Detection":

    st.subheader("Bulk Transaction Analysis")

    uploaded = st.file_uploader(
        "Upload transaction CSV",
        type=["csv"]
    )

    if uploaded:

        try:
            data = pd.read_csv(uploaded)

            X = prepare_data(data)

            probabilities = model.predict_proba(X)[:, 1]

            results = data.copy()

            results["Fraud Probability"] = probabilities

            results["Prediction"] = np.where(
                probabilities >= threshold,
                "Fraud Alert",
                "Lower Risk"
            )

            total = len(results)
            fraud = int((probabilities >= threshold).sum())

            a, b, c = st.columns(3)

            a.metric("Total Transactions", total)
            b.metric("Flagged Transactions", fraud)
            c.metric(
                "Flag Rate",
                f"{fraud / total:.2%}" if total else "0%"
            )

            st.dataframe(
                results,
                use_container_width=True
            )

            fig = px.histogram(
                results,
                x="Fraud Probability",
                nbins=30,
                title="Fraud Probability Distribution"
            )

            fig.update_layout(template="plotly_dark")

            st.plotly_chart(fig, use_container_width=True)

            st.download_button(
                "Download Results",
                results.to_csv(index=False),
                "fraud_results.csv",
                "text/csv"
            )

        except Exception as e:
            st.error(f"Error: {e}")

elif page == "Model Analytics":

    st.subheader("Model Evaluation")

    metrics = pd.DataFrame({
        "Metric": [
            "Precision",
            "Recall",
            "F1 Score",
            "ROC AUC",
            "PR AUC"
        ],
        "Value": [
            metadata["precision"],
            metadata["recall"],
            metadata["f1_score"],
            metadata["roc_auc"],
            metadata["pr_auc"]
        ]
    })

    st.dataframe(metrics, use_container_width=True)

    st.write("Model: Random Forest Classifier")
    st.write("Estimators: 100")
    st.write("Feature count: 30")
    st.write("Threshold: Adjustable")

    st.json(metadata)

st.divider()

st.caption(
    "FinGuard | Academic Machine Learning Project | "
    "Credit Card Fraud Detection"
)
