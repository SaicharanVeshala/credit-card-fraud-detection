import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import sys



sys.path.append(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide"
)

# ---------------------------------------------------
# CONSTANTS
# ---------------------------------------------------
import os

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:7860"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------
st.markdown("""
<style>

/* MAIN APP */
.stApp {
    background-color: #0E1117;
    color: #FAFAFA;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background-color: #1A1D29;
    border-right: 1px solid #2D3748;
}

/* HEADINGS */
.main-title {
    font-size: 52px;
    font-weight: 800;
    color: white;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 18px;
    color: #A0AEC0;
    margin-bottom: 25px;
}

/* SECTION TITLES */
.section-title {
    font-size: 28px;
    font-weight: 700;
    margin-top: 35px;
    margin-bottom: 20px;
    color: #FFFFFF;
}

/* ALERT BOXES */
.fraud-box {
    background: rgba(255, 77, 77, 0.15);
    border-left: 6px solid #FF4D4D;
    padding: 25px;
    border-radius: 14px;
    margin-top: 15px;
    margin-bottom: 15px;
    color: white;
}

.legit-box {
    background: rgba(0, 200, 83, 0.15);
    border-left: 6px solid #00C853;
    padding: 25px;
    border-radius: 14px;
    margin-top: 15px;
    margin-bottom: 15px;
    color: white;
}

/* BUTTON */
.stButton > button {
    opacity: 1 !important;
    color: white !important;
    background-color: #FF4D4D;
    color: white;
    border-radius: 12px;
    height: 55px;
    font-size: 18px;
    font-weight: 700;
    border: none;
}

.stButton > button:hover {
    background-color: #FF3333;
    color: white;
}

/* INPUTS */
.stNumberInput input {
    background-color: #1A202C !important;
    color: white !important;
    border-radius: 10px !important;
    border: 1px solid #2D3748 !important;
}

/* SELECT BOX */
.stSelectbox div[data-baseweb="select"] {
    background-color: #1A202C !important;
    border-radius: 10px !important;
}

/* METRICS */
[data-testid="metric-container"] {
    background-color: #111827;
    border: 1px solid #2D3748;
    padding: 15px;
    border-radius: 12px;
}

/* DATAFRAME */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------
def call_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except:
        return False

def call_predict(payload):
    try:
        r = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=15
        )

       

        if r.status_code != 200:
            return {
                "error": f"API Error {r.status_code}"
            }

        return r.json()

    except Exception as e:
        return {
            "error": str(e)
        }

def make_gauge(value, title):

    color = "#FF4D4D" if value > 50 else "#00C853"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title},
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 30], "color": "#163D2B"},
                {"range": [30, 70], "color": "#5C4B1F"},
                {"range": [70, 100], "color": "#4A1F1F"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 4},
                "value": 50
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor="#0E1117",
        font={"color": "white"},
        height=350
    )

    return fig


def make_shap_chart(top_reasons):

    features = [r["feature"] for r in top_reasons]
    impacts = [r["impact"] for r in top_reasons]
    directions = [r["direction"] for r in top_reasons]

    colors = [
        "#FF4D4D" if d == "increase"
        else "#00C853"
        for d in directions
    ]

    fig = go.Figure(go.Bar(
        x=impacts,
        y=features,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.4f}" for v in impacts],
        textposition="outside"
    ))

    fig.add_vline(
        x=0,
        line_color="white",
        line_width=1
    )

    fig.update_layout(
        title="SHAP Feature Contribution",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font={"color": "white"},
        xaxis_title="Impact on Fraud Prediction",
        height=400,
        margin=dict(t=40, b=40, l=120, r=80),
        showlegend=False
    )

    return fig


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
with st.sidebar:

    st.title("FraudShield AI")

    api_ok = call_api()

    if api_ok:
        st.success("API ONLINE")
    else:
        st.error("API OFFLINE")

    st.markdown("### About System")

    st.markdown("""
This system detects fraudulent credit card transactions in real time using:

- **LightGBM**
- **SHAP Explainability**
- **FastAPI Backend**
- **Streamlit Dashboard**
- **284,807 Training Transactions**
""")

    st.markdown("### Model Performance")

    st.metric("AUC-ROC", "0.9783")
    st.metric("F1 Score", "0.7137")
    st.metric("Avg Precision", "0.8733")

    st.markdown("### Load Example")

    example = st.selectbox(
        "Choose Transaction Type",
        [
            "Custom Input",
            "Known Fraud",
            "Known Legit"
        ]
    )

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------
st.markdown(
    '<div class="main-title">FraudShield AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Real-time credit card fraud detection with explainable AI using LightGBM, SHAP, FastAPI, and Streamlit.</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------
# INPUTS
# ---------------------------------------------------
st.markdown(
    '<p class="section-title">Transaction Details</p>',
    unsafe_allow_html=True
)

if example == "Known Fraud":
    default_v14 = -5.92
    default_v1 = -3.04
    default_v2 = -3.15
    default_v17 = -3.43
    default_amount = 9.99
    default_time = 87120.0

elif example == "Known Legit":
    default_v14 = -0.31
    default_v1 = -1.35
    default_v2 = -0.07
    default_v17 = 0.20
    default_amount = 149.62
    default_time = 406.0

else:
    default_v14 = 0.0
    default_v1 = 0.0
    default_v2 = 0.0
    default_v17 = 0.0
    default_amount = 50.0
    default_time = 3600.0

col1, col2, col3 = st.columns(3)

# ---------------------------------------------------
# COLUMN 1
# ---------------------------------------------------
with col1:

    st.markdown("### Key PCA Features")

    v1 = st.number_input("V1", value=default_v1, step=0.01, format="%.4f")
    v2 = st.number_input("V2", value=default_v2, step=0.01, format="%.4f")
    v3 = st.number_input("V3", value=0.0, step=0.01, format="%.4f")
    v4 = st.number_input("V4", value=0.0, step=0.01, format="%.4f")
    v5 = st.number_input("V5", value=0.0, step=0.01, format="%.4f")
    v6 = st.number_input("V6", value=0.0, step=0.01, format="%.4f")
    v7 = st.number_input("V7", value=0.0, step=0.01, format="%.4f")

# ---------------------------------------------------
# COLUMN 2
# ---------------------------------------------------
with col2:

    st.markdown("### Additional PCA Features")

    v8 = st.number_input("V8", value=0.0, step=0.01, format="%.4f")
    v9 = st.number_input("V9", value=0.0, step=0.01, format="%.4f")
    v10 = st.number_input("V10", value=0.0, step=0.01, format="%.4f")
    v11 = st.number_input("V11", value=0.0, step=0.01, format="%.4f")
    v12 = st.number_input("V12", value=0.0, step=0.01, format="%.4f")
    v13 = st.number_input("V13", value=0.0, step=0.01, format="%.4f")

    v14 = st.number_input(
        "V14 (Most Important)",
        value=default_v14,
        step=0.01,
        format="%.4f"
    )

# ---------------------------------------------------
# COLUMN 3
# ---------------------------------------------------
with col3:

    st.markdown("### Transaction Information")

    amount = st.number_input(
        "Amount ($)",
        value=default_amount,
        min_value=0.0,
        step=1.0,
        format="%.2f"
    )

    time_val = st.number_input(
        "Time (seconds)",
        value=default_time,
        min_value=0.0,
        step=100.0
    )

    v15 = st.number_input("V15", value=0.0, step=0.01, format="%.4f")
    v16 = st.number_input("V16", value=0.0, step=0.01, format="%.4f")
    v17 = st.number_input("V17", value=default_v17, step=0.01, format="%.4f")
    v18 = st.number_input("V18", value=0.0, step=0.01, format="%.4f")

# ---------------------------------------------------
# EXTRA FEATURES
# ---------------------------------------------------
with st.expander("V19 - V28 Features"):

    c1, c2, c3, c4, c5 = st.columns(5)

    v19 = c1.number_input("V19", value=0.0)
    v20 = c2.number_input("V20", value=0.0)
    v21 = c3.number_input("V21", value=0.0)
    v22 = c4.number_input("V22", value=0.0)
    v23 = c5.number_input("V23", value=0.0)

    v24 = c1.number_input("V24", value=0.0)
    v25 = c2.number_input("V25", value=0.0)
    v26 = c3.number_input("V26", value=0.0)
    v27 = c4.number_input("V27", value=0.0)
    v28 = c5.number_input("V28", value=0.0)

# ---------------------------------------------------
# BUTTON
# ---------------------------------------------------
predict_btn = st.button(
    "Analyze Transaction",
    use_container_width=True
)

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------
if predict_btn:

    if not api_ok:
        st.error("API is offline.")
    else:

        payload = {
            "V1": v1,
            "V2": v2,
            "V3": v3,
            "V4": v4,
            "V5": v5,
            "V6": v6,
            "V7": v7,
            "V8": v8,
            "V9": v9,
            "V10": v10,
            "V11": v11,
            "V12": v12,
            "V13": v13,
            "V14": v14,
            "V15": v15,
            "V16": v16,
            "V17": v17,
            "V18": v18,
            "V19": v19,
            "V20": v20,
            "V21": v21,
            "V22": v22,
            "V23": v23,
            "V24": v24,
            "V25": v25,
            "V26": v26,
            "V27": v27,
            "V28": v28,
            "Amount": amount,
            "Time": time_val
        }

        with st.spinner("Analyzing transaction..."):
            result = call_predict(payload)

        if "error" in result:
            st.error(result["error"])

        else:

            fraud_pct = result["fraud_probability"] * 100

            st.markdown(
                '<p class="section-title">Analysis Result</p>',
                unsafe_allow_html=True
            )

            if result["verdict"] == "FRAUD":

                st.markdown(f"""
                <div class="fraud-box">
                    <h2>Fraud Detected</h2>
                    <p>
                    Risk Score:
                    <strong>{result['risk_score']}/100</strong>
                    &nbsp; | &nbsp;
                    Confidence:
                    <strong>{result['confidence']}</strong>
                    &nbsp; | &nbsp;
                    Response Time:
                    <strong>{result['response_time_ms']} ms</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown(f"""
                <div class="legit-box">
                    <h2>Transaction Approved</h2>
                    <p>
                    Risk Score:
                    <strong>{result['risk_score']}/100</strong>
                    &nbsp; | &nbsp;
                    Confidence:
                    <strong>{result['confidence']}</strong>
                    &nbsp; | &nbsp;
                    Response Time:
                    <strong>{result['response_time_ms']} ms</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)

            m1.metric("Fraud Probability", f"{fraud_pct:.2f}%")
            m2.metric("Risk Score", f"{result['risk_score']}/100")
            m3.metric("Verdict", result["verdict"])
            m4.metric("Confidence", result["confidence"])

            g_col, s_col = st.columns([1, 2])

            with g_col:
                st.plotly_chart(
                    make_gauge(fraud_pct, "Fraud Risk"),
                    use_container_width=True
                )

            with s_col:
                if result.get("top_reasons"):
                    st.plotly_chart(
                        make_shap_chart(result["top_reasons"]),
                        use_container_width=True
                    )

            st.markdown(
                '<p class="section-title">Why did the model decide this?</p>',
                unsafe_allow_html=True
            )

            reasons_df = pd.DataFrame(result["top_reasons"])

            reasons_df["impact"] = reasons_df["impact"].round(4)
            reasons_df["value"] = reasons_df["value"].round(4)

            reasons_df.index = range(
                1,
                len(reasons_df) + 1
            )

            st.dataframe(
                reasons_df,
                use_container_width=True,
                column_config={
                    "feature": "Feature",
                    "value": "Feature Value",
                    "impact": "SHAP Impact",
                    "direction": "Direction"
                }
            )

            with st.expander("Raw API Response (JSON)"):
                st.json(result)

# ---------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------
st.markdown(
    '<p class="section-title">Model Comparison</p>',
    unsafe_allow_html=True
)

perf_data = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "XGBoost",
        "LightGBM"
    ],
    "AUC-ROC": [
        0.952,
        0.971,
        0.9783
    ],
    "F1 Score": [
        0.65,
        0.69,
        0.7137
    ]
})

fig_perf = px.bar(
    perf_data.melt(
        id_vars="Model",
        var_name="Metric",
        value_name="Score"
    ),
    x="Model",
    y="Score",
    color="Metric",
    barmode="group",
    title="Performance Comparison",
    color_discrete_map={
        "AUC-ROC": "#3B82F6",
        "F1 Score": "#FF4D4D"
    }
)

fig_perf.update_layout(
    paper_bgcolor="#0E1117",
    plot_bgcolor="#0E1117",
    font={"color": "white"},
    yaxis_range=[0, 1],
    height=400
)

st.plotly_chart(
    fig_perf,
    use_container_width=True
)
