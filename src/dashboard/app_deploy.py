import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os


st.set_page_config(
    page_title="FRAUD DETECTION SYSTEM",
    page_icon="🛡️",
    layout="wide"
)


#LOADING THE MODEL DIRECTLY-------------------------
@st.cache_resource
def load_explainer():
    base = os.path.dirname(__file__)

    explainer = joblib.load(
        os.path.join(
            base,
            "model",
            "shap_explainer.pkl"
        )
    )

    return explainer
@st.cache_resource
def load_model():
    base = os.path.dirname(__file__)

    model = joblib.load(
        os.path.join(base, "model", "final_model.pkl")
    )

    scaler = joblib.load(
        os.path.join(base, "model", "scaler.pkl")
    )

    return model, scaler

model, scaler = load_model()

####----------------CSS PART-------------------------------------------------------


st.markdown("""
<style>
            .fraud-box{
                    background:#ffe5e5;border-left:5px solid #e74c3c;
                    padding:20px;border-radius:8px;margin:10px 0;
            }
            .legit-box{
                    background:#e5ffe5;border-left:5px solid #2ecc71;
                    padding:20px;border-radius:8px;margin:10px 0; 
            }
</style>
            """,unsafe_allow_html=True)
#------------PREDICT FUTURE-----------------------------------------------------------------------------
def predict_direct(features: dict):

    df = pd.DataFrame([features])

    df["Amount_log"] = np.log1p(df["Amount"])
    df["Hour"] = (df["Time"] // 3600) % 24

    df["Is_night"] = (
        (df["Hour"] >= 22) |
        (df["Hour"] <= 5)
    ).astype(float)

    df = df.drop(columns=["Time", "Amount"])

    scale_cols = ["Amount_log", "Hour", "Is_night"]

    df[scale_cols] = scaler.transform(df[scale_cols])

    fraud_prob = float(
        model.predict_proba(df)[0][1]
    )
    
    explainer = load_explainer()
    try:
        shap_vals = explainer(df)
        if hasattr(shap_vals, "values"):
            shap_vals = shap_vals.values
    except Exception:
        shap_vals = np.zeros((1, len(df.columns)))

    try:
        explanation = pd.DataFrame({
            "feature": df.columns,
            "value": df.values[0],
            "impact": shap_vals[0],
            "direction": [
            "toward fraud" if s > 0 else "AWAY FROM FRAUD"
            for s in shap_vals[0]
            ]
            }).reindex(
                pd.Series(shap_vals[0])
                .abs()
                .sort_values(ascending=False)
                .index
                )

    except Exception:
        explanation = pd.DataFrame({
            "feature": df.columns[:5],
            "value": df.values[0][:5],
            "impact": [0]*5,
            "direction": ["N/A"]*5
            })
    
    return {
        "fraud_probability": round(fraud_prob, 4),
        "verdict": "FRAUD" if fraud_prob >= 0.5 else "LEGIT",
        "risk_score": int(fraud_prob * 100),
        "top_reasons": explanation.head(5).to_dict(orient="records")
    }
#--------------UI------------------------------------------------------------
st.title("🛡️ Real-Time Fraud Detection")
with st.sidebar:
    st.markdown('Load Example')
    example=st.selectbox("",["Customer","Known fraud","Known legit"])
dv14=-5.92 if example=="Known fraud" else (-0.31 if example=="Known legit" else 0.0)
damt=9.99 if example=="Known fraud" else (149.62 if example=="Known legit" else 50.0)
dtime=87120 if example=="Known fraud" else (406.0 if example=="Known legit" else 3600.0)

col1,col2,col3=st.columns(3)
v1  = col1.number_input("V1",  value=0.0, step=0.01, format="%.4f")
v2  = col1.number_input("V2",  value=0.0, step=0.01, format="%.4f")
v3  = col1.number_input("V3",  value=0.0, step=0.01, format="%.4f")
v4  = col1.number_input("V4",  value=0.0, step=0.01, format="%.4f")
v14 = col2.number_input("V14 (key feature)", value=dv14, step=0.01, format="%.4f")
v17 = col2.number_input("V17", value=0.0, step=0.01, format="%.4f")
v12 = col2.number_input("V12", value=0.0, step=0.01, format="%.4f")
amt = col3.number_input("Amount ($)", value=damt, min_value=0.0, step=1.0)
tme = col3.number_input("Time (sec)", value=dtime, min_value=0.0, step=100.0)

all_v={f"V{i}":0.0 for i in range(1,29)}
all_v.update({"V1":v1,"V2": v2, "V3": v3, "V4": v4,
              "V14": v14, "V17": v17, "V12": v12,
              "Amount": amt, "Time": tme})
if st.button("Analyze Transaction",type="primary",use_container_width=True):
    with st.spinner("Analyzing...."):
        result=predict_direct(all_v)
    
    fraud_pct=result["fraud_probability"]*100

    if result["verdict"]=="FRAUD":
        st.markdown(f'<div class="fraud-box"><h2>FRAUD DETECTED-Risk: {result["risk_score"]}/100</h2></div>',
                    unsafe_allow_html=True)
                    
    else:
        st.markdown(f'<div class="legit-box"><h2>APPROVED - Risk:{result["risk_score"]}/100</h2></div>',
                    unsafe_allow_html=True)
        
    g_col,s_col=st.columns([1,2])
    with g_col:
        color="#e74c3c" if fraud_pct > 50 else "#2ecc71"
        fig=go.Figure(go.Indicator(
            mode="gauge+number",value=fraud_pct,
            title={"text":"Fraud Risk"},
            number={"suffix":"%"},
            gauge={"axis":{"range":[0,100]},
            "bar":{"color":color},
            "steps":[
                       {"range": [0, 30],  "color": "#d5f5e3"},
                       {"range": [30, 70], "color": "#fef9e7"},
                       {"range": [70, 100],"color": "#fadbd8"}]}
            ))
        fig.update_layout(height=220,margin=dict(t=40,b=0,l=0,r=20))
        st.plotly_chart(fig,use_container_width=True)

    with s_col:
        reasons=result["top_reasons"]
        fig2=go.Figure(go.Bar(
            x=[r["impact"] for r in reasons],
            y=[r["feature"] for r in reasons],
            orientation="h",
            marker_color=["#e74c3c" if r["direction"]=="toward fraud" else "#2ecc71" for r in reasons]
        ))
        fig2.add_vline(x=0,line_color="gray")
        fig2.update_layout(title="Why?",height=300,
                           margin=dict(t=40,b=20,l=100,r=40))
        st.plotly_chart(fig2,use_container_width=True)
    st.dataframe(pd.DataFrame(reasons),use_container_width=True)