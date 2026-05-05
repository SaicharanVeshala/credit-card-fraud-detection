import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os
import sys


sys.path.append(
    os.path.join(os.path.dirname(__file__), "..", "..")
)


#----------page config------------------------------------------
st.set_page_config(
    page_title="FRAUD DETECTION SYSTEM",
    page_icon="🛡️",
    layout="wide"

)

#-----------Condtanats----------------------------------------------
API_URL="http://127.0.0.1:8000"
DATA_PATH=os.path.join(os.path.dirname(__file__),
                       "..","..","data","processed")

#-----------Custom CSS-----------------------------------------------

st.markdown(
    """
<style>
    .fraud-box{
    background-color: #ffe5e5;
    border-left: 5px solid #e74c3c;
    padding: 20px;
    border-radius: 8px;
    margin: 10px 0;
    }
    .legit-box{
    background-color: #e5ffe5;
        border-left: 5px solid #2ecc71;
        padding: 20px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .metric-card{
    background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    .section-title {
        font-size: 20px;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 10px;
    }
</style>
""",unsafe_allow_html=True
)
#--------------------Helper functions-----------------------------------------
def call_api():
    try:
        r=requests.get(f"{API_URL}/health",timeout=3)
        return r.status_code==200
    except:
        return False

def call_predict(payload):
    try:
        r=requests.post(f"{API_URL}/predict",
                        json=payload,timeout=10)
        return r.json()
    except Exception as e:
        return{"error":str(e)}
    
def make_gauge(value,title):
    color="#e74c3c" if value > 50 else "#2ecc71"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text":title,"font":{"size":16}},
        number={"suffix":"%","font":{"size":28}},
        gauge={
            "axis":{"range":[0,100]},
            "bar" :{"color":color},
            "steps":[
                {"range":[0,30],"color":"#d5f5e3"},
                {"range": [30, 70], "color": "#fef9e7"},
                {"range": [70, 100],"color": "#fadbd8"},
            ],
            "threshold" : {
                "line":{"color":"black","width":3},
                "value":50
            } 
        }
    ))
    fig.update_layout(height=220,margin=dict(t=40,b=0,l=20,r=20))
    return fig

def make_shap_chart(top_reasons):
    features=[r["feature"] for r in top_reasons]
    impacts=[r["impact"] for r in top_reasons]
    directions=[r["direction"] for r in top_reasons]
    colors=["#e74c3c" if d=="increase"
            else "#2ecc71" for d in directions]
    
    fig=go.Figure(go.Bar(
        x =impacts,
        y =features,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.4f}" for v in impacts],
        textposition="outside"
    ))
    fig.add_vline(x=0,line_color="gray",line_width=1)
    fig.update_layout(
        title="SHAP Feature Contribution",
        xaxis_title="Impact on Fraud Prediction",
        height=350,
        margin=dict(t=40,b=40,l=120,r=80),
        showlegend=False
    )
    return fig


#--------------SIDE BAR-----------------------------------------------------------------
with st.sidebar:
    st.title("🛡️ Fraud Detector")
    st.markdown("---")

    api_ok=call_api()
    if api_ok:
        st.success("API ONLINE")
    else:
        st.error("API OFFLINE - RUN UVICORN FIRST")
        st.code("cd src/api\nuvicorn main:app --reload")

    st.markdown("-----")
    st.markdown("###ABOUT####")
    st.markdown("""
    This system detects credit card fraud in real time using:
    - **LightGBM** model
    - **SHAP** explainability
    - **FastAPI** backend
    - Trained on 284,807 transactions
    """)

    st.markdown("------")
    st.markdown('####MODEL PERFORMANCE')
    st.metric("AUC ROC", "0.9783")
    st.metric("F1 SCORE","0.7137")
    st.metric("TEST SIZE","56,962 Transactions")


    st.markdown("--------")
    st.markdown("#### LOAD A REAL EXAMPLE")
    example=st.selectbox(
        "CHOOOSE TRANSACTION TYPE",
        ["Custom input","Known Fraud","Known legit"]
    )


st.title("🛡️ Real-Time Fraud Detection System")
st.markdown("ENTER TRANSACTION DETAILS BELOW TO GET AN  INSTANT FRAUD RISK SCORE WITH AI EXPLANATION")
st.markdown("-------")
#----------TRANSACTION INPUT FORM-------------------------------------------------
st.markdown('<p class="section-title">Transaction Details</p>',unsafe_allow_html=True)


if example=="Known Fraud":
    default_v14 = -5.92
    default_v1 = -3.04
    default_v2 = -3.15
    default_v17 = -3.43
    default_amount = 9.99
    default_time = 87120.0

elif example=="Known legit":
    default_v14    = -0.31
    default_v1     = -1.35
    default_v2     = -0.07
    default_v17    = 0.20
    default_amount = 149.62
    default_time   = 406.0

else:
    default_v14    = 0.0
    default_v1     = 0.0
    default_v2     = 0.0
    default_v17    = 0.0
    default_amount = 50.0
    default_time   = 3600.0

col1,col2,col3=st.columns(3)
with col1:
    st.markdown("KEY PCA FEATURES")
    v1  =st.number_input("V1", value=default_v1,step=0.01, format="%.4f")
    v2  = st.number_input("V2",  value=default_v2,  step=0.01, format="%.4f")
    v3  = st.number_input("V3",  value=0.0,          step=0.01, format="%.4f")
    v4  = st.number_input("V4",  value=0.0,          step=0.01, format="%.4f")
    v5  = st.number_input("V5",  value=0.0,          step=0.01, format="%.4f")
    v6  = st.number_input("V6",  value=0.0,          step=0.01, format="%.4f")
    v7  = st.number_input("V7",  value=0.0,          step=0.01, format="%.4f")

with col2:
    st.markdown("**More PCA Features**")
    v8  = st.number_input("V8",  value=0.0,  step=0.01, format="%.4f")
    v9  = st.number_input("V9",  value=0.0,  step=0.01, format="%.4f")
    v10 = st.number_input("V10", value=0.0,  step=0.01, format="%.4f")
    v11 = st.number_input("V11", value=0.0,  step=0.01, format="%.4f")
    v12 = st.number_input("V12", value=0.0,  step=0.01, format="%.4f")
    v13 = st.number_input("V13", value=0.0,  step=0.01, format="%.4f")
    v14 = st.number_input("V14 (most important)",
                           value=default_v14, step=0.01, format="%.4f")
with col3:
    st.markdown("** TRANSACTION INFO***")
    amount=st.number_input("Amount ($)",
                           value=default_amount,
                           min_value=0.0,
                           step=1.0,format="%.2f")
    time_val=st.number_input("Time(seconds)",
                         value=default_time,
                         min_value=0.0,
                         step=100.0)
    v15 = st.number_input("V15", value=0.0, step=0.01, format="%.4f")
    v16 = st.number_input("V16", value=0.0, step=0.01, format="%.4f")
    v17 = st.number_input("V17", value=default_v17, step=0.01, format="%.4f")
    v18 = st.number_input("V18", value=0.0, step=0.01, format="%.4f")


with  st.expander("V19 - V28 (click to expand)"):
    c1,c2,c3,c4,c5=st.columns(5)
    v19 = c1.number_input("V19", value=0.0, step=0.01, format="%.4f")
    v20 = c2.number_input("V20", value=0.0, step=0.01, format="%.4f")
    v21 = c3.number_input("V21", value=0.0, step=0.01, format="%.4f")
    v22 = c4.number_input("V22", value=0.0, step=0.01, format="%.4f")
    v23 = c5.number_input("V23", value=0.0, step=0.01, format="%.4f")
    v24 = c1.number_input("V24", value=0.0, step=0.01, format="%.4f")
    v25 = c2.number_input("V25", value=0.0, step=0.01, format="%.4f")
    v26 = c3.number_input("V26", value=0.0, step=0.01, format="%.4f")
    v27 = c4.number_input("V27", value=0.0, step=0.01, format="%.4f")
    v28 = c5.number_input("V28", value=0.0, step=0.01, format="%.4f")

st.markdown("----------")


#-----------------------PREDICTION BUTTON--------------
# ----------------------
predict_btn=st.button("ANALYZE TRANSACTION",type="primary",use_container_width=True)

if predict_btn:
    if not api_ok:
        st.error("API IS OFFLINE START IT WITH : 'uvicorn main:app --reload in src/api folder")
    else:
        payload={
            "V1":v1,"V2":v2,"V3": v3,   "V4": v4,
            "V5": v5,   "V6": v6,   "V7": v7,   "V8": v8,
            "V9": v9,   "V10": v10, "V11": v11, "V12": v12,
            "V13": v13, "V14": v14, "V15": v15, "V16": v16,
            "V17": v17, "V18": v18, "V19": v19, "V20": v20,
            "V21": v21, "V22": v22, "V23": v23, "V24": v24,
            "V25": v25, "V26": v26, "V27": v27, "V28": v28,
            "Amount": amount,
            "Time"  : time_val
        }

        with st.spinner("ANALYZING TRANSACTION....."):
            result=call_predict(payload)
        
        if "error" in result:
            st.error(f"API ERROR: {result['error']}")

        else:
            fraud_pct=result["fraud_probability"]*100

            st.markdown("------")
            st.markdown('<p class="section-title">Analysis Result </p>',
                       unsafe_allow_html=True)
            
            if result["verdict"]=="FRAUD":
                st.markdown(f"""
                <div class ="fraud-box">
                    <h2> FRAUD DETECTED</h2>
                    <p>Risk Score:<strong>{result['risk_score']}/100</strong>
                        &nbsp;|&nbsp; Confidence: <strong>{result['confidence']}</strong>
                        &nbsp;|&nbsp; Response time: <strong>{result['response_time_ms']}ms</strong>
                        &nbsp;|&nbsp; Cached <strong> {result['cached']}</strong> 
                    </p>
                </div>
                """,unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="legit-box">
                    <h2>TRANSACTION APPROVED</h2>
                    <p>Risk Score: <strong>{result['risk_score']}/100</strong>
                       &nbsp;|&nbsp; Confidence: <strong>{result['confidence']}</strong>
                       &nbsp;|&nbsp; Response time: <strong>{result['response_time_ms']}ms</strong>
                       &nbsp;|&nbsp; Cached: <strong>{result['cached']}</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            m1,m2,m3,m4=st.columns(4)
            m1.metric("Fraud Probability", f"{fraud_pct:.2f}%")
            m2.metric("Risk Score"        , f"{result['risk_score']}/100")
            m3.metric("Verdict",            result["verdict"])
            m4.metric("Confidence",           result["confidence"])



            ###-----GAUGE + SHAP SIDE BY SIDE
            g_col,s_col=st.columns([1,2])

            with g_col:
                st.plotly_chart(
                    make_gauge(fraud_pct,"Fraud Risk"),
                    use_container_width=True
                )
            with s_col:
                if result.get("top_reasons"):
                    st.plotly_chart(
                        make_shap_chart(result["top_reasons"]),
                        use_container_width=True
                    )
            st.markdown('<p class="section-title">Why did the model decide this?</p>',
                        unsafe_allow_html=True)
            
            reasons_df=pd.DataFrame(result["top_reasons"])
            reasons_df["impact"]=reasons_df["impact"].round(4)
            reasons_df["value"]=reasons_df["value"].round(4)
            reasons_df.index=range(1,len(reasons_df)+1)

            st.dataframe(
                reasons_df,
                use_container_width=True,
                column_config={
                    "feature"     : "Feature",
                    "value"       : "Feature Value",
                    "impact"      : "SHAP Impact",
                    "direction"   : "Direction"      
                }
            )


            with st.expander("RAW API RESPONSE(JSON)"):
                st.json(result)


        #------------MODEL PERFORMANCE--------------------------------------
        st.markdown("----")
        st.markdown('<p class="section-title">Model Performance</p>',
                    unsafe_allow_html=True)
        
        p1,p2,p3,p4=st.columns(4)
        p1.metric("Model","LightGBM (Final)")
        p2.metric("AUC-ROC","0.9783")
        p3.metric("F1 SCORE","0.7137")
        p4.metric("Avg Precision","0.8733")


    perf_data = pd.DataFrame({
           "Model": [
           "Logistic Regression",
           "XGBoost",
           "LightGBM (Final)"
           ],
           "AUC-ROC": [
               0.952,
               0.971,
               0.9783
               ],
               "F1": [
                   0.65,
                   0.69,
                   0.7137
                   ]
                   })

    fig_pref=px.bar(
            perf_data.melt(id_vars="Model",var_name="Metric",value_name="Score"),
            x="Model",
            y="Score",
            color="Metric",
            barmode="group",
            title="Model Comparison",
            color_discrete_map={"AUC-ROC":"#3498db","F1":"#e74c3c"}

        )
    fig_pref.update_layout(height=300,yaxis_range=[0,1])
    st.plotly_chart(fig_pref,use_container_width=True)

    st.markdown("-------")
    #st.markdown("BUILT AS AS PORTFOLIO PROJECT . LIGHT GBM + SHAP + FastAPI + Streamlit")





