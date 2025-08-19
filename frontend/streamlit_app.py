# diabetes-prediction/frontend/streamlit_app.py
import os
import json
import time
import requests
import streamlit as st

DEFAULT_API = os.getenv("API_URL", "https://diabetes-prediction-fastapi-mlops.onrender.com")

st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide",
)

# Lite CSS polish
st.markdown(
    """
    <style>
      .app-title {font-size: 2.2rem; font-weight: 800; margin: 0 0 .4rem 0;}
      .muted {color: #94a3b8; font-size: .95rem}
      .card {background: #0f172a; border: 1px solid #1f2a44; padding: 1rem 1.1rem; border-radius: 16px;}
      .badge {display:inline-block; padding:.35rem .75rem; border-radius:999px; font-weight:700; letter-spacing:.25px;}
      .badge.ok {background: #052e1a; color:#34d399; border:1px solid #065f46;}
      .badge.bad {background: #2b0a0a; color:#fda4af; border:1px solid #7f1d1d;}
      .progress-wrap {margin-top:.35rem; background:#0b1223; border-radius: 999px; border:1px solid #1e293b; height: 14px; overflow:hidden}
      .progress {height:100%; width:0%; background: linear-gradient(90deg,#22c55e,#06b6d4); transition: width 600ms ease;}
      .pill {background:#0b1223; border:1px solid #1e293b; border-radius:999px; padding:.25rem .6rem; color:#cbd5e1; font-size:.85rem; display:inline-block; margin-right:.35rem;}
      .sep {height: 1px; background: #1f2a44; margin: 14px 0;}
      .mono {font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
left, right = st.columns([0.7, 0.3], vertical_alignment="center")
with left:
    st.markdown('<div class="app-title">🩺 Diabetes Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Enter patient data → Predict → View confidence and model test metrics.</div>', unsafe_allow_html=True)

with right:
    api_url = st.text_input("API URL", value=DEFAULT_API, help="Your FastAPI base URL (no trailing slash)")

st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

# Form
st.markdown("#### Patient Inputs")
c1, c2, c3, c4 = st.columns(4)
with c1:
    Pregnancies = st.number_input("Pregnancies", min_value=0, step=1, value=1)
    SkinThickness = st.number_input("Skin Thickness (mm)", min_value=0.0, step=1.0, value=20.0)
with c2:
    Glucose = st.number_input("Glucose (mg/dL)", min_value=0.0, step=1.0, value=120.0)
    Insulin = st.number_input("Insulin (IU/mL)", min_value=0.0, step=1.0, value=80.0)
with c3:
    BloodPressure = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, step=1.0, value=70.0)
    BMI = st.number_input("BMI", min_value=0.0, step=0.1, value=30.0, format="%.1f")
with c4:
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, step=0.01, value=0.30, format="%.2f")
    Age = st.number_input("Age (years)", min_value=0, step=1, value=30)

st.write("")
go_col, _ = st.columns([0.2, 0.8])
predict_clicked = go_col.button("🔮 Predict", use_container_width=True)

# Helpers
def call_api(payload: dict) -> dict:
    url = f"{api_url.rstrip('/')}/predict"
    r = requests.post(url, json=payload, timeout=20)
    r.raise_for_status()
    return r.json()

def get_metrics() -> dict:
    try:
        url = f"{api_url.rstrip('/')}/metrics"
        r = requests.get(url, timeout=10)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return {}

# Prediction
if "history" not in st.session_state:
    st.session_state.history = []

if predict_clicked:
    payload = {
        "Pregnancies": int(Pregnancies),
        "Glucose": float(Glucose),
        "BloodPressure": float(BloodPressure),
        "SkinThickness": float(SkinThickness),
        "Insulin": float(Insulin),
        "BMI": float(BMI),
        "DiabetesPedigreeFunction": float(DiabetesPedigreeFunction),
        "Age": int(Age),
    }
    with st.spinner("Scoring…"):
        try:
            t0 = time.time()
            res = call_api(payload)
            latency_ms = int((time.time() - t0) * 1000)
            st.session_state.history.insert(0, {"input": payload, "output": res, "latency_ms": latency_ms})
            st.session_state.history = st.session_state.history[:5]
        except Exception as e:
            st.error(f"API error: {e}")
            res = None

    if res:
        pred = int(res.get("prediction", 0))
        label = "Diabetic" if pred == 1 else "Not Diabetic"
        conf = float(res.get("confidence", 0))
        conf_pct = int(round(conf * 100))

        colA, colB = st.columns([0.58, 0.42])
        with colA:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(
                f'<span class="badge {"bad" if pred==1 else "ok"}">{label}</span> '
                f'<span class="pill mono">confidence: {conf_pct}%</span> '
                f'<span class="pill mono">latency: {latency_ms} ms</span>',
                unsafe_allow_html=True
            )
            st.markdown('<div class="progress-wrap"><div class="progress" id="pb"></div></div>', unsafe_allow_html=True)
            st.markdown(
                f"""
                <script>
                const el = window.parent.document.getElementById('pb');
                if (el) el.style.width = '{conf_pct}%';
                </script>
                """,
                unsafe_allow_html=True,
            )
            st.write("")
            st.json(res)
            st.markdown('</div>', unsafe_allow_html=True)

        with colB:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.caption("Model test metrics (from training)")
            m = get_metrics()
            if m:
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Accuracy", f"{m.get('accuracy', 0):.2f}")
                    st.metric("Recall", f"{m.get('recall', 0):.2f}")
                with c2:
                    st.metric("Precision", f"{m.get('precision', 0):.2f}")
                    st.metric("F1 (weighted)", f"{m.get('f1', 0):.2f}")
            else:
                st.info("Metrics not available (check `/metrics`).")
            st.markdown('</div>', unsafe_allow_html=True)

# Tiny history
st.write("")
st.markdown("#### Recent Predictions")
if st.session_state.history:
    for item in st.session_state.history:
        pred = int(item["output"].get("prediction", 0))
        label = "Diabetic" if pred == 1 else "Not Diabetic"
        conf = float(item["output"].get("confidence", 0))
        conf_pct = int(round(conf * 100))
        with st.expander(f'{label} • {conf_pct}% • {item["latency_ms"]} ms'):
            c1, c2 = st.columns([0.55, 0.45])
            with c1:
                st.code(json.dumps(item["input"], indent=2), language="json")
            with c2:
                st.code(json.dumps(item["output"], indent=2), language="json")
else:
    st.caption("No predictions yet. Fill the form above and click **Predict**.")
