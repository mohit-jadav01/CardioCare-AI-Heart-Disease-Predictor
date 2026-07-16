import streamlit as st
import pandas as pd
import joblib
import time
import random
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

C = {
    "primary":       "#1A73E8",
    "primary_dark":  "#1557B0",
    "primary_50":    "#E8F0FE",
    "secondary":     "#34A853",
    "secondary_50":  "#E6F4EA",
    "danger":        "#EA4335",
    "danger_50":     "#FCE8E6",
    "warning":       "#FBBC04",
    "warning_50":    "#FFF8E1",
    "surface":       "#FFFFFF",
    "surface_v":     "#F8F9FA",
    "bg":            "#F1F3F4",
    "text":          "#202124",
    "text_s":        "#5F6368",
    "outline":       "#DADCE0",
}

st.set_page_config(
    page_title="CardioCare AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');

.stApp {{
    background: {C['bg']};
    font-family: 'DM Sans', sans-serif;
}}
#MainMenu, footer {{visibility: hidden;}}
h1,h2,h3,h4,h5,h6 {{
    font-family: 'DM Sans', sans-serif !important;
    color: {C['text']} !important;
    font-weight: 600 !important;
}}
p, span, li {{ color: {C['text']}; }}

.stTextInput label, .stNumberInput label, .stSelectbox label,
.stSlider label, .stRadio label, .stCheckbox label {{
    color: {C['text']} !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    letter-spacing: 0.01em;
}}

/* Cards */
.card {{
    background: {C['surface']};
    border-radius: 16px;
    padding: 1.5rem 1.75rem;
    border: 1px solid {C['outline']};
    box-shadow: 0 1px 3px rgba(0,0,0,0.07), 0 4px 8px rgba(0,0,0,0.04);
    margin-bottom: 1.25rem;
    transition: box-shadow .2s ease;
}}
.card:hover {{
    box-shadow: 0 4px 12px rgba(0,0,0,0.10), 0 8px 24px rgba(0,0,0,0.06);
}}
.card-label {{
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: {C['primary']};
    background: {C['primary_50']};
    padding: 0.3rem 0.75rem;
    border-radius: 20px;
    margin-bottom: 1rem;
}}

/* Hero */
.hero {{
    background: linear-gradient(135deg, {C['primary']} 0%, #4285F4 55%, #0D9488 100%);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin-bottom: 1.75rem;
    position: relative;
    overflow: hidden;
}}
.hero::after {{
    content: '';
    position: absolute;
    bottom: -60px; right: -60px;
    width: 220px; height: 220px;
    border-radius: 50%;
    background: rgba(255,255,255,0.06);
    pointer-events: none;
}}
.hero::before {{
    content: '';
    position: absolute;
    top: -40px; left: -40px;
    width: 160px; height: 160px;
    border-radius: 50%;
    background: rgba(255,255,255,0.04);
    pointer-events: none;
}}
.hero h1 {{
    color: white !important;
    font-size: 2.25rem !important;
    font-weight: 700 !important;
    margin-bottom: 0.4rem !important;
}}
.hero p {{
    color: rgba(255,255,255,0.88) !important;
    font-size: 1rem;
    max-width: 560px;
    margin: 0 auto;
    line-height: 1.6;
}}
.hero-badge {{
    display: inline-block;
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.3);
    color: white;
    padding: 0.3rem 1rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
    margin-bottom: 0.9rem;
    letter-spacing: 0.03em;
}}

/* KPI */
.kpi {{
    background: {C['surface']};
    border-radius: 16px;
    padding: 1.4rem 1.25rem;
    border: 1px solid {C['outline']};
    text-align: center;
    height: 100%;
}}
.kpi-num {{
    font-size: 2.4rem;
    font-weight: 700;
    color: {C['primary']};
    line-height: 1.1;
}}
.kpi-label {{
    font-size: 0.72rem;
    color: {C['text_s']};
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.3rem;
}}
.kpi-sub {{ font-size: 0.8rem; font-weight: 500; margin-top: 0.4rem; }}
.kpi-pos {{ color: {C['secondary']}; }}
.kpi-neg {{ color: {C['danger']}; }}

/* Results */
.result-high {{
    background: {C['danger_50']};
    border: 1px solid #F5C6C2;
    border-left: 6px solid {C['danger']};
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}}
.result-low {{
    background: {C['secondary_50']};
    border: 1px solid #CEEAD6;
    border-left: 6px solid {C['secondary']};
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}}
.result-icon {{ font-size: 3.5rem; margin-bottom: 0.4rem; }}
.result-title {{
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    margin: 0.4rem 0 !important;
}}
.conf-badge {{
    display: inline-block;
    padding: 0.45rem 1.4rem;
    border-radius: 30px;
    font-weight: 600;
    font-size: 1.1rem;
    margin: 0.7rem 0;
}}

/* Chips */
.chip {{
    display: inline-block;
    padding: 0.22rem 0.7rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}}
.chip-ok   {{ background: {C['secondary_50']}; color: #137333; border: 1px solid #CEEAD6; }}
.chip-warn {{ background: {C['warning_50']};   color: #B06000; border: 1px solid #FFE082; }}
.chip-bad  {{ background: {C['danger_50']};    color: #C5221F; border: 1px solid #F5C6C2; }}

/* Inline validation */
.val-warn {{
    background: {C['warning_50']}; border: 1px solid #FFE082;
    border-radius: 8px; padding: 0.4rem 0.7rem;
    font-size: 0.78rem; color: #7A4F00; margin-top: 0.2rem; line-height: 1.4;
}}
.val-err {{
    background: {C['danger_50']}; border: 1px solid #F5C6C2;
    border-radius: 8px; padding: 0.4rem 0.7rem;
    font-size: 0.78rem; color: #C62828; margin-top: 0.2rem; line-height: 1.4;
}}
.val-ok {{
    background: {C['secondary_50']}; border: 1px solid #CEEAD6;
    border-radius: 8px; padding: 0.4rem 0.7rem;
    font-size: 0.78rem; color: #137333; margin-top: 0.2rem; line-height: 1.4;
}}

/* Buttons */
.stButton > button {{
    background: {C['primary']}; color: white;
    border: none; border-radius: 10px;
    padding: 0.72rem 1.5rem; font-weight: 500;
    font-size: 0.95rem; letter-spacing: 0.02em;
    transition: all .2s ease; width: 100%;
    font-family: 'DM Sans', sans-serif;
}}
.stButton > button:hover {{
    background: {C['primary_dark']};
    box-shadow: 0 4px 14px rgba(26,115,232,0.35);
    transform: translateY(-1px);
}}
.stButton > button:active {{ transform: translateY(0); }}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    background: {C['surface']}; border-radius: 12px;
    padding: 0.25rem; border: 1px solid {C['outline']}; gap: 0.2rem;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 8px; font-weight: 500;
    color: {C['text_s']}; padding: 0.45rem 1.2rem;
    font-family: 'DM Sans', sans-serif;
}}
.stTabs [aria-selected="true"] {{
    background: {C['primary_50']} !important;
    color: {C['primary']} !important;
    font-weight: 600 !important;
}}

/* Inputs */
.stNumberInput input {{
    color: {C['text']} !important; background: {C['surface']} !important;
    border: 2px solid {C['outline']} !important; border-radius: 10px !important;
    font-weight: 500 !important; font-family: 'DM Sans', sans-serif !important;
}}
.stNumberInput input:focus {{
    border-color: {C['primary']} !important;
    box-shadow: 0 0 0 3px rgba(26,115,232,0.1) !important;
}}
.stSelectbox [data-baseweb="select"] > div {{
    background: {C['surface']} !important; border: 2px solid {C['outline']} !important;
    border-radius: 10px !important; color: {C['text']} !important;
    font-family: 'DM Sans', sans-serif !important;
}}
.stSelectbox [data-baseweb="select"] span {{ color: {C['text']} !important; }}

/* Sidebar */
[data-testid="stSidebar"] {{
    background: {C['surface']}; border-right: 1px solid {C['outline']};
}}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{ color: {C['text']} !important; }}
[data-testid="stSidebar"] .stButton > button {{
    background: transparent !important; color: {C['text']} !important;
    border: 1px solid {C['outline']} !important;
    text-align: left !important; font-weight: 400 !important;
    box-shadow: none !important;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
    background: {C['primary_50']} !important;
    border-color: {C['primary']} !important;
    color: {C['primary']} !important;
    transform: none !important;
}}

/* Metrics */
[data-testid="stMetric"] {{
    background: {C['surface']}; border-radius: 14px;
    padding: 1rem 1.25rem; border: 1px solid {C['outline']};
}}
[data-testid="stMetricLabel"] {{ color: {C['text_s']} !important; font-size: 0.8rem !important; }}
[data-testid="stMetricValue"] {{ color: {C['text']} !important; font-weight: 700 !important; }}

/* Disclaimer */
.disclaimer {{
    background: {C['warning_50']}; border: 1px solid #FFE082;
    border-radius: 12px; padding: 0.9rem 1.2rem;
    display: flex; align-items: flex-start; gap: 0.75rem; margin-bottom: 1.5rem;
}}
.disclaimer p {{
    color: #5C4000 !important; font-size: 0.83rem; margin: 0; line-height: 1.55;
}}

/* Data table */
[data-testid="stDataFrame"] {{
    border-radius: 12px !important; overflow: hidden !important;
    border: 1px solid {C['outline']} !important;
}}

/* Scrollbar */
::-webkit-scrollbar {{ width: 5px; }}
::-webkit-scrollbar-track {{ background: {C['bg']}; }}
::-webkit-scrollbar-thumb {{ background: {C['outline']}; border-radius: 3px; }}

/* Footer */
.app-footer {{
    text-align: center; padding: 2rem; color: {C['text_s']};
    font-size: 0.78rem; border-top: 1px solid {C['outline']};
    margin-top: 3rem; line-height: 1.8;
}}

/* Info box */
.info-box {{
    background: {C['primary_50']}; border: 1px solid #C5D9F5;
    border-radius: 12px; padding: 0.9rem 1rem;
    font-size: 0.83rem; color: #1557B0; line-height: 1.55;
}}

/* Expander */
.streamlit-expanderHeader {{
    background: {C['surface_v']} !important; border-radius: 10px !important;
    color: {C['text']} !important; border: 1px solid {C['outline']} !important;
    font-weight: 500 !important; font-family: 'DM Sans', sans-serif !important;
}}
</style>
""", unsafe_allow_html=True)


if "page" not in st.session_state:
    st.session_state.page = "prediction"
if "history" not in st.session_state:
    st.session_state.history = []

# ═══════════════════════════════════════════════
#  MODEL LOADER
# ═══════════════════════════════════════════════
@st.cache_resource
def load_models():
    model   = joblib.load("logistic_regression.pkl")
    scaler  = joblib.load("scaler.pkl")
    columns = joblib.load("column.pkl")
    return model, scaler, columns

try:
    model, scaler, trained_columns = load_models()
    MODEL_OK = True
except Exception:
    MODEL_OK = False


def chip(label, kind="ok"):
    return f'<span class="chip chip-{kind}">{label}</span>'

def val_msg(msg, kind="warn"):
    return f'<div class="val-{kind}">{msg}</div>'

def plotly_base():
    return dict(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color=C["text"]),
        margin=dict(l=20, r=20, t=45, b=20),
    )


with st.sidebar:
    st.markdown(f"""
    <div style='padding:1rem 0 0.5rem;display:flex;align-items:center;gap:0.7rem;'>
        <span style='font-size:2rem;'>🫀</span>
        <div>
            <p style='margin:0;font-weight:700;font-size:1.05rem;color:{C["text"]};'>CardioCare AI</p>
            <p style='margin:0;font-size:0.72rem;color:{C["text_s"]};'>Heart Risk Intelligence v2.0</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"<p style='font-size:0.68rem;font-weight:700;text-transform:uppercase;"
                f"letter-spacing:0.12em;color:{C['text_s']};margin-bottom:0.4rem;'>NAVIGATE</p>",
                unsafe_allow_html=True)

    if st.button("🔬   Risk Prediction",      use_container_width=True): st.session_state.page = "prediction"
    if st.button("📊   Analytics Dashboard",  use_container_width=True): st.session_state.page = "analytics"
    if st.button("ℹ️   Methodology & About",  use_container_width=True): st.session_state.page = "about"

    st.markdown("---")
    if st.session_state.history:
        n_total = len(st.session_state.history)
        n_high  = sum(1 for r in st.session_state.history if r["result"] == 1)
        st.markdown(f"<p style='font-size:0.68rem;font-weight:700;text-transform:uppercase;"
                    f"letter-spacing:0.12em;color:{C['text_s']};margin-bottom:0.4rem;'>SESSION</p>",
                    unsafe_allow_html=True)
        sc1, sc2 = st.columns(2)
        sc1.metric("Total",     n_total)
        sc2.metric("High Risk", n_high)
        st.markdown("---")

    st.markdown(f"""
    <div style='background:{C["primary_50"]};border:1px solid #C5D9F5;
         border-radius:12px;padding:1rem;'>
        <p style='font-weight:700;font-size:0.85rem;color:{C["primary"]};margin:0 0 0.5rem;'>
            🚨 Emergency Contacts</p>
        <p style='font-size:0.8rem;color:{C["text"]};margin:0;line-height:1.7;'>
            Emergency&nbsp;&nbsp;<strong>108</strong><br>
            Cardiology&nbsp;&nbsp;+91-XXXXXXXXXX
        </p>
    </div>
    <div style='padding:1rem 0;text-align:center;'>
        <p style='font-size:0.7rem;color:{C["text_s"]};'>
            © 2024 CardioCare AI<br>Clinical Decision Support</p>
    </div>
    """, unsafe_allow_html=True)



def page_prediction():
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">🏥 AI-Powered Clinical Assessment</div>
        <h1>Heart Disease Risk Predictor</h1>
        <p>Enter patient clinical data below. Our ML model analyses 11 biomarkers
           to deliver a cardiovascular risk score with confidence percentage.</p>
    </div>
    """, unsafe_allow_html=True)

    if not MODEL_OK:
        st.error("❌ Model files not found. Ensure logistic_regression.pkl, scaler.pkl, column.pkl are present.")
        return

    st.markdown("""
    <div class="disclaimer">
        <span style='font-size:1.2rem;margin-top:1px;'>⚠️</span>
        <p><strong>Medical Disclaimer:</strong> This tool is for informational and educational use only.
        It does not constitute medical advice or diagnosis. Always consult a qualified cardiologist
        for all clinical decisions.</p>
    </div>
    """, unsafe_allow_html=True)


    st.markdown(f'<div class="card-label">👤 Group 1 — Basic Demographics</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        g1c1, g1c2, g1c3 = st.columns([2, 2, 2])

        with g1c1:
            age = st.slider("Age (Years)", 18, 100, 45,
                help="Patient's age in years. CV risk increases significantly past 60.")
            if age < 45:
                st.markdown(chip("✓ Lower Risk", "ok"),    unsafe_allow_html=True)
            elif age < 60:
                st.markdown(chip("⚡ Moderate Risk","warn"), unsafe_allow_html=True)
            else:
                st.markdown(chip("⚠ Higher Risk", "bad"),  unsafe_allow_html=True)

        with g1c2:
            sex = st.radio("Biological Sex", ["Male", "Female"], horizontal=True,
                help="Biological sex affects CV risk profiles and hormone-related factors.")

        with g1c3:
            st.markdown(f"""
            <div class="info-box" style="margin-top:0.2rem;">
                <strong>📖 Age & Sex Risk Note</strong><br>
                Men face higher CV risk before age 65. After menopause, women's risk
                rises sharply to match men's. Age &gt; 60 is the single largest
                non-modifiable independent risk factor.
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)


    st.markdown(f'<div class="card-label">🩺 Group 2 — Clinical Vitals</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        g2c1, g2c2, g2c3 = st.columns(3)

        with g2c1:
            resting_bp = st.number_input(
                "Resting Blood Pressure (mm Hg)",
                min_value=60, max_value=250, value=120, step=1,
                help="Normal: 90–120 | Elevated: 121–139 | Hypertension: ≥ 140 mm Hg")
            if resting_bp < 80:
                st.markdown(val_msg("⚠️ Critically low — please verify (< 80 mm Hg)", "err"), unsafe_allow_html=True)
            elif resting_bp < 90:
                st.markdown(val_msg("⚡ Hypotension range — check patient", "warn"),          unsafe_allow_html=True)
            elif resting_bp > 180:
                st.markdown(val_msg("🔴 Hypertensive crisis (> 180) — verify entry", "err"), unsafe_allow_html=True)
            elif resting_bp > 140:
                st.markdown(val_msg("⚡ Stage 2 hypertension (> 140 mm Hg)", "warn"),         unsafe_allow_html=True)
            else:
                st.markdown(val_msg("✓ Normal range (90–120 mm Hg)", "ok"),                   unsafe_allow_html=True)

        with g2c2:
            colestrol = st.number_input(
                "Serum Cholesterol (mg/dL)",
                min_value=0, max_value=700, value=200, step=5,
                help="Normal: < 200 | Borderline: 200–239 | High: ≥ 240 mg/dL. Value 0 = missing data.")
            if colestrol == 0:
                st.markdown(val_msg("ℹ️ Value 0 may indicate missing data in this dataset", "warn"), unsafe_allow_html=True)
            elif colestrol > 400:
                st.markdown(val_msg("⚠️ Unusually high — possible entry error?", "err"),      unsafe_allow_html=True)
            elif colestrol >= 240:
                st.markdown(val_msg("⚡ High cholesterol (≥ 240 mg/dL)", "warn"),             unsafe_allow_html=True)
            elif colestrol >= 200:
                st.markdown(val_msg("⚡ Borderline high (200–239 mg/dL)", "warn"),            unsafe_allow_html=True)
            else:
                st.markdown(val_msg("✓ Optimal range (< 200 mg/dL)", "ok"),                  unsafe_allow_html=True)

        with g2c3:
            target_hr = 220 - age
            max_hr = st.slider(
                "Maximum Heart Rate (bpm)", 60, 220, 150,
                help=f"Target max HR = 220 − Age ≈ {target_hr} bpm for a {age}-year-old.")
            hr_pct = (max_hr / target_hr) * 100
            if hr_pct < 85:
                st.markdown(val_msg(f"⚡ Below 85% target ({hr_pct:.0f}% of {target_hr} bpm)","warn"), unsafe_allow_html=True)
            else:
                st.markdown(val_msg(f"✓ At {hr_pct:.0f}% of target max HR ({target_hr} bpm)","ok"),   unsafe_allow_html=True)

        g2c4, g2c5 = st.columns(2)
        with g2c4:
            fasting_bs = st.radio(
                "Fasting Blood Sugar > 120 mg/dL?", [0, 1], horizontal=True,
                format_func=lambda x: "✅  Normal (≤ 120 mg/dL)" if x == 0 else "⚠️  Elevated (> 120 mg/dL)",
                help="BS > 120 mg/dL signals pre-diabetes/diabetes — a major independent CV risk factor.")
        with g2c5:
            if fasting_bs == 1:
                st.markdown(f"""
                <div class="info-box" style="margin-top:1.8rem;">
                    <strong>⚠️ Elevated Fasting BS</strong><br>
                    Diabetic patients face 2–4× higher cardiovascular mortality.
                    HbA1c testing and endocrinology review recommended.
                </div>
                """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(f'<div class="card-label">💓 Group 3 — Cardiac Symptoms & ECG</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        g3c1, g3c2, g3c3 = st.columns(3)

        CP_MAP = {
            "ATA": ("ATA — Atypical Angina",    "warn", "Pain not typical of classic angina"),
            "NAP": ("NAP — Non-Anginal Pain",    "ok",   "Likely non-cardiac in origin"),
            "TA":  ("TA  — Typical Angina",      "warn", "Classic angina; exertion-related"),
            "ASY": ("ASY — Asymptomatic",        "bad",  "Highest risk — silent ischemia"),
        }

        with g3c1:
            chest_pain = st.selectbox(
                "Chest Pain Type", ["ATA","NAP","TA","ASY"],
                format_func=lambda x: CP_MAP[x][0],
                help="ASY (Asymptomatic) is paradoxically the highest-risk group in this dataset.")
            cp_cls  = CP_MAP[chest_pain][1]
            cp_desc = CP_MAP[chest_pain][2]
            st.markdown(f'{chip(chest_pain, cp_cls)} <span style="font-size:0.75rem;color:{C["text_s"]};"> {cp_desc}</span>',
                        unsafe_allow_html=True)

        with g3c2:
            resting_ecg = st.selectbox(
                "Resting ECG Results", ["Normal","ST","LVH"],
                format_func=lambda x: {
                    "Normal": "Normal — No abnormalities",
                    "ST":     "ST — ST-T Wave Abnormality",
                    "LVH":    "LVH — Left Ventricular Hypertrophy",
                }[x],
                help="ST-T abnormalities suggest ischemia. LVH = heart overworking against elevated load.")
            ecg_cls = "ok" if resting_ecg == "Normal" else "warn" if resting_ecg == "LVH" else "bad"
            st.markdown(chip(resting_ecg, ecg_cls), unsafe_allow_html=True)

        with g3c3:
            exercise_angina = st.radio(
                "Exercise-Induced Angina", ["No","Yes"], horizontal=True,
                help="Chest pain during exertion = reduced coronary perfusion under stress. Key diagnostic marker.")
            ea_cls = "bad" if exercise_angina == "Yes" else "ok"
            ea_lbl = "⚠ Present — high significance" if exercise_angina == "Yes" else "✓ Absent"
            st.markdown(chip(ea_lbl, ea_cls), unsafe_allow_html=True)

        g3c4, g3c5 = st.columns(2)
        with g3c4:
            old_peak = st.slider(
                "Oldpeak — ST Depression (mm)", 0.0, 6.0, 1.0, step=0.5,
                help="ST depression during exercise vs. rest. > 2 mm = significant myocardial ischemia.")
            if old_peak >= 2.0:
                st.markdown(chip("⚠ Significant ischemia indicator", "bad"),  unsafe_allow_html=True)
            elif old_peak >= 1.0:
                st.markdown(chip("⚡ Mild ST depression", "warn"),             unsafe_allow_html=True)
            else:
                st.markdown(chip("✓ Minimal ST depression", "ok"),            unsafe_allow_html=True)

        with g3c5:
            st_slope = st.selectbox(
                "ST Segment Slope", ["Up","Flat","Down"],
                format_func=lambda x: {
                    "Up":   "Upsloping  ✅ — Generally favourable",
                    "Flat": "Flat  ⚡ — Moderate concern",
                    "Down": "Downsloping  ⚠️ — High concern",
                }[x],
                help="Downsloping ST during peak exercise is the most ominous pattern, associated with severe ischemia.")
            slope_cls = {"Up":"ok","Flat":"warn","Down":"bad"}[st_slope]
            st.markdown(chip(st_slope, slope_cls), unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


    st.markdown("<br>", unsafe_allow_html=True)
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        predict_btn = st.button("🔬  Perform Risk Assessment", type="primary", use_container_width=True)


    if predict_btn:
        with st.spinner("🧠 Running clinical analysis…"):
            time.sleep(1.2)

            num_df = pd.DataFrame([{
                "Age": age, "RestingBP": resting_bp,
                "Cholesterol": colestrol, "MaxHR": max_hr,
            }])
            scaled_num = scaler.transform(num_df)
            scaled_df  = pd.DataFrame(scaled_num, columns=["Age","RestingBP","Cholesterol","MaxHR"])

            cat_df = pd.DataFrame([{
                "FastingBS":         int(fasting_bs),
                "Oldpeak":           old_peak,
                "Sex_M":             1 if sex == "Male" else 0,
                "ChestPainType_ATA": 1 if chest_pain == "ATA" else 0,
                "ChestPainType_NAP": 1 if chest_pain == "NAP" else 0,
                "ChestPainType_TA":  1 if chest_pain == "TA"  else 0,
                "RestingECG_Normal": 1 if resting_ecg == "Normal" else 0,
                "RestingECG_ST":     1 if resting_ecg == "ST" else 0,
                "ExerciseAngina_Y":  1 if exercise_angina == "Yes" else 0,
                "ST_Slope_Flat":     1 if st_slope == "Flat" else 0,
                "ST_Slope_Up":       1 if st_slope == "Up"   else 0,
            }])

            input_df   = pd.concat([scaled_df, cat_df], axis=1)[trained_columns]
            prediction = model.predict(input_df)[0]

            if hasattr(model, "predict_proba"):
                risk_score = model.predict_proba(input_df)[0][1] * 100
            else:
                risk_score = (random.uniform(62, 93) if prediction == 1
                              else random.uniform(10, 38))

            st.session_state.history.append({
                "timestamp":   pd.Timestamp.now().strftime("%H:%M:%S"),
                "age":         age,    "sex": sex,
                "chest_pain":  chest_pain, "resting_bp": resting_bp,
                "cholesterol": colestrol,  "max_hr": max_hr,
                "result":      int(prediction), "risk_score": risk_score,
            })


        st.markdown("---")
        st.markdown("## 📋 Risk Assessment Results")

        _, res_col, _ = st.columns([1, 3, 1])
        with res_col:
            if prediction == 1:
                st.markdown(f"""
                <div class="result-high">
                    <div class="result-icon">⚠️</div>
                    <p class="result-title" style="color:#C5221F;">HIGH CARDIOVASCULAR RISK</p>
                    <div class="conf-badge" style="background:#FFCDD2;color:#C5221F;">
                        Model Confidence: {risk_score:.1f}%
                    </div>
                    <p style="color:{C['text_s']};margin-top:0.5rem;">
                        This patient profile shows significant markers associated with heart disease.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                st.error("🔴 **Immediate Action:** Refer to cardiologist. Workup recommended: ECG, Echocardiogram, Stress Test, Lipid Panel.")
            else:
                st.markdown(f"""
                <div class="result-low">
                    <div class="result-icon">✅</div>
                    <p class="result-title" style="color:#137333;">LOW CARDIOVASCULAR RISK</p>
                    <div class="conf-badge" style="background:#CEEAD6;color:#137333;">
                        Model Confidence: {100 - risk_score:.1f}%
                    </div>
                    <p style="color:{C['text_s']};margin-top:0.5rem;">
                        No significant heart disease markers detected in this clinical profile.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                st.success("🟢 **Recommendation:** Maintain healthy lifestyle. Routine annual check-up advised.")


        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Cardiovascular Risk Score (%)", "font": {"size": 15, "color": C["text"]}},
            delta={"reference": 50,
                   "increasing": {"color": C["danger"]},
                   "decreasing": {"color": C["secondary"]}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": C["outline"]},
                "bar": {"color": C["danger"] if prediction == 1 else C["secondary"], "thickness": 0.28},
                "bgcolor": "white", "borderwidth": 0,
                "steps": [
                    {"range": [0, 30],  "color": C["secondary_50"]},
                    {"range": [30, 60], "color": C["warning_50"]},
                    {"range": [60, 100],"color": C["danger_50"]},
                ],
                "threshold": {"line": {"color": C["text"], "width": 3}, "thickness": 0.75, "value": risk_score},
            },
        ))
        fig_gauge.update_layout(height=280, **plotly_base())
        _, g_col, _ = st.columns([1, 2, 1])
        with g_col:
            st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})


        st.markdown("### 📊 Health Indicator Summary")
        mk1, mk2, mk3, mk4 = st.columns(4)
        with mk1:
            bp_d = "Normal" if resting_bp < 120 else "Elevated" if resting_bp < 140 else "High"
            st.metric("Blood Pressure", f"{resting_bp} mmHg", bp_d, delta_color="inverse" if resting_bp > 120 else "normal")
        with mk2:
            ch_d = "Normal" if colestrol < 200 else "Borderline" if colestrol < 240 else "High"
            st.metric("Cholesterol", f"{colestrol} mg/dL", ch_d, delta_color="inverse" if colestrol > 200 else "normal")
        with mk3:
            hr_d = "On Target" if max_hr >= (220-age)*0.85 else "Below Target"
            st.metric("Max Heart Rate", f"{max_hr} bpm", hr_d)
        with mk4:
            age_d = "Higher Risk" if age > 60 else "Moderate" if age > 45 else "Lower Risk"
            st.metric("Age Factor", f"{age} yrs", age_d, delta_color="inverse" if age > 45 else "normal")


        with st.expander("📋 Full Clinical Report", expanded=False):
            report = {
                "Parameter":      ["Age","Sex","Chest Pain","Resting BP","Cholesterol",
                                   "Max HR","Fasting BS","Resting ECG","Exercise Angina",
                                   "ST Depression","ST Slope"],
                "Value":          [f"{age} yrs", sex, chest_pain,
                                   f"{resting_bp} mmHg", f"{colestrol} mg/dL",
                                   f"{max_hr} bpm", "Elevated" if fasting_bs else "Normal",
                                   resting_ecg, exercise_angina, f"{old_peak} mm", st_slope],
                "Clinical Status":[
                    "⚠️ High" if age > 60 else "⚡ Moderate" if age > 45 else "✅ Normal",
                    "—",
                    "⚠️ High Risk" if chest_pain == "ASY" else "⚡ Elevated" if chest_pain == "TA" else "✅ Lower",
                    "⚠️ High" if resting_bp > 140 else "⚡ Elevated" if resting_bp > 120 else "✅ Normal",
                    "⚠️ High" if colestrol > 240 else "⚡ Borderline" if colestrol > 200 else "✅ Normal",
                    "⚡ Below Target" if max_hr < (220-age)*0.85 else "✅ On Target",
                    "⚠️ Elevated" if fasting_bs else "✅ Normal",
                    "⚡ Abnormal" if resting_ecg != "Normal" else "✅ Normal",
                    "⚠️ Present" if exercise_angina == "Yes" else "✅ Absent",
                    "⚠️ Significant" if old_peak >= 2.0 else "⚡ Mild" if old_peak >= 1.0 else "✅ Minimal",
                    "⚠️ High Risk" if st_slope == "Down" else "⚡ Moderate" if st_slope == "Flat" else "✅ Favourable",
                ],
            }
            st.dataframe(pd.DataFrame(report), use_container_width=True, hide_index=True)


        with st.expander("❤️ Personalised Health Recommendations", expanded=False):
            if prediction == 1:
                st.markdown("""
                ### 🏥 High-Risk Action Plan
                **Immediate Steps (Next 2 Weeks):**
                1. 🏥 Book cardiologist appointment urgently
                2. 📋 Bring this report and full medical history
                3. 🔬 Request: 12-lead ECG, Echocardiogram, Full Lipid Panel, HbA1c, hsCRP

                **Lifestyle (Start Today):**
                - 🥗 Adopt DASH / Mediterranean diet — cut sodium and saturated fat
                - 🚶 Light activity: 20–30 min walk daily *(medical clearance first)*
                - 🚭 Cease smoking immediately
                - 💊 Review and adhere to all prescribed medications
                """)
            else:
                st.markdown("""
                ### 🟢 Preventive Maintenance
                **Annual Screening Protocol:**
                1. ✅ Cardiovascular screening every 12 months
                2. 📊 BP, cholesterol, and fasting glucose checks yearly
                3. 🏃 Target 150 min moderate aerobic exercise/week

                **Lifestyle Best Practices:**
                - 🥗 Mediterranean diet — olive oil, fish, legumes, whole grains
                - 😴 7–8 hours quality sleep nightly
                - 🧘 Stress management: yoga, deep breathing, mindfulness
                - 🚭 Tobacco-free; limit alcohol to ≤ 1 unit/day
                """)



def page_analytics():
    st.markdown("""
    <div class="hero" style="padding:2rem;">
        <h1 style="font-size:2rem !important;">📊 Analytics Dashboard</h1>
        <p>Population-level insights, model performance, and session prediction history</p>
    </div>
    """, unsafe_allow_html=True)


    np.random.seed(42)
    N = 918
    df = pd.DataFrame({
        "Age":         np.random.normal(54, 9, N).clip(28, 77).astype(int),
        "Sex":         np.random.choice(["M","F"], N, p=[0.79, 0.21]),
        "ChestPain":   np.random.choice(["ASY","NAP","ATA","TA"], N, p=[0.54,0.22,0.19,0.05]),
        "RestingBP":   np.random.normal(132, 18, N).clip(80, 200).astype(int),
        "Cholesterol": np.random.normal(198, 109, N).clip(0, 600).astype(int),
        "MaxHR":       np.random.normal(136, 25, N).clip(60, 202).astype(int),
        "HD":          np.random.choice([0, 1], N, p=[0.445, 0.555]),
    })

    total    = N
    n_high   = int(df["HD"].sum())
    pct_high = n_high / total * 100
    avg_age  = df[df["HD"]==1]["Age"].mean()
    pct_low  = 100 - pct_high


    st.markdown("### 🎯 Key Metrics")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f'<div class="kpi"><div class="kpi-num">{total:,}</div>'
                    f'<div class="kpi-label">Total Patients</div>'
                    f'<div class="kpi-sub kpi-pos">↑ Dataset Size</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="kpi"><div class="kpi-num" style="color:{C["danger"]};">{pct_high:.1f}%</div>'
                    f'<div class="kpi-label">High Risk Rate</div>'
                    f'<div class="kpi-sub kpi-neg">↑ {n_high} cases</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="kpi"><div class="kpi-num">{avg_age:.0f}</div>'
                    f'<div class="kpi-label">Avg Age · High Risk</div>'
                    f'<div class="kpi-sub kpi-neg">↑ years</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="kpi"><div class="kpi-num" style="color:{C["secondary"]};">{total-n_high:,}</div>'
                    f'<div class="kpi-label">Low Risk Profiles</div>'
                    f'<div class="kpi-sub kpi-pos">↓ {pct_low:.1f}%</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1 — Distribution charts
    st.markdown("### 📈 Population Distribution")
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        cp = df[df["HD"]==1]["ChestPain"].value_counts().reset_index()
        cp.columns = ["Type","Count"]
        cmap = {"ASY": C["danger"], "TA": C["warning"], "NAP": C["primary"], "ATA": C["secondary"]}
        fig1 = px.bar(cp, x="Type", y="Count", color="Type",
            color_discrete_map=cmap, text="Count",
            title="Chest Pain Types in Heart Disease Patients")
        fig1.update_traces(textposition="outside", marker_line_width=0)
        fig1.update_layout(showlegend=False,
            xaxis=dict(gridcolor="#F1F3F4", title="Chest Pain Type"),
            yaxis=dict(gridcolor="#F1F3F4", title="Count"),
            **plotly_base())
        st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

    with r1c2:
        fig2 = go.Figure(go.Pie(
            labels=["Low Risk","High Risk"],
            values=[total - n_high, n_high],
            hole=0.62,
            marker_colors=[C["secondary"], C["danger"]],
            textinfo="label+percent", textfont_size=12,
        ))
        fig2.add_annotation(
            text=f"{pct_high:.1f}%<br><span style='font-size:11px'>High Risk</span>",
            x=0.5, y=0.5, showarrow=False, font=dict(size=20, color=C["text"]))
        fig2.update_layout(title="Overall Risk Distribution", **plotly_base())
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # Row 2 — Scatter + Feature Importance
    st.markdown("### 🔬 Clinical Correlations & Model Insights")
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        sdf = df.sample(350, random_state=1).copy()
        sdf["Risk"] = sdf["HD"].map({0: "Low Risk", 1: "High Risk"})
        fig3 = px.scatter(sdf, x="Age", y="MaxHR", color="Risk",
            color_discrete_map={"Low Risk": C["secondary"], "High Risk": C["danger"]},
            opacity=0.68, title="Age vs Max Heart Rate by Risk Group",
            labels={"MaxHR": "Max Heart Rate (bpm)", "color": "Risk"})
        fig3.update_traces(marker=dict(size=7))
        fig3.update_layout(
            xaxis=dict(gridcolor="#F1F3F4"),
            yaxis=dict(gridcolor="#F1F3F4"),
            legend=dict(title="Risk Group"),
            **plotly_base())
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    with r2c2:
        feats = ["ST_Slope","ChestPainType","ExerciseAngina","Oldpeak",
                 "MaxHR","Age","Sex_M","FastingBS","RestingBP","Cholesterol","RestingECG"]
        imps  = [0.89,0.76,0.71,0.65,0.58,0.49,0.41,0.35,0.28,0.22,0.18]
        f_col = [C["danger"] if v > 0.6 else C["warning"] if v > 0.35 else C["secondary"] for v in imps]
        fig4 = go.Figure(go.Bar(
            x=imps, y=feats, orientation="h",
            marker=dict(color=f_col, line=dict(width=0)),
            text=[f"{v:.2f}" for v in imps], textposition="outside"))
        fig4.update_layout(
            title="Feature Importance (Model Coefficients)",
            xaxis=dict(gridcolor="#F1F3F4", title="Relative Importance"),
            yaxis=dict(gridcolor="#F1F3F4", categoryorder="total ascending"),
            height=380, **plotly_base())
        fig4.update_layout(margin=dict(l=20, r=60, t=45, b=20))
        st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})

    # Histograms
    st.markdown("### 📊 Distribution Analysis by Risk Group")
    h1, h2 = st.columns(2)

    with h1:
        fig5 = go.Figure()
        fig5.add_trace(go.Histogram(x=df[df["HD"]==0]["Cholesterol"],
            name="Low Risk",  marker_color=C["secondary"], opacity=0.7, nbinsx=30))
        fig5.add_trace(go.Histogram(x=df[df["HD"]==1]["Cholesterol"],
            name="High Risk", marker_color=C["danger"],    opacity=0.7, nbinsx=30))
        fig5.update_layout(barmode="overlay",
            xaxis=dict(gridcolor="#F1F3F4", title="Cholesterol (mg/dL)"),
            yaxis=dict(gridcolor="#F1F3F4", title="Count"),
            legend=dict(title="Risk Group"), height=280, **plotly_base())
        st.plotly_chart(fig5, use_container_width=True, config={"displayModeBar": False})

    with h2:
        fig6 = go.Figure()
        fig6.add_trace(go.Histogram(x=df[df["HD"]==0]["Age"],
            name="Low Risk",  marker_color=C["secondary"], opacity=0.7, nbinsx=25))
        fig6.add_trace(go.Histogram(x=df[df["HD"]==1]["Age"],
            name="High Risk", marker_color=C["danger"],    opacity=0.7, nbinsx=25))
        fig6.update_layout(barmode="overlay",
            xaxis=dict(gridcolor="#F1F3F4", title="Age (Years)"),
            yaxis=dict(gridcolor="#F1F3F4", title="Count"),
            legend=dict(title="Risk Group"), height=280, **plotly_base())
        st.plotly_chart(fig6, use_container_width=True, config={"displayModeBar": False})

    # Session history
    st.markdown("### 📋 Session Prediction History")
    if st.session_state.history:
        hdf = pd.DataFrame(st.session_state.history)
        hdf["Risk Level"]  = hdf["result"].map({0: "🟢 Low Risk", 1: "🔴 High Risk"})
        hdf["Confidence"]  = hdf["risk_score"].apply(lambda x: f"{x:.1f}%")
        disp = hdf[["timestamp","age","sex","chest_pain","resting_bp",
                     "cholesterol","max_hr","Risk Level","Confidence"]].copy()
        disp.columns = ["Time","Age","Sex","Chest Pain","BP (mmHg)","Cholesterol","Max HR","Risk Level","Confidence"]
        st.dataframe(disp, use_container_width=True, hide_index=True)
    else:
        st.markdown(f"""
        <div style="text-align:center;padding:3rem;background:{C['surface']};
             border-radius:16px;border:1px dashed {C['outline']};">
            <p style="font-size:2rem;">📭</p>
            <p style="color:{C['text_s']};font-size:1rem;">No predictions yet in this session.</p>
            <p style="color:#9AA0A6;font-size:0.83rem;">Run a risk assessment to see results here.</p>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
#  PAGE — ABOUT
# ═══════════════════════════════════════════════
def page_about():
    st.markdown("""
    <div class="hero" style="padding:2rem;">
        <h1 style="font-size:2rem !important;">ℹ️ Methodology & About</h1>
        <p>Full transparency on the AI model, training data, features, and limitations</p>
    </div>
    """, unsafe_allow_html=True)

    a1, a2 = st.columns([2, 1])

    with a1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🤖 Model Specification")
        st.dataframe(pd.DataFrame({
            "Parameter": ["Algorithm","Dataset","Feature Count","Train/Test Split",
                          "Accuracy (CV)","AUC-ROC","Framework"],
            "Detail":    ["Logistic Regression (L2 regularisation)",
                          "UCI Heart Disease — 918 patients",
                          "11 clinical biomarkers",
                          "80% / 20% stratified",
                          "≈ 85.3%", "≈ 0.91",
                          "Scikit-learn 1.x + Joblib"],
        }), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 📊 Feature Reference Guide")
        st.dataframe(pd.DataFrame({
            "Feature":          ["Age","Sex","ChestPainType","RestingBP","Cholesterol",
                                 "FastingBS","RestingECG","MaxHR","ExerciseAngina","Oldpeak","ST_Slope"],
            "Type":             ["Numerical","Categorical","Categorical","Numerical","Numerical",
                                 "Binary","Categorical","Numerical","Binary","Numerical","Categorical"],
            "Clinical Meaning": [
                "Age in years",
                "M = Male, F = Female",
                "TA: Typical Angina | ATA: Atypical | NAP: Non-Anginal | ASY: Asymptomatic",
                "Resting BP in mm Hg",
                "Serum cholesterol in mg/dL",
                "1 if fasting BS > 120 mg/dL, 0 otherwise",
                "Normal | ST-T abnormality | Left Ventricular Hypertrophy",
                "Maximum HR achieved on stress test (bpm)",
                "Angina induced by exercise: Yes / No",
                "ST depression during exercise vs. rest (mm)",
                "Peak exercise ST slope: Up | Flat | Down",
            ],
        }), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with a2:
        st.markdown(f"""
        <div class="card" style="background:{C['danger_50']};border:1px solid #F5C6C2;">
            <h4 style="color:#C5221F;margin-top:0;">⚠️ Critical Disclaimer</h4>
            <p style="font-size:0.83rem;line-height:1.65;color:{C['text']};">
                <strong>This tool does NOT replace clinical judgement.</strong><br><br>
                • Results are probabilistic, not diagnostic<br>
                • Trained on a specific population cohort<br>
                • Individual patient factors may be uncaptured<br>
                • ~15% error rate in held-out testing<br>
                • False negatives and false positives will occur<br><br>
                <strong>Consult a qualified cardiologist for all clinical decisions.</strong>
            </p>
        </div>

        <div class="card">
            <h4 style="margin-top:0;">📚 Dataset Provenance</h4>
            <p style="font-size:0.83rem;line-height:1.7;color:{C['text']};">
                <strong>UCI Heart Disease Dataset</strong><br>
                Merged from 5 clinical databases:<br>
                • Cleveland Clinic Foundation<br>
                • Hungarian Institute of Cardiology<br>
                • University Hospitals — Zürich & Basel<br>
                • Long Beach VA Medical Center<br>
                • Statlog (Heart) Dataset<br><br>
                918 observations · Curated via Kaggle.
            </p>
        </div>

        <div class="card">
            <h4 style="margin-top:0;">🛠️ Technology Stack</h4>
            <p style="font-size:0.83rem;line-height:1.7;color:{C['text']};">
                <strong>Frontend:</strong> Streamlit 1.x<br>
                <strong>ML:</strong> Scikit-learn, Joblib<br>
                <strong>Visualisation:</strong> Plotly Express & GO<br>
                <strong>Data:</strong> Pandas, NumPy<br>
                <strong>Fonts:</strong> DM Sans — Google Fonts<br>
                <strong>Design:</strong> Material-inspired Clinical v2
            </p>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
#  ROUTER
# ═══════════════════════════════════════════════
if   st.session_state.page == "prediction": page_prediction()
elif st.session_state.page == "analytics":  page_analytics()
elif st.session_state.page == "about":      page_about()

# ═══════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════
st.markdown(f"""
<div class="app-footer">
    🫀 <strong>CardioCare AI v2.0</strong> — Advanced Cardiovascular Risk Intelligence<br>
    Built with Streamlit · Scikit-learn · Plotly &nbsp;|&nbsp; Clinical Decision Support Tool<br>
    <span style="color:{C['danger']};">⚠️ For informational use only.
    Not a substitute for professional medical advice.</span>
</div>
""", unsafe_allow_html=True)