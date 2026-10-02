"""
╔══════════════════════════════════════════════════════════════════════════╗
║       🏃 Human Activity Recognition — LSTM & GRU Dashboard 🏃        ║
║                  Ultra-Premium Interactive Analytics                    ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import json, os, math, time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from streamlit_lottie import st_lottie
import requests

# ─────────────────── PAGE CONFIG ──────────────────────────────────
st.set_page_config(
    page_title="HAR — LSTM & GRU Dashboard",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────── LOAD DATA ────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "model_outputs")

@st.cache_data
def load_json(filename):
    with open(os.path.join(OUTPUT_DIR, filename), "r") as f:
        return json.load(f)

try:
    data_stats = load_json("data_stats.json")
    lstm_results = load_json("LSTM_results.json")
    gru_results = load_json("GRU_results.json")
    comparison = load_json("comparison.json")
except FileNotFoundError:
    st.error("⚠️ Model outputs not found. Please run `python train_models.py` first!")
    st.stop()

ACTIVITY_LABELS = data_stats["activity_labels"]
ACTIVITY_EMOJIS = {
    "WALKING": "🚶", "WALKING_UPSTAIRS": "🧗", "WALKING_DOWNSTAIRS": "⬇️",
    "SITTING": "🪑", "STANDING": "🧍", "LAYING": "🛌"
}

# ─────────────────── LOTTIE ANIMATIONS ────────────────────────────
@st.cache_data
def load_lottie_url(url: str):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

lottie_walking = load_lottie_url("https://lottie.host/4db68bbd-31f6-4cd8-84eb-189571e0dc7a/BhEJVGLN59.json")
lottie_brain   = load_lottie_url("https://lottie.host/c8e20547-63fd-4a81-b8dd-dfe0fdb9f8b4/a3kJz1CGNG.json")
lottie_chart   = load_lottie_url("https://lottie.host/71eac90b-3e6c-4bb2-beaf-aa6359e5b7cc/hGGRWCb1PY.json")
lottie_success = load_lottie_url("https://lottie.host/37b07c90-efde-4935-aa18-9033ceafa803/85zrpQoJBD.json")

# ─────────────────── CUSTOM CSS — THE MAGIC ───────────────────────
st.markdown("""
<style>
/* ────── Google Font ────── */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

/* ────── Root Palette ────── */
:root {
    --bg-dark: #0a0e1a;
    --bg-card: rgba(15, 23, 42, 0.65);
    --bg-glass: rgba(30, 41, 82, 0.35);
    --accent-1: #6366f1;   /* indigo */
    --accent-2: #8b5cf6;   /* violet */
    --accent-3: #a78bfa;   /* light violet */
    --accent-4: #06b6d4;   /* cyan */
    --accent-5: #22d3ee;   /* light cyan */
    --accent-6: #f43f5e;   /* rose */
    --accent-7: #10b981;   /* emerald */
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --border-glow: rgba(99, 102, 241, 0.3);
    --gradient-1: linear-gradient(135deg, #6366f1, #8b5cf6, #a78bfa);
    --gradient-2: linear-gradient(135deg, #06b6d4, #22d3ee);
    --gradient-3: linear-gradient(135deg, #f43f5e, #fb7185);
    --gradient-4: linear-gradient(135deg, #10b981, #34d399);
}

/* ────── Global Overrides ────── */
html, body, [data-testid="stAppViewContainer"], .stApp {
    background: var(--bg-dark) !important;
    color: var(--text-primary) !important;
    font-family: 'Outfit', sans-serif !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%) !important;
    border-right: 1px solid var(--border-glow) !important;
}

[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

/* ────── Scrollbar ────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-dark); }
::-webkit-scrollbar-thumb { background: var(--accent-1); border-radius: 3px; }

/* ────── Hero Header ────── */
.hero-container {
    text-align: center;
    padding: 2rem 1rem 1rem;
    position: relative;
    overflow: hidden;
}

.hero-container::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: conic-gradient(from 0deg at 50% 50%,
        transparent 0deg,
        rgba(99,102,241,0.05) 60deg,
        transparent 120deg,
        rgba(6,182,212,0.05) 180deg,
        transparent 240deg,
        rgba(244,63,94,0.05) 300deg,
        transparent 360deg);
    animation: heroRotate 20s linear infinite;
}

@keyframes heroRotate {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

.hero-title {
    font-family: 'Outfit', sans-serif;
    font-size: 3.2rem;
    font-weight: 900;
    background: linear-gradient(135deg, #6366f1 0%, #a78bfa 30%, #06b6d4 60%, #22d3ee 100%);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientShift 5s ease infinite;
    position: relative;
    z-index: 1;
    margin-bottom: 0.5rem;
    letter-spacing: -1px;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.hero-subtitle {
    font-size: 1.15rem;
    color: var(--text-secondary);
    font-weight: 400;
    position: relative;
    z-index: 1;
    letter-spacing: 0.5px;
}

/* ────── Metric Cards ────── */
.metric-card {
    background: var(--bg-card);
    backdrop-filter: blur(20px);
    border: 1px solid var(--border-glow);
    border-radius: 20px;
    padding: 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}

.metric-card:hover {
    transform: translateY(-6px) scale(1.02);
    border-color: var(--accent-1);
    box-shadow: 0 20px 60px -10px rgba(99,102,241,0.25),
                0 0 40px -10px rgba(99,102,241,0.2);
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    border-radius: 20px 20px 0 0;
}

.metric-card.indigo::before { background: var(--gradient-1); }
.metric-card.cyan::before   { background: var(--gradient-2); }
.metric-card.rose::before   { background: var(--gradient-3); }
.metric-card.emerald::before { background: var(--gradient-4); }

.metric-value {
    font-size: 2.4rem;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    margin: 0.5rem 0 0.25rem;
    line-height: 1;
}

.metric-value.indigo  { color: var(--accent-1); }
.metric-value.cyan    { color: var(--accent-4); }
.metric-value.rose    { color: var(--accent-6); }
.metric-value.emerald { color: var(--accent-7); }

.metric-label {
    font-size: 0.85rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 600;
}

.metric-emoji {
    font-size: 2rem;
    display: block;
    margin-bottom: 0.25rem;
}

/* ────── Glass Panels ────── */
.glass-panel {
    background: var(--bg-glass);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 20px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}

.glass-panel:hover {
    border-color: rgba(99,102,241,0.35);
    box-shadow: 0 8px 32px -8px rgba(99,102,241,0.15);
}

/* ────── Section Title ────── */
.section-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    margin: 2rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.section-title .icon {
    font-size: 1.6rem;
}

.section-title .text {
    background: linear-gradient(90deg, var(--accent-3), var(--accent-5));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ────── Model Comparison Badge ────── */
.model-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1.25rem;
    border-radius: 50px;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 1px;
    font-family: 'JetBrains Mono', monospace;
}

.model-badge.lstm {
    background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(139,92,246,0.2));
    border: 1px solid rgba(99,102,241,0.4);
    color: var(--accent-3);
}

.model-badge.gru {
    background: linear-gradient(135deg, rgba(6,182,212,0.2), rgba(34,211,238,0.2));
    border: 1px solid rgba(6,182,212,0.4);
    color: var(--accent-5);
}

/* ────── Rotating Ring Animation ────── */
.rotating-ring-container {
    position: relative;
    width: 180px;
    height: 180px;
    margin: 0 auto 1rem;
}

.rotating-ring {
    position: absolute;
    border-radius: 50%;
    border: 2px solid transparent;
}

.ring-1 {
    inset: 0;
    border-top-color: var(--accent-1);
    border-right-color: var(--accent-2);
    animation: spinRing 3s linear infinite;
}

.ring-2 {
    inset: 12px;
    border-bottom-color: var(--accent-4);
    border-left-color: var(--accent-5);
    animation: spinRing 2s linear infinite reverse;
}

.ring-3 {
    inset: 24px;
    border-top-color: var(--accent-6);
    border-right-color: var(--accent-7);
    animation: spinRing 4s linear infinite;
}

.ring-center {
    position: absolute;
    inset: 36px;
    border-radius: 50%;
    background: var(--bg-card);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
}

.ring-center.indigo { color: var(--accent-1); }
.ring-center.cyan   { color: var(--accent-4); }

@keyframes spinRing {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

/* ────── Pulsing Dot ────── */
.pulse-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
    animation: pulse 2s ease-in-out infinite;
}

.pulse-dot.green { background: var(--accent-7); }
.pulse-dot.blue  { background: var(--accent-1); }

@keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.5); opacity: 0.5; }
}

/* ────── Floating Particles ────── */
.particles-container {
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
}

.particle {
    position: absolute;
    width: 4px;
    height: 4px;
    border-radius: 50%;
    opacity: 0.15;
    animation: float linear infinite;
}

@keyframes float {
    0% { transform: translateY(100vh) rotate(0deg); opacity: 0; }
    10% { opacity: 0.15; }
    90% { opacity: 0.15; }
    100% { transform: translateY(-10vh) rotate(720deg); opacity: 0; }
}

/* ────── Activity Icons Row ────── */
.activity-icons {
    display: flex;
    justify-content: center;
    gap: 1.5rem;
    padding: 1rem 0;
    flex-wrap: wrap;
}

.activity-icon {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.35rem;
    padding: 0.75rem 1rem;
    border-radius: 16px;
    background: var(--bg-glass);
    border: 1px solid rgba(99,102,241,0.1);
    transition: all 0.3s ease;
    min-width: 95px;
}

.activity-icon:hover {
    transform: translateY(-4px);
    border-color: rgba(99,102,241,0.4);
    box-shadow: 0 8px 24px -6px rgba(99,102,241,0.2);
}

.activity-icon .emoji { font-size: 1.8rem; }
.activity-icon .name {
    font-size: 0.65rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
}

/* ────── Winner Banner ────── */
.winner-banner {
    background: linear-gradient(135deg, rgba(16,185,129,0.1), rgba(52,211,153,0.05));
    border: 1px solid rgba(16,185,129,0.3);
    border-radius: 16px;
    padding: 1rem 1.5rem;
    text-align: center;
    margin: 1rem 0;
    animation: winnerGlow 3s ease-in-out infinite;
}

@keyframes winnerGlow {
    0%, 100% { box-shadow: 0 0 20px rgba(16,185,129,0.1); }
    50% { box-shadow: 0 0 40px rgba(16,185,129,0.25); }
}

.winner-text {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--accent-7);
}

/* ────── Streamlit Overrides ────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent !important;
}

.stTabs [data-baseweb="tab"] {
    background: var(--bg-glass) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(99,102,241,0.15) !important;
    color: var(--text-secondary) !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.25rem !important;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(139,92,246,0.15)) !important;
    border-color: var(--accent-1) !important;
    color: var(--accent-3) !important;
}

div[data-testid="stSelectbox"] label,
div[data-testid="stMultiSelect"] label {
    color: var(--text-secondary) !important;
}

/* ────── Hide streamlit elements ────── */
#MainMenu, footer, [data-testid="stDecoration"] { display: none !important; }

/* ────── Orbit Animation for Comparison ────── */
.orbit-container {
    position: relative;
    width: 250px;
    height: 250px;
    margin: 0 auto;
}

.orbit-center {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 60px;
    height: 60px;
    border-radius: 50%;
    background: var(--gradient-1);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    z-index: 2;
    box-shadow: 0 0 30px rgba(99,102,241,0.4);
}

.orbit-path {
    position: absolute;
    top: 50%;
    left: 50%;
    border: 1px dashed rgba(99,102,241,0.2);
    border-radius: 50%;
    animation: orbitFade 4s ease-in-out infinite;
}

.orbit-path-1 {
    width: 140px; height: 140px;
    margin-top: -70px; margin-left: -70px;
}

.orbit-path-2 {
    width: 200px; height: 200px;
    margin-top: -100px; margin-left: -100px;
}

@keyframes orbitFade {
    0%, 100% { opacity: 0.3; }
    50% { opacity: 0.8; }
}

/* ────── Data Flow Animation ────── */
.data-flow {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    padding: 1rem;
    margin: 1rem 0;
}

.flow-node {
    padding: 0.6rem 1rem;
    border-radius: 12px;
    font-size: 0.8rem;
    font-weight: 600;
    text-align: center;
    animation: flowPulse 2s ease-in-out infinite;
}

.flow-arrow {
    color: var(--accent-4);
    font-size: 1.2rem;
    animation: arrowBounce 1.5s ease-in-out infinite;
}

@keyframes flowPulse {
    0%, 100% { opacity: 0.7; transform: scale(1); }
    50% { opacity: 1; transform: scale(1.05); }
}

@keyframes arrowBounce {
    0%, 100% { transform: translateX(0); }
    50% { transform: translateX(4px); }
}

/* ────── Hyperparameter Table ────── */
.hp-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid rgba(99,102,241,0.15);
}

.hp-table th {
    background: rgba(99,102,241,0.15);
    padding: 0.75rem 1rem;
    text-align: left;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--accent-3);
    font-weight: 700;
}

.hp-table td {
    padding: 0.65rem 1rem;
    border-bottom: 1px solid rgba(99,102,241,0.07);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: var(--text-primary);
}

.hp-table tr:last-child td { border-bottom: none; }

.hp-table tr:hover td {
    background: rgba(99,102,241,0.05);
}

/* ────── Neon glow on Plotly charts ────── */
[data-testid="stPlotlyChart"] {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid rgba(99,102,241,0.1);
}
</style>
""", unsafe_allow_html=True)

# ────── Floating Particles Background ──────
particles_html = '<div class="particles-container">'
import random
random.seed(42)
for i in range(20):
    colors = ["#6366f1", "#8b5cf6", "#06b6d4", "#22d3ee", "#f43f5e", "#10b981"]
    c = colors[i % len(colors)]
    left = random.randint(0, 100)
    delay = random.uniform(0, 15)
    dur = random.uniform(15, 30)
    size = random.randint(2, 6)
    particles_html += (
        f'<div class="particle" style="left:{left}%;width:{size}px;height:{size}px;'
        f'background:{c};animation-delay:{delay}s;animation-duration:{dur}s;"></div>'
    )
particles_html += '</div>'
st.markdown(particles_html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# ░░░░░░░░░░░░░░░░░  SIDEBAR  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem;">
        <div style="font-size:3rem; margin-bottom:0.5rem;">🏃</div>
        <div style="font-size:1.3rem; font-weight:800; 
            background: linear-gradient(90deg, #6366f1, #06b6d4);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            HAR Dashboard
        </div>
        <div style="font-size:0.75rem; color:#94a3b8; letter-spacing:2px; text-transform:uppercase; margin-top:0.25rem;">
            LSTM · GRU · Deep Learning
        </div>
    </div>
    <hr style="border-color: rgba(99,102,241,0.2); margin: 1rem 0;">
    """, unsafe_allow_html=True)

    nav = st.radio(
        "🧭 Navigation",
        ["🏠 Overview", "📊 Model Comparison", "📈 Training Analytics",
         "🔬 Confusion Matrices", "🌊 Sensor Signals", "🎯 Per-Class Analysis",
         "⚙️ Architecture"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color: rgba(99,102,241,0.2); margin: 1rem 0;'>", unsafe_allow_html=True)

    # Sidebar stats
    st.markdown(f"""
    <div class="glass-panel" style="padding:1rem;">
        <div style="font-size:0.7rem; text-transform:uppercase; letter-spacing:2px; color:#94a3b8; margin-bottom:0.75rem;">
            📋 Dataset Summary
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:0.4rem;">
            <span style="color:#94a3b8; font-size:0.8rem;">Train Samples</span>
            <span style="color:#a78bfa; font-weight:700; font-family:'JetBrains Mono',monospace; font-size:0.8rem;">{data_stats['train_samples']:,}</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:0.4rem;">
            <span style="color:#94a3b8; font-size:0.8rem;">Test Samples</span>
            <span style="color:#22d3ee; font-weight:700; font-family:'JetBrains Mono',monospace; font-size:0.8rem;">{data_stats['test_samples']:,}</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:0.4rem;">
            <span style="color:#94a3b8; font-size:0.8rem;">Time Steps</span>
            <span style="color:#fb7185; font-weight:700; font-family:'JetBrains Mono',monospace; font-size:0.8rem;">{data_stats['n_timesteps']}</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:0.4rem;">
            <span style="color:#94a3b8; font-size:0.8rem;">Channels</span>
            <span style="color:#34d399; font-weight:700; font-family:'JetBrains Mono',monospace; font-size:0.8rem;">{data_stats['n_channels']}</span>
        </div>
        <div style="display:flex; justify-content:space-between;">
            <span style="color:#94a3b8; font-size:0.8rem;">Activities</span>
            <span style="color:#fbbf24; font-weight:700; font-family:'JetBrains Mono',monospace; font-size:0.8rem;">{data_stats['n_classes']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar model badges
    st.markdown(f"""
    <div style="margin-top:0.5rem;">
        <div class="model-badge lstm" style="margin-bottom:0.5rem; width:100%; justify-content:center;">
            ⚡ LSTM — {comparison['LSTM']['test_accuracy']*100:.1f}%
        </div>
        <div class="model-badge gru" style="width:100%; justify-content:center;">
            ⚡ GRU — {comparison['GRU']['test_accuracy']*100:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# ░░░░░  PLOTLY THEME  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
# ═══════════════════════════════════════════════════════════════════
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Outfit, sans-serif", color="#f1f5f9"),
    margin=dict(l=40, r=40, t=60, b=40),
    xaxis=dict(gridcolor="rgba(99,102,241,0.08)", zerolinecolor="rgba(99,102,241,0.15)"),
    yaxis=dict(gridcolor="rgba(99,102,241,0.08)", zerolinecolor="rgba(99,102,241,0.15)"),
    hoverlabel=dict(bgcolor="#1e293b", font_size=13, font_family="Outfit"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(99,102,241,0.15)", borderwidth=1,
                font=dict(size=12)),
)

def update_plotly_layout(fig, **kwargs):
    import copy
    layout = copy.deepcopy(PLOTLY_LAYOUT)
    for k, v in kwargs.items():
        if k in layout and isinstance(layout[k], dict) and isinstance(v, dict):
            layout[k].update(v)
        else:
            layout[k] = v
    fig.update_layout(**layout)

COLORS = {
    "lstm": "#8b5cf6",
    "gru": "#06b6d4",
    "lstm_light": "#c4b5fd",
    "gru_light": "#67e8f9",
    "activities": ["#6366f1", "#06b6d4", "#f43f5e", "#10b981", "#f59e0b", "#ec4899"]
}


# ═══════════════════════════════════════════════════════════════════
#  HELPER: metric card HTML
# ═══════════════════════════════════════════════════════════════════
def metric_card(emoji, label, value, color_class="indigo"):
    return f"""
    <div class="metric-card {color_class}">
        <span class="metric-emoji">{emoji}</span>
        <div class="metric-value {color_class}">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """


# ═══════════════════════════════════════════════════════════════════
# ░░░░░░░░░░░░░░░░░░░  PAGE: OVERVIEW  ░░░░░░░░░░░░░░░░░░░░░░░░░
# ═══════════════════════════════════════════════════════════════════
if nav == "🏠 Overview":
    # ── Hero ──
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">Human Activity Recognition</div>
        <div class="hero-subtitle">
            Deep Learning Classification of Physical Activities from Smartphone Sensor Data
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Lottie animation ──
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        if lottie_walking:
            st_lottie(lottie_walking, height=180, key="hero_lottie")

    # ── Activity Icons ──
    icons_html = '<div class="activity-icons">'
    for act in ACTIVITY_LABELS:
        icons_html += f"""
        <div class="activity-icon">
            <span class="emoji">{ACTIVITY_EMOJIS.get(act, '🏃')}</span>
            <span class="name">{act.replace('_', ' ')}</span>
        </div>"""
    icons_html += '</div>'
    st.markdown(icons_html, unsafe_allow_html=True)

    # ── Data Flow Pipeline ──
    st.markdown("""
    <div class="data-flow">
        <div class="flow-node" style="background:rgba(99,102,241,0.15); border:1px solid rgba(99,102,241,0.3); color:#a78bfa;">
            📱 Sensor<br>Data
        </div>
        <span class="flow-arrow">→</span>
        <div class="flow-node" style="background:rgba(6,182,212,0.15); border:1px solid rgba(6,182,212,0.3); color:#22d3ee;">
            ⚡ Inertial<br>Signals
        </div>
        <span class="flow-arrow">→</span>
        <div class="flow-node" style="background:rgba(139,92,246,0.15); border:1px solid rgba(139,92,246,0.3); color:#c4b5fd;">
            🧠 LSTM<br>/ GRU
        </div>
        <span class="flow-arrow">→</span>
        <div class="flow-node" style="background:rgba(16,185,129,0.15); border:1px solid rgba(16,185,129,0.3); color:#34d399;">
            🎯 Activity<br>Class
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Key Metrics ──
    best_model = "LSTM" if comparison["LSTM"]["test_accuracy"] > comparison["GRU"]["test_accuracy"] else "GRU"
    best_acc = max(comparison["LSTM"]["test_accuracy"], comparison["GRU"]["test_accuracy"])
    total_samples = data_stats["train_samples"] + data_stats["test_samples"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card("🎯", "Best Accuracy", f"{best_acc*100:.1f}%", "indigo"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("🏆", "Best Model", best_model, "cyan"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("📊", "Total Samples", f"{total_samples:,}", "rose"), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card("🧪", "Activities", "6", "emerald"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Quick Comparison Bars ──
    st.markdown('<div class="section-title"><span class="icon">⚔️</span><span class="text">Quick Comparison</span></div>', unsafe_allow_html=True)

    fig = go.Figure()
    metrics_list = ["test_accuracy", "weighted_f1", "weighted_precision", "weighted_recall"]
    metric_labels = ["Accuracy", "F1 Score", "Precision", "Recall"]
    for model_name, color in [("LSTM", COLORS["lstm"]), ("GRU", COLORS["gru"])]:
        fig.add_trace(go.Bar(
            name=model_name,
            x=metric_labels,
            y=[comparison[model_name][m] for m in metrics_list],
            marker=dict(color=color, cornerradius=8,
                        line=dict(color="rgba(255,255,255,0.1)", width=1)),
            text=[f"{comparison[model_name][m]*100:.1f}%" for m in metrics_list],
            textposition="outside",
            textfont=dict(size=14, family="JetBrains Mono", color=color),
        ))
    update_plotly_layout(fig, barmode="group", height=400,
                         title=dict(text="Model Performance Metrics", font=dict(size=20)),
                         yaxis=dict(range=[0, 1.12], tickformat=".0%"),
                         bargap=0.3, bargroupgap=0.15)
    st.plotly_chart(fig, use_container_width=True)

    # ── Winner Banner ──
    diff = abs(comparison["LSTM"]["test_accuracy"] - comparison["GRU"]["test_accuracy"]) * 100
    st.markdown(f"""
    <div class="winner-banner">
        <div class="winner-text">
            🏆 {best_model} wins by {diff:.2f}% with {best_acc*100:.2f}% test accuracy!
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# ░░░░░░░░░░░░  PAGE: MODEL COMPARISON  ░░░░░░░░░░░░░░░░░░░░░░░░░
# ═══════════════════════════════════════════════════════════════════
elif nav == "📊 Model Comparison":
    st.markdown("""
    <div class="hero-container" style="padding:1.5rem 1rem 0.5rem;">
        <div class="hero-title" style="font-size:2.4rem;">📊 Model Comparison</div>
        <div class="hero-subtitle">LSTM vs GRU — Head to Head Deep Dive</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Rotating Ring Accuracies ──
    c1, c2 = st.columns(2)
    for col, model, color_class in [(c1, "LSTM", "indigo"), (c2, "GRU", "cyan")]:
        acc = comparison[model]["test_accuracy"]
        with col:
            st.markdown(f"""
            <div class="glass-panel" style="text-align:center; padding:2rem;">
                <div class="model-badge {'lstm' if model=='LSTM' else 'gru'}" style="margin:0 auto 1rem; justify-content:center; display:inline-flex;">
                    {'🔮' if model=='LSTM' else '⚡'} {model}
                </div>
                <div class="rotating-ring-container">
                    <div class="rotating-ring ring-1"></div>
                    <div class="rotating-ring ring-2"></div>
                    <div class="rotating-ring ring-3"></div>
                    <div class="ring-center {color_class}">{acc*100:.1f}%</div>
                </div>
                <div style="color:#94a3b8; font-size:0.8rem; text-transform:uppercase; letter-spacing:2px;">
                    Test Accuracy
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Detailed Metrics Table ──
    st.markdown('<div class="section-title"><span class="icon">📋</span><span class="text">Detailed Metrics</span></div>', unsafe_allow_html=True)

    metrics_data = []
    for m_name in ["LSTM", "GRU"]:
        res = lstm_results if m_name == "LSTM" else gru_results
        metrics_data.append({
            "Model": m_name,
            "Accuracy": f"{res['test_accuracy']*100:.2f}%",
            "Precision": f"{res['weighted_precision']*100:.2f}%",
            "Recall": f"{res['weighted_recall']*100:.2f}%",
            "F1 Score": f"{res['weighted_f1']*100:.2f}%",
            "Best Val Acc": f"{res['best_val_accuracy']*100:.2f}%",
        })
    df_metrics = pd.DataFrame(metrics_data)
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)

    # ── Radar Chart ──
    st.markdown('<div class="section-title"><span class="icon">🕸️</span><span class="text">Performance Radar</span></div>', unsafe_allow_html=True)

    categories = ["Accuracy", "Precision", "Recall", "F1", "Val Acc"]
    fig = go.Figure()
    for m_name, color in [("LSTM", COLORS["lstm"]), ("GRU", COLORS["gru"])]:
        res = lstm_results if m_name == "LSTM" else gru_results
        vals = [res["test_accuracy"], res["weighted_precision"], res["weighted_recall"],
                res["weighted_f1"], res["best_val_accuracy"]]
        vals.append(vals[0])  # close
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=categories + [categories[0]],
            fill="toself", name=m_name,
            fillcolor=f"rgba({','.join(str(int(color.lstrip('#')[i:i+2], 16)) for i in (0,2,4))},0.15)",
            line=dict(color=color, width=2),
            marker=dict(size=6, color=color),
        ))
    update_plotly_layout(fig, height=500, polar=dict(
        bgcolor="rgba(0,0,0,0)",
        radialaxis=dict(range=[0.8, 1.0], gridcolor="rgba(99,102,241,0.1)",
                        tickfont=dict(size=10, color="#94a3b8")),
        angularaxis=dict(gridcolor="rgba(99,102,241,0.1)",
                         tickfont=dict(size=13, color="#f1f5f9")),
    ))
    st.plotly_chart(fig, use_container_width=True)

    # ── Per-Class F1 Grouped Bar ──
    st.markdown('<div class="section-title"><span class="icon">🎯</span><span class="text">Per-Class F1 Scores</span></div>', unsafe_allow_html=True)

    fig = go.Figure()
    for m_name, color in [("LSTM", COLORS["lstm"]), ("GRU", COLORS["gru"])]:
        res = lstm_results if m_name == "LSTM" else gru_results
        f1s = [res["classification_report"][act]["f1-score"] for act in ACTIVITY_LABELS]
        fig.add_trace(go.Bar(
            name=m_name, x=ACTIVITY_LABELS, y=f1s,
            marker=dict(color=color, cornerradius=6),
            text=[f"{v:.3f}" for v in f1s], textposition="outside",
            textfont=dict(size=11, family="JetBrains Mono"),
        ))
    update_plotly_layout(fig, barmode="group", height=450,
                         yaxis=dict(range=[0, 1.15], title="F1 Score"),
                         xaxis=dict(title="Activity"))
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# ░░░░░░░░░░░  PAGE: TRAINING ANALYTICS  ░░░░░░░░░░░░░░░░░░░░░░░░
# ═══════════════════════════════════════════════════════════════════
elif nav == "📈 Training Analytics":
    st.markdown("""
    <div class="hero-container" style="padding:1.5rem 1rem 0.5rem;">
        <div class="hero-title" style="font-size:2.4rem;">📈 Training Analytics</div>
        <div class="hero-subtitle">Epoch-by-Epoch Training Dynamics</div>
    </div>
    """, unsafe_allow_html=True)

    if lottie_chart:
        col_l, col_c, col_r = st.columns([2, 1, 2])
        with col_c:
            st_lottie(lottie_chart, height=120, key="training_lottie")

    tab1, tab2, tab3, tab4 = st.tabs(["📉 Loss Curves", "📈 Accuracy Curves", "⏱️ Timing", "📊 Learning Rate"])

    with tab1:
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Training Loss", "Validation Loss"),
                            horizontal_spacing=0.1)
        for m_name, color in [("LSTM", COLORS["lstm"]), ("GRU", COLORS["gru"])]:
            h = (lstm_results if m_name == "LSTM" else gru_results)["history"]
            epochs = list(range(1, len(h["train_loss"]) + 1))
            fig.add_trace(go.Scatter(x=epochs, y=h["train_loss"], name=f"{m_name} Train",
                                     line=dict(color=color, width=2.5), mode="lines+markers",
                                     marker=dict(size=5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=epochs, y=h["val_loss"], name=f"{m_name} Val",
                                     line=dict(color=color, width=2.5, dash="dot"), mode="lines+markers",
                                     marker=dict(size=5)), row=1, col=2)
        update_plotly_layout(fig, height=450, showlegend=True)
        fig.update_xaxes(title_text="Epoch", gridcolor="rgba(99,102,241,0.08)")
        fig.update_yaxes(title_text="Loss", gridcolor="rgba(99,102,241,0.08)")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Training Accuracy", "Validation Accuracy"),
                            horizontal_spacing=0.1)
        for m_name, color in [("LSTM", COLORS["lstm"]), ("GRU", COLORS["gru"])]:
            h = (lstm_results if m_name == "LSTM" else gru_results)["history"]
            epochs = list(range(1, len(h["train_acc"]) + 1))
            fig.add_trace(go.Scatter(x=epochs, y=h["train_acc"], name=f"{m_name} Train",
                                     line=dict(color=color, width=2.5),
                                     fill="tozeroy", fillcolor=f"rgba({','.join(str(int(color.lstrip('#')[i:i+2], 16)) for i in (0,2,4))},0.08)",
                                     mode="lines"), row=1, col=1)
            fig.add_trace(go.Scatter(x=epochs, y=h["val_acc"], name=f"{m_name} Val",
                                     line=dict(color=color, width=2.5, dash="dot"),
                                     mode="lines"), row=1, col=2)
        update_plotly_layout(fig, height=450, showlegend=True)
        fig.update_xaxes(title_text="Epoch", gridcolor="rgba(99,102,241,0.08)")
        fig.update_yaxes(title_text="Accuracy", tickformat=".0%", gridcolor="rgba(99,102,241,0.08)")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = go.Figure()
        for m_name, color in [("LSTM", COLORS["lstm"]), ("GRU", COLORS["gru"])]:
            h = (lstm_results if m_name == "LSTM" else gru_results)["history"]
            epochs = list(range(1, len(h["epoch_time"]) + 1))
            fig.add_trace(go.Bar(x=epochs, y=h["epoch_time"], name=m_name,
                                 marker=dict(color=color, cornerradius=4)))
        update_plotly_layout(fig, height=400, barmode="group",
                             title="Epoch Training Time (seconds)",
                             xaxis_title="Epoch", yaxis_title="Time (s)")
        st.plotly_chart(fig, use_container_width=True)

        # Total time cards
        c1, c2 = st.columns(2)
        lstm_total = sum(lstm_results["history"]["epoch_time"])
        gru_total = sum(gru_results["history"]["epoch_time"])
        with c1:
            st.markdown(metric_card("⏱️", "LSTM Total Time", f"{lstm_total:.0f}s", "indigo"), unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card("⏱️", "GRU Total Time", f"{gru_total:.0f}s", "cyan"), unsafe_allow_html=True)

    with tab4:
        fig = go.Figure()
        for m_name, color in [("LSTM", COLORS["lstm"]), ("GRU", COLORS["gru"])]:
            h = (lstm_results if m_name == "LSTM" else gru_results)["history"]
            epochs = list(range(1, len(h["learning_rates"]) + 1))
            fig.add_trace(go.Scatter(x=epochs, y=h["learning_rates"], name=m_name,
                                     line=dict(color=color, width=2.5), mode="lines+markers",
                                     marker=dict(size=6)))
        update_plotly_layout(fig, height=400,
                             title="Learning Rate Schedule",
                             xaxis_title="Epoch", yaxis_title="Learning Rate",
                             yaxis_type="log")
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# ░░░░░░░░░  PAGE: CONFUSION MATRICES  ░░░░░░░░░░░░░░░░░░░░░░░░░░
# ═══════════════════════════════════════════════════════════════════
elif nav == "🔬 Confusion Matrices":
    st.markdown("""
    <div class="hero-container" style="padding:1.5rem 1rem 0.5rem;">
        <div class="hero-title" style="font-size:2.4rem;">🔬 Confusion Matrices</div>
        <div class="hero-subtitle">Detailed Classification Error Analysis</div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🔮 LSTM", "⚡ GRU"])

    for tab, m_name, color_scale in [(tab1, "LSTM", "Purples"), (tab2, "GRU", "Teal")]:
        with tab:
            res = lstm_results if m_name == "LSTM" else gru_results
            cm = np.array(res["confusion_matrix"])

            # Normalised
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

            short_labels = [a.replace("WALKING_", "W_") for a in ACTIVITY_LABELS]

            fig = go.Figure(data=go.Heatmap(
                z=cm_norm,
                x=short_labels, y=short_labels,
                colorscale=color_scale,
                text=cm, texttemplate="%{text}",
                textfont=dict(size=14, family="JetBrains Mono"),
                hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{text}<br>Rate: %{z:.2%}<extra></extra>",
                showscale=True,
                colorbar=dict(title="Rate", tickformat=".0%"),
            ))
            update_plotly_layout(fig, height=550,
                                 title=dict(text=f"{m_name} Confusion Matrix", font=dict(size=20)),
                                 xaxis_title="Predicted", yaxis_title="Actual",
                                 yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

            # Per-class metrics
            st.markdown(f'<div class="section-title"><span class="icon">📋</span><span class="text">{m_name} Per-Class Report</span></div>', unsafe_allow_html=True)
            report_data = []
            for act in ACTIVITY_LABELS:
                r = res["classification_report"][act]
                report_data.append({
                    "Activity": f"{ACTIVITY_EMOJIS.get(act, '🏃')} {act}",
                    "Precision": f"{r['precision']:.4f}",
                    "Recall": f"{r['recall']:.4f}",
                    "F1-Score": f"{r['f1-score']:.4f}",
                    "Support": int(r["support"]),
                })
            st.dataframe(pd.DataFrame(report_data), use_container_width=True, hide_index=True)

    # ── Misclassification Heatmap (difference) ──
    st.markdown('<div class="section-title"><span class="icon">🔍</span><span class="text">Misclassification Comparison</span></div>', unsafe_allow_html=True)

    cm_lstm = np.array(lstm_results["confusion_matrix"]).astype(float)
    cm_gru  = np.array(gru_results["confusion_matrix"]).astype(float)
    cm_lstm_n = cm_lstm / cm_lstm.sum(axis=1, keepdims=True)
    cm_gru_n  = cm_gru  / cm_gru.sum(axis=1, keepdims=True)
    diff = cm_gru_n - cm_lstm_n  # positive = GRU better at predicting this

    fig = go.Figure(data=go.Heatmap(
        z=diff, x=short_labels, y=short_labels,
        colorscale="RdBu", zmid=0,
        text=np.round(diff * 100, 1),
        texttemplate="%{text}%",
        textfont=dict(size=12, family="JetBrains Mono"),
        hovertemplate="True: %{y}<br>Pred: %{x}<br>GRU − LSTM: %{z:.2%}<extra></extra>",
        colorbar=dict(title="GRU − LSTM"),
    ))
    update_plotly_layout(fig, height=480,
                         title=dict(text="GRU − LSTM Difference (Blue = GRU better)", font=dict(size=18)),
                         xaxis_title="Predicted", yaxis_title="Actual",
                         yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# ░░░░░░░░░░░  PAGE: SENSOR SIGNALS  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░
# ═══════════════════════════════════════════════════════════════════
elif nav == "🌊 Sensor Signals":
    st.markdown("""
    <div class="hero-container" style="padding:1.5rem 1rem 0.5rem;">
        <div class="hero-title" style="font-size:2.4rem;">🌊 Sensor Signals Explorer</div>
        <div class="hero-subtitle">Raw Accelerometer & Gyroscope Time-Series Visualisation</div>
    </div>
    """, unsafe_allow_html=True)

    sample_signals = data_stats.get("sample_signals", {})

    if sample_signals:
        selected_activity = st.selectbox(
            "Select Activity",
            ACTIVITY_LABELS,
            format_func=lambda x: f"{ACTIVITY_EMOJIS.get(x, '🏃')} {x}"
        )

        sig = sample_signals[selected_activity]

        # 3×3 grid of all 9 channels
        channel_groups = [
            ("Body Accelerometer", ["body_acc_x", "body_acc_y", "body_acc_z"], "#6366f1"),
            ("Body Gyroscope", ["body_gyro_x", "body_gyro_y", "body_gyro_z"], "#06b6d4"),
            ("Total Accelerometer", ["total_acc_x", "total_acc_y", "total_acc_z"], "#f43f5e"),
        ]

        for group_name, channels, base_color in channel_groups:
            st.markdown(f'<div class="section-title"><span class="icon">📡</span><span class="text">{group_name}</span></div>', unsafe_allow_html=True)

            fig = make_subplots(rows=1, cols=3, subplot_titles=[c.split("_")[-1].upper() + " Axis" for c in channels],
                                horizontal_spacing=0.06)
            for i, ch in enumerate(channels):
                time_axis = list(range(len(sig[ch])))
                fig.add_trace(go.Scatter(
                    x=time_axis, y=sig[ch], mode="lines",
                    line=dict(color=base_color, width=1.5),
                    fill="tozeroy",
                    fillcolor=f"rgba({','.join(str(int(base_color.lstrip('#')[j:j+2], 16)) for j in (0,2,4))},0.1)",
                    name=ch, showlegend=False,
                ), row=1, col=i+1)
            update_plotly_layout(fig, height=280,
                                 margin=dict(l=30, r=20, t=40, b=30))
            fig.update_xaxes(title_text="Time Step", gridcolor="rgba(99,102,241,0.06)")
            fig.update_yaxes(gridcolor="rgba(99,102,241,0.06)")
            st.plotly_chart(fig, use_container_width=True)

        # ── Overlaid 3D-like comparison ──
        st.markdown('<div class="section-title"><span class="icon">🔀</span><span class="text">All Activities Overlay — Body Acc X</span></div>', unsafe_allow_html=True)

        fig = go.Figure()
        for act_idx, act in enumerate(ACTIVITY_LABELS):
            s = sample_signals[act]["body_acc_x"]
            fig.add_trace(go.Scatter(
                x=list(range(len(s))), y=s, mode="lines",
                name=f"{ACTIVITY_EMOJIS.get(act, '')} {act}",
                line=dict(color=COLORS["activities"][act_idx], width=1.8),
                opacity=0.85,
            ))
        update_plotly_layout(fig, height=400,
                             title="Body Accelerometer X-axis — All Activities",
                             xaxis_title="Time Step", yaxis_title="Amplitude")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Sample signals not available in data_stats.json")


# ═══════════════════════════════════════════════════════════════════
# ░░░░░░░░░░░  PAGE: PER-CLASS ANALYSIS  ░░░░░░░░░░░░░░░░░░░░░░░░
# ═══════════════════════════════════════════════════════════════════
elif nav == "🎯 Per-Class Analysis":
    st.markdown("""
    <div class="hero-container" style="padding:1.5rem 1rem 0.5rem;">
        <div class="hero-title" style="font-size:2.4rem;">🎯 Per-Class Analysis</div>
        <div class="hero-subtitle">Deep Dive into Individual Activity Performance</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Class Distribution ──
    st.markdown('<div class="section-title"><span class="icon">📊</span><span class="text">Class Distribution</span></div>', unsafe_allow_html=True)

    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}]],
                        subplot_titles=("Training Set", "Test Set"))

    train_vals = list(data_stats["train_class_dist"].values())
    test_vals = list(data_stats["test_class_dist"].values())

    fig.add_trace(go.Pie(
        labels=ACTIVITY_LABELS, values=train_vals,
        marker=dict(colors=COLORS["activities"],
                    line=dict(color="#0a0e1a", width=2)),
        textinfo="percent+label", textfont=dict(size=11),
        hole=0.45, pull=[0.02]*6,
    ), row=1, col=1)

    fig.add_trace(go.Pie(
        labels=ACTIVITY_LABELS, values=test_vals,
        marker=dict(colors=COLORS["activities"],
                    line=dict(color="#0a0e1a", width=2)),
        textinfo="percent+label", textfont=dict(size=11),
        hole=0.45, pull=[0.02]*6,
    ), row=1, col=2)

    update_plotly_layout(fig, height=450, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # ── Per-class metric comparison (grouped) ──
    st.markdown('<div class="section-title"><span class="icon">📈</span><span class="text">Per-Activity Metrics Comparison</span></div>', unsafe_allow_html=True)

    metric_choice = st.selectbox("Select Metric", ["precision", "recall", "f1-score"])

    fig = go.Figure()
    for m_name, color in [("LSTM", COLORS["lstm"]), ("GRU", COLORS["gru"])]:
        res = lstm_results if m_name == "LSTM" else gru_results
        vals = [res["classification_report"][act][metric_choice] for act in ACTIVITY_LABELS]
        fig.add_trace(go.Bar(
            name=m_name, x=[f"{ACTIVITY_EMOJIS.get(a,'')} {a}" for a in ACTIVITY_LABELS],
            y=vals, marker=dict(color=color, cornerradius=6),
            text=[f"{v:.3f}" for v in vals], textposition="outside",
            textfont=dict(size=11, family="JetBrains Mono"),
        ))
    update_plotly_layout(fig, barmode="group", height=450,
                         yaxis=dict(range=[0, 1.12], title=metric_choice.title()),
                         xaxis_title="Activity")
    st.plotly_chart(fig, use_container_width=True)

    # ── Prediction Confidence Distribution ──
    st.markdown('<div class="section-title"><span class="icon">🎲</span><span class="text">Prediction Confidence Distribution</span></div>', unsafe_allow_html=True)

    model_sel = st.selectbox("Select Model", ["LSTM", "GRU"], key="conf_model")
    res = lstm_results if model_sel == "LSTM" else gru_results
    probs = np.array(res["prediction_probabilities"])
    true_labels = np.array(res["true_labels"])
    preds = np.array(res["predictions"])

    # Max confidence per prediction
    max_confs = probs.max(axis=1)
    correct_mask = preds == true_labels

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=max_confs[correct_mask], name="Correct",
        marker_color="#10b981", opacity=0.7, nbinsx=40,
    ))
    fig.add_trace(go.Histogram(
        x=max_confs[~correct_mask], name="Incorrect",
        marker_color="#f43f5e", opacity=0.7, nbinsx=40,
    ))
    update_plotly_layout(fig, barmode="overlay", height=400,
                         title=f"{model_sel} — Prediction Confidence Distribution",
                         xaxis_title="Max Softmax Probability", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# ░░░░░░░░░░░  PAGE: ARCHITECTURE  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
# ═══════════════════════════════════════════════════════════════════
elif nav == "⚙️ Architecture":
    st.markdown("""
    <div class="hero-container" style="padding:1.5rem 1rem 0.5rem;">
        <div class="hero-title" style="font-size:2.4rem;">⚙️ Model Architecture</div>
        <div class="hero-subtitle">Neural Network Design & Hyperparameters</div>
    </div>
    """, unsafe_allow_html=True)

    if lottie_brain:
        col_l, col_c, col_r = st.columns([2, 1, 2])
        with col_c:
            st_lottie(lottie_brain, height=150, key="arch_lottie")

    # ── Architecture Diagrams ──
    c1, c2 = st.columns(2)

    for col, model_name, color, gradient in [
        (c1, "LSTM", "#8b5cf6", "linear-gradient(135deg, rgba(99,102,241,0.1), rgba(139,92,246,0.05))"),
        (c2, "GRU", "#06b6d4", "linear-gradient(135deg, rgba(6,182,212,0.1), rgba(34,211,238,0.05))")
    ]:
        with col:
            cell_type = "LSTM Cell<br><span style='font-size:0.65rem;color:#94a3b8;'>Forget + Input + Output Gates</span>" if model_name == "LSTM" else "GRU Cell<br><span style='font-size:0.65rem;color:#94a3b8;'>Reset + Update Gates</span>"
            st.markdown(f"""
            <div class="glass-panel" style="text-align:center; background:{gradient};">
                <div class="model-badge {'lstm' if model_name=='LSTM' else 'gru'}" style="margin:0 auto 1.5rem; display:inline-flex; justify-content:center;">
                    {'🔮' if model_name=='LSTM' else '⚡'} {model_name} Architecture
                </div>
                <div style="display:flex; flex-direction:column; align-items:center; gap:0.6rem;">
                    <div style="padding:0.6rem 1.5rem; border-radius:12px; background:rgba(99,102,241,0.1); border:1px solid rgba(99,102,241,0.2); font-size:0.85rem;">
                        📥 Input: (Batch, 128, 9)
                    </div>
                    <div style="color:{color}; font-size:1.2rem;">↓</div>
                    <div style="padding:0.8rem 1.5rem; border-radius:12px; background:rgba({','.join(str(int(color.lstrip('#')[i:i+2], 16)) for i in (0,2,4))},0.15); border:1px solid rgba({','.join(str(int(color.lstrip('#')[i:i+2], 16)) for i in (0,2,4))},0.3); font-size:0.85rem;">
                        {cell_type}
                    </div>
                    <div style="color:{color}; font-size:0.7rem;">× 2 Layers, Hidden=128</div>
                    <div style="color:{color}; font-size:1.2rem;">↓</div>
                    <div style="padding:0.5rem 1.5rem; border-radius:12px; background:rgba(244,63,94,0.1); border:1px solid rgba(244,63,94,0.2); font-size:0.85rem;">
                        Last Time-Step → BatchNorm
                    </div>
                    <div style="color:{color}; font-size:1.2rem;">↓</div>
                    <div style="padding:0.5rem 1.5rem; border-radius:12px; background:rgba(245,158,11,0.1); border:1px solid rgba(245,158,11,0.2); font-size:0.85rem;">
                        Dropout (0.3)
                    </div>
                    <div style="color:{color}; font-size:1.2rem;">↓</div>
                    <div style="padding:0.6rem 1.5rem; border-radius:12px; background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.2); font-size:0.85rem;">
                        📤 Dense → 6 Classes
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Hyperparameters Table ──
    st.markdown('<div class="section-title"><span class="icon">🔧</span><span class="text">Hyperparameters</span></div>', unsafe_allow_html=True)

    hp = lstm_results["hyperparameters"]
    hp_html = """
    <table class="hp-table">
        <thead>
            <tr><th>Parameter</th><th>LSTM</th><th>GRU</th></tr>
        </thead>
        <tbody>
    """
    params = [
        ("Hidden Size", "hidden_size"), ("Num Layers", "num_layers"),
        ("Dropout", "dropout"), ("Learning Rate", "learning_rate"),
        ("Batch Size", "batch_size"), ("Epochs", "epochs"),
        ("Optimizer", "optimizer"), ("Scheduler", "scheduler"),
    ]
    for label, key in params:
        v1 = lstm_results["hyperparameters"][key]
        v2 = gru_results["hyperparameters"][key]
        hp_html += f"<tr><td>{label}</td><td>{v1}</td><td>{v2}</td></tr>"
    hp_html += "</tbody></table>"
    st.markdown(hp_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Model Size Comparison ──
    st.markdown('<div class="section-title"><span class="icon">📦</span><span class="text">Model Complexity</span></div>', unsafe_allow_html=True)

    # Calculate parameter counts
    def count_params(hidden, layers, input_size, n_classes, model_type):
        if model_type == "LSTM":
            # LSTM: 4 * (input*hidden + hidden*hidden + hidden) per layer
            rnn_params = 4 * (input_size * hidden + hidden * hidden + 2 * hidden)  # layer 1
            for _ in range(1, layers):
                rnn_params += 4 * (hidden * hidden + hidden * hidden + 2 * hidden)
        else:
            # GRU: 3 * (input*hidden + hidden*hidden + hidden) per layer
            rnn_params = 3 * (input_size * hidden + hidden * hidden + 2 * hidden)
            for _ in range(1, layers):
                rnn_params += 3 * (hidden * hidden + hidden * hidden + 2 * hidden)
        bn_params = 2 * hidden
        fc_params = hidden * n_classes + n_classes
        return rnn_params + bn_params + fc_params

    lstm_params = count_params(128, 2, 9, 6, "LSTM")
    gru_params  = count_params(128, 2, 9, 6, "GRU")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(metric_card("🔮", "LSTM Parameters", f"{lstm_params:,}", "indigo"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("⚡", "GRU Parameters", f"{gru_params:,}", "cyan"), unsafe_allow_html=True)
    with c3:
        ratio = lstm_params / gru_params if gru_params > 0 else 0
        st.markdown(metric_card("📐", "LSTM/GRU Ratio", f"{ratio:.2f}×", "rose"), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# ░░░░░░░░░░░░░░░░░░  FOOTER  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
# ═══════════════════════════════════════════════════════════════════
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; padding:2rem 0; border-top:1px solid rgba(99,102,241,0.1);">
    <div style="font-size:0.8rem; color:#64748b; letter-spacing:1px;">
        Built with 💜 using PyTorch & Streamlit
    </div>
    <div style="font-size:0.7rem; color:#475569; margin-top:0.35rem;">
        UCI HAR Dataset • LSTM & GRU Deep Learning Models • Interactive Analytics Dashboard
    </div>
    <div style="margin-top:0.75rem;">
        <span class="pulse-dot green"></span>
        <span style="color:#64748b; font-size:0.75rem; margin-left:0.5rem;">System Online</span>
    </div>
</div>
""", unsafe_allow_html=True)
