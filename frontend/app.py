"""
frontend/app.py

Streamlit frontend for the Source Credibility Analyzer — HUD/scanner-styled UI.
Sends article text to the FastAPI backend and displays the prediction.

Run with (from the project root, in a SEPARATE terminal from the backend):
    streamlit run frontend/app.py
"""

import requests
import streamlit as st

API_URL = "http://localhost:8000/predict"

st.set_page_config(
    page_title="CREDSCAN // Source Analyzer",
    page_icon="◈",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Design tokens
#   Background : #05070d  (near-black, deep space navy)
#   Panel      : #0c1120
#   Primary    : #00e5ff  (electric cyan  — REAL / structural accent)
#   Secondary  : #ff2e63  (signal magenta — FAKE)
#   Muted      : #5b6478
#   Display    : Orbitron (headers, all-caps, wide tracking)
#   Data/mono  : JetBrains Mono (labels, stats, chips)
#   Body       : Inter
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;800&family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500&display=swap');

.stApp {
    background:
        repeating-linear-gradient(0deg, rgba(255,255,255,0.012) 0px, rgba(255,255,255,0.012) 1px, transparent 1px, transparent 3px),
        radial-gradient(ellipse at top, #0a0f1c 0%, #05070d 60%);
}
.block-container { 
    padding-top: 3rem;  /* Increase this from 2.2rem to ensure nothing is clipped */
    max-width: 760px; 
}

/* ---------- Header ---------- */
.hud-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.25em;
    color: #00e5ff;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
    opacity: 0.85;
}
.hud-title {
    font-family: 'Orbitron', sans-serif;
    font-weight: 800;
    font-size: 2.0rem;
    letter-spacing: 0.04em;
    color: #eef2f7;
    text-shadow: 0 0 18px rgba(0,229,255,0.25);
    margin-bottom: 0.3rem;
}
.hud-title span { color: #00e5ff; }
.hud-subtitle {
    font-family: 'Inter', sans-serif;
    color: #7d8598;
    font-size: 0.92rem;
    line-height: 1.55;
    margin-bottom: 1.1rem;
    max-width: 560px;
}
.stat-strip {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 1.6rem;
    flex-wrap: wrap;
}
.stat-chip {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.04em;
    color: #00e5ff;
    background: rgba(0,229,255,0.06);
    border: 1px solid rgba(0,229,255,0.3);
    padding: 4px 11px;
    border-radius: 2px;
}

/* ---------- HUD panel (native bordered container) ---------- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    position: relative;
    border: 1px solid rgba(0,229,255,0.18) !important;
    background: rgba(12,17,32,0.55);
    margin-bottom: 1.3rem;
    overflow: visible !important;
}
div[data-testid="stVerticalBlockBorderWrapper"]::before,
div[data-testid="stVerticalBlockBorderWrapper"]::after {
    content: "";
    position: absolute;
    width: 14px; height: 14px;
    border: 2px solid #00e5ff;
    z-index: 2;
}
div[data-testid="stVerticalBlockBorderWrapper"]::before { top: -1px; left: -1px; border-right: none; border-bottom: none; }
div[data-testid="stVerticalBlockBorderWrapper"]::after { bottom: -1px; right: -1px; border-left: none; border-top: none; }
.panel-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    color: #4a90a4;
    text-transform: uppercase;
    margin-bottom: 0.7rem;
}
.panel-label b { color: #00e5ff; }

/* ---------- Inputs ---------- */
div[data-testid="stTextArea"] textarea {
    background: #080b14 !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 3px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
    color: #d3d9e6 !important;
}
div[data-testid="stTextArea"] textarea:focus {
    border-color: #00e5ff !important;
    box-shadow: 0 0 0 1px #00e5ff, 0 0 14px rgba(0,229,255,0.25) !important;
}
div.stButton > button {
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.08em;
    font-size: 0.78rem;
    border-radius: 2px;
    text-transform: uppercase;
}
div.stButton > button[kind="primary"] {
    background: #00e5ff;
    color: #05070d;
    border: none;
    font-weight: 700;
    box-shadow: 0 0 20px rgba(0,229,255,0.35);
}
div.stButton > button[kind="primary"]:hover { box-shadow: 0 0 28px rgba(0,229,255,0.55); }
div.stButton > button[kind="secondary"] {
    background: transparent;
    color: #7d8598;
    border: 1px solid rgba(255,255,255,0.15);
}

/* ---------- Result ---------- */
.verdict-row {
    display: flex;
    align-items: center;
    gap: 1.6rem;
}
.gauge-wrap { position: relative; width: 108px; height: 108px; flex-shrink: 0; }
.gauge-ring { width: 108px; height: 108px; border-radius: 50%; }
.gauge-hole {
    position: absolute; top: 10px; left: 10px;
    width: 88px; height: 88px; border-radius: 50%;
    background: #0a0e1a;
    display: flex; align-items: center; justify-content: center;
    flex-direction: column;
}
.gauge-pct {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700; font-size: 1.15rem;
}
.gauge-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.55rem; color: #5b6478; letter-spacing: 0.1em;
}
.verdict-label {
    font-family: 'Orbitron', sans-serif;
    font-weight: 800;
    font-size: 1.55rem;
    letter-spacing: 0.03em;
}
.verdict-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #7d8598;
    margin-top: 0.2rem;
}
.real-color { color: #00e5ff; }
.fake-color { color: #ff2e63; }

.word-chip {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.76rem;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 2px;
    padding: 3px 9px;
    margin: 3px 6px 3px 0;
    color: #c3cadb;
}

.footer-note {
    font-family: 'JetBrains Mono', monospace;
    color: #4a5064;
    font-size: 0.72rem;
    line-height: 1.6;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(255,255,255,0.06);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="hud-eyebrow">// SYSTEM: TEXT PATTERN ANALYSIS</div>', unsafe_allow_html=True)
st.markdown('<div class="hud-title">CRED<span>SCAN</span></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hud-subtitle">Paste a news article for stylistic pattern analysis. '
    'The classifier compares its language against signatures learned from 44,000 '
    'labeled real and fake articles.</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="stat-strip">'
    '<span class="stat-chip">ACCURACY 99.47%</span>'
    '<span class="stat-chip">MODEL: LOGISTIC REGRESSION</span>'
    '<span class="stat-chip">FEATURES: TF-IDF · 5000</span>'
    '</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------------
# Example articles
# ---------------------------------------------------------------------------
EXAMPLES = {
    "▸ LOAD SAMPLE: REAL-STYLE": (
        "Federal Reserve Holds Interest Rates Steady Amid Inflation Concerns\n\n"
        "WASHINGTON (Reuters) - The Federal Reserve left its benchmark interest rate "
        "unchanged on Wednesday, citing ongoing concerns about inflation and labor "
        "market stability. In a statement released after the two-day policy meeting, "
        "the central bank said it would continue to monitor economic data before "
        "making further adjustments. Analysts had largely expected the decision, "
        "according to a survey conducted prior to the meeting."
    ),
    "▸ LOAD SAMPLE: FAKE-STYLE": (
        "SHOCKING: Scientists Discover Government Is Hiding Miracle Cure That Big "
        "Pharma Doesn't Want You To Know\n\n"
        "You won't believe what researchers have uncovered! According to anonymous "
        "insiders, a simple household ingredient has been PROVEN to cure multiple "
        "diseases, but powerful pharmaceutical companies are spending billions to "
        "keep this secret buried. Share this before it gets taken down!!!"
    ),
}

if "article_text" not in st.session_state:
    st.session_state.article_text = ""
if st.session_state.get("_do_clear", False):
    st.session_state.article_text = ""
    st.session_state._do_clear = False

with st.container(border=True):
    st.markdown('<div class="panel-label"><b>01</b> // INPUT SOURCE TEXT</div>', unsafe_allow_html=True)

    ex_cols = st.columns(2)
    for col, (label, text) in zip(ex_cols, EXAMPLES.items()):
        if col.button(label, use_container_width=True):
            st.session_state.article_text = text

    article_text = st.text_area(
        "Article text",
        height=200,
        placeholder="> paste article headline / body text here...",
        key="article_text",
        label_visibility="collapsed",
    )

    col1, col2 = st.columns([2, 1])
    analyze_clicked = col1.button("▶ RUN ANALYSIS", type="primary", use_container_width=True)
    clear_clicked = col2.button("CLEAR", use_container_width=True)

if clear_clicked:
    st.session_state._do_clear = True
    st.rerun()

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if analyze_clicked:
    if not article_text.strip():
        st.warning("Input required — paste article text before running analysis.")
    else:
        with st.spinner("SCANNING TEXT PATTERNS..."):
            try:
                response = requests.post(API_URL, json={"text": article_text}, timeout=10)
                response.raise_for_status()
                result = response.json()

                label = result["label"]
                confidence = result["confidence"]
                top_words = result["top_words"]

                color = "#00e5ff" if label == "REAL" else "#ff2e63"
                color_class = "real-color" if label == "REAL" else "fake-color"
                angle = confidence * 360

                with st.container(border=True):
                    st.markdown(f'''
                    <div class="panel-label"><b>02</b> // ANALYSIS RESULT</div>
                    <div class="verdict-row">
                        <div class="gauge-wrap">
                            <div class="gauge-ring" style="background: conic-gradient(from -90deg, {color} 0deg, {color} {angle}deg, rgba(255,255,255,0.08) {angle}deg 360deg);"></div>
                            <div class="gauge-hole">
                                <div class="gauge-pct" style="color:{color};">{confidence:.0%}</div>
                                <div class="gauge-tag">CONFIDENCE</div>
                            </div>
                        </div>
                        <div>
                            <div class="verdict-label {color_class}">{label}</div>
                            <div class="verdict-sub">CLASSIFICATION COMPLETE</div>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)

                if top_words:
                    chips = "".join(f'<span class="word-chip">{w}</span>' for w in top_words)
                    with st.container(border=True):
                        st.markdown(f'''
                        <div class="panel-label"><b>03</b> // SIGNAL TOKENS DETECTED</div>
                        {chips}
                        ''', unsafe_allow_html=True)

            except requests.exceptions.ConnectionError:
                st.error(
                    "Backend unreachable. Start it with:\n\n"
                    "`uvicorn backend.main:app --reload --port 8000`"
                )
            except Exception as e:
                st.error(f"Error: {e}")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="footer-note">MODEL: TF-IDF + LOGISTIC REGRESSION · TRAINED ON FAKE-AND-REAL-NEWS-DATASET (KAGGLE)<br>'
    'NOTE: THIS SYSTEM DETECTS STYLISTIC PATTERNS, NOT FACTUAL ACCURACY. NOT A FACT-CHECKER.</div>',
    unsafe_allow_html=True
)
