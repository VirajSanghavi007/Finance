"""
AlgoTrade-X Dashboard — Bloomberg-inspired terminal UI.
Run: streamlit run src/dashboard/app.py
"""
from __future__ import annotations

import streamlit as st

from src.dashboard.theme import BASE, SURFACE, BORDER, AMBER, TEXT

st.set_page_config(
    page_title="AlgoTrade-X",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ──────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <style>
    html, body, [data-testid="stApp"] {{
        background-color: {BASE} !important;
        color: {TEXT};
        font-family: Consolas, "Courier New", monospace;
    }}
    [data-testid="stSidebar"] {{
        background-color: {SURFACE} !important;
        border-right: 1px solid {BORDER};
    }}
    h1, h2, h3 {{
        color: {AMBER} !important;
        font-family: Consolas, monospace !important;
    }}
    .stMetricLabel {{ color: {AMBER} !important; font-family: Consolas !important; }}
    .stMetricValue {{ color: #E0E0E0 !important; font-family: Consolas !important; }}
    [data-testid="stTab"] {{ color: {AMBER}; }}
    div[data-testid="stDataFrame"] {{ background-color: {SURFACE}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f'<div style="color:{AMBER};font-size:18px;font-weight:700;'
        f'font-family:Consolas;letter-spacing:2px;padding:10px 0">ALGOTRADE-X</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div style="color:#757575;font-size:10px">v0.1.0 | Bloomberg-Style Terminal</div>',
                unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["Overview", "Signals", "Backtest", "Risk", "Models", "Explainability", "Settings"],
        label_visibility="collapsed",
    )

# ── Page Routing ────────────────────────────────────────────────────────────
import importlib, sys
_PAGE_MAP = {
    "Overview":      "src.dashboard.pages.01_overview",
    "Signals":       "src.dashboard.pages.02_signals",
    "Backtest":      "src.dashboard.pages.03_backtest",
    "Risk":          "src.dashboard.pages.04_risk",
    "Models":        "src.dashboard.pages.05_models",
    "Explainability":"src.dashboard.pages.06_explainability",
    "Settings":      "src.dashboard.pages.07_settings",
}

module_name = _PAGE_MAP[page]
if module_name not in sys.modules:
    import importlib.util
    from pathlib import Path
    # Map module name to file path
    file_map = {
        "src.dashboard.pages.01_overview":      Path(__file__).parent / "pages" / "01_overview.py",
        "src.dashboard.pages.02_signals":       Path(__file__).parent / "pages" / "02_signals.py",
        "src.dashboard.pages.03_backtest":      Path(__file__).parent / "pages" / "03_backtest.py",
        "src.dashboard.pages.04_risk":          Path(__file__).parent / "pages" / "04_risk.py",
        "src.dashboard.pages.05_models":        Path(__file__).parent / "pages" / "05_models.py",
        "src.dashboard.pages.06_explainability":Path(__file__).parent / "pages" / "06_explainability.py",
        "src.dashboard.pages.07_settings":      Path(__file__).parent / "pages" / "07_settings.py",
    }
    spec = importlib.util.spec_from_file_location(module_name, file_map[module_name])
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
else:
    mod = sys.modules[module_name]

mod.render()
