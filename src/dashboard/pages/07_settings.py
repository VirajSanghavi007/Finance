"""Settings page -- configuration viewer."""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_root = _Path(__file__).resolve().parents[3]
if str(_root) not in _sys.path:
    _sys.path.insert(0, str(_root))



import streamlit as st
from src.dashboard.theme import AMBER, SURFACE, BORDER


def render():
    st.markdown(
        f'<h2 style="color:{AMBER};font-family:Consolas">SETTINGS</h2>',
        unsafe_allow_html=True,
    )

    from src.config.settings import Settings
    settings = Settings()

    st.markdown(f'<div style="color:{AMBER}">API Keys Status</div>', unsafe_allow_html=True)
    sources = settings.available_sources()
    for source in ["alpaca", "newsapi", "fred", "sec"]:
        has_key = source in sources
        color = "#00E676" if has_key else "#FF1744"
        label = "✓ Configured" if has_key else "✗ Not configured"
        st.markdown(
            f'<div style="font-family:Consolas;font-size:12px">'
            f'<span style="color:{AMBER}">{source.upper()}</span> '
            f'<span style="color:{color}">{label}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(f'<div style="color:{AMBER}">Risk Limits</div>', unsafe_allow_html=True)
    from src.config import constants as C
    limits = {
        "MAX_SINGLE_POSITION":   f"{C.MAX_SINGLE_POSITION:.0%}",
        "MAX_PORTFOLIO_DRAWDOWN": f"{C.MAX_PORTFOLIO_DRAWDOWN:.0%}",
        "MAX_DAILY_LOSS":         f"{C.MAX_DAILY_LOSS:.0%}",
        "MAX_GROSS_EXPOSURE":     f"{C.MAX_GROSS_EXPOSURE:.1f}x",
        "VOL_TARGET":             f"{C.VOL_TARGET:.0%}",
    }
    for k, v in limits.items():
        st.markdown(
            f'<div style="font-family:Consolas;font-size:12px">'
            f'<span style="color:#757575">{k}</span> = '
            f'<span style="color:#E0E0E0">{v}</span></div>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    render()
