"""Signal card component -- shows ticker signal + confidence + top features."""
from __future__ import annotations

import streamlit as st
from src.dashboard.theme import AMBER, GREEN, RED, SURFACE, BORDER, MUTED, SIGNAL_COLORS


_SIGNAL_LABELS = {1: "LONG ▲", 0: "FLAT ●", -1: "SHORT ▼"}
_REGIME_LABELS = {0: "LOW-VOL", 1: "MID-VOL", 2: "HIGH-VOL"}
_REGIME_COLORS = {0: GREEN, 1: AMBER, 2: RED}


def render_signal_card(
    ticker: str,
    signal: int,
    confidence: float,
    regime: int,
    top_features: dict[str, float],
    price: float | None = None,
    price_change_pct: float | None = None,
) -> None:
    sig_color   = SIGNAL_COLORS.get(signal, AMBER)
    sig_label   = _SIGNAL_LABELS.get(signal, "FLAT ●")
    reg_color   = _REGIME_COLORS.get(regime, AMBER)
    reg_label   = _REGIME_LABELS.get(regime, "MID-VOL")

    price_html = ""
    if price is not None:
        chg_color = GREEN if (price_change_pct or 0) >= 0 else RED
        price_html = (
            f'<div style="font-size:12px;color:{chg_color};font-family:Consolas">'
            f'${price:,.2f} '
            f'{"+" if (price_change_pct or 0) >= 0 else ""}'
            f'{(price_change_pct or 0):.2f}%</div>'
        )

    features_html = ""
    if top_features:
        rows = "".join(
            f'<tr><td style="color:{MUTED};font-size:9px">{k[:18]}</td>'
            f'<td style="color:{"" if v >= 0 else RED};font-size:9px;text-align:right">{v:+.3f}</td></tr>'
            for k, v in list(top_features.items())[:5]
        )
        features_html = f'<table style="width:100%;margin-top:6px">{rows}</table>'

    st.markdown(
        f"""
        <div style="background:{SURFACE};border:1px solid {BORDER};
                    border-left:3px solid {sig_color};padding:10px;
                    border-radius:4px;margin-bottom:8px">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <span style="color:{AMBER};font-size:14px;font-weight:700;
                             font-family:Consolas">{ticker}</span>
                <span style="color:{sig_color};font-size:13px;font-weight:700;
                             font-family:Consolas">{sig_label}</span>
            </div>
            {price_html}
            <div style="margin-top:4px;font-family:Consolas;font-size:10px">
                <span style="color:{MUTED}">CONF: </span>
                <span style="color:{AMBER}">{confidence:.0%}</span>
                &nbsp;
                <span style="color:{MUTED}">REGIME: </span>
                <span style="color:{reg_color}">{reg_label}</span>
            </div>
            {features_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
