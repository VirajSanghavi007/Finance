"""Top-of-page metrics bar — key portfolio stats displayed Bloomberg-style."""
from __future__ import annotations

import streamlit as st
from src.dashboard.theme import AMBER, GREEN, RED, SURFACE, BORDER, MUTED


def render_metrics_bar(
    portfolio_value: float,
    daily_pnl: float,
    daily_pnl_pct: float,
    sharpe: float,
    drawdown: float,
    n_positions: int,
    circuit_open: bool,
) -> None:
    cols = st.columns(7)
    metrics = [
        ("PORTFOLIO",   f"${portfolio_value:,.0f}", None),
        ("DAY P&L",     f"${daily_pnl:+,.0f}", daily_pnl),
        ("DAY P&L %",   f"{daily_pnl_pct:+.2f}%", daily_pnl_pct),
        ("SHARPE",      f"{sharpe:.2f}", sharpe - 1),
        ("MAX DD",      f"{drawdown:.1f}%", -drawdown),
        ("POSITIONS",   str(n_positions), None),
        ("CIRCUIT",     "OPEN" if circuit_open else "CLOSED",
         -1 if circuit_open else 1),
    ]
    for col, (label, value, direction) in zip(cols, metrics):
        if direction is not None:
            color = GREEN if direction > 0 else (RED if direction < 0 else AMBER)
        else:
            color = AMBER
        col.markdown(
            f"""
            <div style="background:{SURFACE};border:1px solid {BORDER};
                        padding:8px;border-radius:4px;text-align:center">
                <div style="color:{AMBER};font-size:10px;font-family:Consolas;
                            letter-spacing:1px">{label}</div>
                <div style="color:{color};font-size:18px;font-weight:700;
                            font-family:Consolas">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
