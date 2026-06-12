"""Risk dashboard — VaR, drawdown, circuit breaker status."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.dashboard.theme import AMBER, GREEN, RED, SURFACE, BORDER
from src.dashboard.components.charts import drawdown_chart, correlation_heatmap


def _get_risk_data() -> dict:
    try:
        from src.api.state import get_state
        return get_state().get_risk()
    except Exception:
        return {
            "var_1d_99": 0.0, "cvar_1d_99": 0.0,
            "current_drawdown": 0.0, "gross_exposure": 0.0,
            "circuit_open": False, "as_of": "",
        }


def _load_returns_df() -> pd.DataFrame | None:
    """Load returns for several tickers for correlation heatmap."""
    tickers = ["SPY", "QQQ", "AAPL", "GLD", "TLT"]
    returns = {}
    try:
        from src.data.pipeline.storage import load_features
        for t in tickers:
            df = load_features(t)
            if df is not None and "close" in df.columns and len(df) > 30:
                returns[t] = df["close"].pct_change().fillna(0).tail(252)
        if len(returns) >= 2:
            return pd.DataFrame(returns).dropna()
    except Exception:
        pass
    return None


def _load_benchmark_returns() -> pd.Series | None:
    try:
        from src.data.pipeline.storage import load_features
        df = load_features("SPY")
        if df is not None and "close" in df.columns and len(df) > 20:
            return df["close"].pct_change().fillna(0).tail(504)
    except Exception:
        pass
    return None


def render():
    st.markdown(
        f'<h2 style="color:{AMBER};font-family:Consolas">RISK MONITOR</h2>',
        unsafe_allow_html=True,
    )

    risk = _get_risk_data()
    circuit_open = risk["circuit_open"]
    cb_color = RED if circuit_open else GREEN
    cb_label  = "CIRCUIT OPEN — TRADING HALTED" if circuit_open else "CIRCUIT CLOSED"

    st.markdown(
        f'<div style="background:{SURFACE};border:2px solid {cb_color};padding:10px;'
        f'border-radius:4px;color:{cb_color};font-family:Consolas;font-size:14px;font-weight:700">'
        f'{cb_label}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    var  = risk["var_1d_99"]
    cvar = risk["cvar_1d_99"]
    dd   = risk["current_drawdown"]
    gex  = risk["gross_exposure"]

    # Compute VaR from real returns if state values are zero
    if var == 0.0:
        try:
            spy_r = _load_benchmark_returns()
            if spy_r is not None and len(spy_r) > 30:
                from src.risk.var_calculator import VaRCalculator
                calc  = VaRCalculator(confidence=0.99)
                all_v = calc.compute_all(spy_r)
                var   = all_v.get("historical_var", 0.0)
                cvar  = all_v.get("cvar", 0.0)
        except Exception:
            pass

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("VaR 99% (1d)",   f"{var:.2%}"  if var != 0 else "N/A")
    col2.metric("CVaR 99% (1d)",  f"{cvar:.2%}" if cvar != 0 else "N/A")
    col3.metric("Current DD",     f"{dd:.2%}"   if dd != 0 else "0.00%")
    col4.metric("Gross Exposure", f"{gex:.2f}x" if gex != 0 else "0.00x")

    # Drawdown chart from real data
    spy_r = _load_benchmark_returns()
    if spy_r is not None and len(spy_r) > 20:
        st.plotly_chart(drawdown_chart(spy_r), use_container_width=True)
    else:
        st.info("No price data yet. Run `python scripts/fetch_data.py` to download data.")

    # Correlation heatmap from real data
    ret_df = _load_returns_df()
    if ret_df is not None and len(ret_df) > 20:
        st.plotly_chart(correlation_heatmap(ret_df), use_container_width=True)
    else:
        # Fallback with fewer tickers
        st.info("Correlation heatmap will populate once data is downloaded.")


render()
