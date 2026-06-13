"""Risk dashboard -- VaR, drawdown, circuit breaker status, correlation heatmap."""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_root = _Path(__file__).resolve().parents[3]
if str(_root) not in _sys.path:
    _sys.path.insert(0, str(_root))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.theme import AMBER, GREEN, RED, SURFACE, BORDER, MUTED, PLOTLY_TEMPLATE
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


def _load_returns_for_heatmap() -> pd.DataFrame | None:
    """Load returns for all tickers with feature data (prefer trained tickers)."""
    try:
        from src.dashboard.predictions import get_trained_tickers
        from src.data.pipeline.storage import load_features

        trained = get_trained_tickers()
        # Use trained tickers + safe macro anchors
        base = ["SPY", "QQQ", "GLD", "TLT"]
        tickers = list(dict.fromkeys(trained[:12] + base))[:16]  # max 16 for readability

        returns = {}
        for t in tickers:
            df = load_features(t)
            if df is not None and "close" in df.columns and len(df) > 30:
                returns[t] = df["close"].pct_change().fillna(0).tail(252)

        if len(returns) >= 2:
            return pd.DataFrame(returns).dropna()
    except Exception:
        pass
    return None


def _load_benchmark_returns(ticker: str = "SPY") -> pd.Series | None:
    try:
        from src.data.pipeline.storage import load_features
        df = load_features(ticker)
        if df is not None and "close" in df.columns and len(df) > 20:
            return df["close"].pct_change().fillna(0).tail(504)
    except Exception:
        pass
    return None


def _get_var_metrics(returns: pd.Series) -> dict:
    try:
        from src.risk.var_calculator import VaRCalculator
        calc = VaRCalculator(confidence=0.99)
        return calc.compute_all(returns)
    except Exception:
        returns_arr = returns.dropna().values
        if len(returns_arr) < 10:
            return {}
        var95 = float(np.percentile(returns_arr, 5))
        cvar  = float(returns_arr[returns_arr <= var95].mean())
        return {"historical_var": var95, "cvar": cvar}


CIRCUIT_BREAKER_DEFS = [
    ("Portfolio Drawdown", "-15%", "HALT_ALL",       "5 days"),
    ("Daily Loss",         "-2%",  "FLAT_ALL",        "1 day"),
    ("Single Name Loss",   "-5%",  "CLOSE_POSITION",  "Immediate"),
    ("Volatility Spike",   "ATR 2.5×", "REDUCE_HALF", "Until normal"),
    ("Correlation Crisis", ">0.85 avg", "MAX_3_POS",   "Until normal"),
    ("Low Confidence",     "<0.55",    "NO_NEW_TRADES","Until signal improves"),
    ("VIX Spike",          ">40",      "REDUCE_75%",   "3 days"),
]


def render():
    st.markdown(
        f'<h2 style="color:{AMBER};font-family:Consolas;letter-spacing:.12em">RISK MONITOR</h2>',
        unsafe_allow_html=True,
    )

    risk = _get_risk_data()
    circuit_open = risk["circuit_open"]
    cb_color = RED if circuit_open else GREEN
    cb_label = "⚠ CIRCUIT OPEN — TRADING HALTED" if circuit_open else "✓ CIRCUIT CLOSED"

    st.markdown(
        f'<div style="background:{SURFACE};border:2px solid {cb_color};padding:10px 16px;'
        f'border-radius:4px;color:{cb_color};font-family:Consolas;font-size:14px;font-weight:700;'
        f'margin-bottom:12px">{cb_label}</div>',
        unsafe_allow_html=True,
    )

    # ── KPI row ──────────────────────────────────────────────────────────────
    spy_r = _load_benchmark_returns("SPY")
    var_metrics = {}
    if spy_r is not None and len(spy_r) > 30:
        var_metrics = _get_var_metrics(spy_r)

    var  = risk["var_1d_99"]  or var_metrics.get("historical_var", 0.0)
    cvar = risk["cvar_1d_99"] or var_metrics.get("cvar", 0.0)
    dd   = risk["current_drawdown"]
    gex  = risk["gross_exposure"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("VaR 99% (1d)",   f"{abs(var):.2%}"  if var != 0 else "N/A",
              delta=None, delta_color="inverse")
    c2.metric("CVaR 99% (1d)",  f"{abs(cvar):.2%}" if cvar != 0 else "N/A",
              delta=None, delta_color="inverse")
    c3.metric("Current DD",     f"{abs(dd):.2%}",   delta=None, delta_color="inverse")
    c4.metric("Gross Exposure", f"{gex:.2f}x" if gex != 0 else "0.00x")

    st.markdown("---")

    col1, col2 = st.columns([3, 2])

    with col1:
        # Drawdown chart
        st.markdown(
            f'<div style="color:{AMBER};font-family:Consolas;font-weight:700;margin-bottom:4px">'
            f'SPY DRAWDOWN</div>',
            unsafe_allow_html=True,
        )
        if spy_r is not None and len(spy_r) > 20:
            st.plotly_chart(drawdown_chart(spy_r), use_container_width=True)
        else:
            st.info("Download data: `python scripts/fetch_data.py`")

        # Correlation heatmap
        st.markdown(
            f'<div style="color:{AMBER};font-family:Consolas;font-weight:700;margin-bottom:4px">'
            f'ROLLING 1-YEAR CORRELATION MATRIX</div>',
            unsafe_allow_html=True,
        )
        with st.spinner("Building correlation matrix..."):
            ret_df = _load_returns_for_heatmap()

        if ret_df is not None and len(ret_df) > 20:
            st.plotly_chart(correlation_heatmap(ret_df), use_container_width=True)
        else:
            st.info("Correlation heatmap requires downloaded data.")

    with col2:
        # Circuit breaker reference table
        st.markdown(
            f'<div style="color:{AMBER};font-family:Consolas;font-weight:700;margin-bottom:8px">'
            f'CIRCUIT BREAKERS</div>',
            unsafe_allow_html=True,
        )
        for name, threshold, action, cooldown in CIRCUIT_BREAKER_DEFS:
            st.markdown(
                f'<div style="background:{SURFACE};border:1px solid {BORDER};'
                f'padding:6px 10px;margin-bottom:4px;font-family:Consolas;font-size:11px">'
                f'<div style="color:{AMBER};font-weight:700">{name}</div>'
                f'<div style="color:{MUTED}">Trigger: {threshold} → <span style="color:{RED}">{action}</span></div>'
                f'<div style="color:{MUTED}">Cooldown: {cooldown}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # VaR breakdown
        st.markdown("---")
        st.markdown(
            f'<div style="color:{AMBER};font-family:Consolas;font-weight:700;margin-bottom:4px">'
            f'VOLATILITY METRICS (SPY 2Y)</div>',
            unsafe_allow_html=True,
        )
        if spy_r is not None and len(spy_r) > 30:
            ann_vol  = spy_r.std() * (252 ** 0.5)
            max_dd_v = ((1 + spy_r).cumprod() / (1 + spy_r).cumprod().cummax() - 1).min()
            skew_v   = float(spy_r.skew())
            kurt_v   = float(spy_r.kurt())
            metrics = [
                ("Ann. Volatility", f"{ann_vol:.1%}"),
                ("Max Drawdown",    f"{max_dd_v:.1%}"),
                ("Skewness",        f"{skew_v:+.2f}"),
                ("Kurtosis",        f"{kurt_v:.2f}"),
                ("Sharpe (SPY)",    f"{(spy_r.mean()*252)/(spy_r.std()*(252**0.5)):.2f}"),
            ]
            for label, val in metrics:
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'font-family:Consolas;font-size:12px;padding:3px 0;'
                    f'border-bottom:1px solid {BORDER}">'
                    f'<span style="color:{MUTED}">{label}</span>'
                    f'<span style="color:{AMBER};font-weight:700">{val}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


if __name__ == "__main__":
    render()
