“””Signals page — detailed signal table + regime overlay.”””
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_root = _Path(__file__).resolve().parents[3]
if str(_root) not in _sys.path:
    _sys.path.insert(0, str(_root))



import numpy as np
import pandas as pd
import streamlit as st

from src.dashboard.theme import AMBER, GREEN, RED, SURFACE, BORDER
from src.dashboard.components.charts import signal_overlay, regime_timeline
from src.config.assets import EQUITY_UNIVERSE


def _load_ticker_data(ticker: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (close, signals, regime) from disk features or synthetic fallback."""
    try:
        from src.data.pipeline.storage import load_features
        from src.data.features.engineer import get_feature_columns
        df = load_features(ticker)
        if df is not None and "close" in df.columns and len(df) > 60:
            close   = df["close"].dropna().tail(504)  # ~2 years
            signals = pd.Series(0, index=close.index, dtype=int)

            # Use target_1d as a proxy for "what the model predicted"
            if "target_1d" in df.columns:
                signals = df["target_1d"].reindex(close.index).fillna(0).astype(int)

            regime = pd.Series(1, index=close.index, dtype=int)
            if "reg_hmm_state" in df.columns:
                regime = df["reg_hmm_state"].reindex(close.index).fillna(1).astype(int)
            elif "vol_regime" in df.columns:
                regime = df["vol_regime"].reindex(close.index).fillna(1).astype(int)

            return close, signals, regime
    except Exception:
        pass

    # Fallback
    np.random.seed(0)
    n     = 252
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close   = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.5), index=dates)
    signals = pd.Series(np.zeros(n, dtype=int), index=dates)
    regime  = pd.Series(np.ones(n, dtype=int), index=dates)
    return close, signals, regime


def render():
    st.markdown(
        f'<h2 style="color:{AMBER};font-family:Consolas">SIGNALS</h2>',
        unsafe_allow_html=True,
    )

    selected = st.selectbox(
        "Ticker",
        [t for t in EQUITY_UNIVERSE if not t.startswith("^")],
    )

    close, signals, regime = _load_ticker_data(selected)

    # Try to get live signal from AppState
    live_sig   = 0
    live_conf  = 0.0
    try:
        from src.api.state import get_state
        s = get_state().get_signal(selected)
        if s:
            live_sig  = s["signal"]
            live_conf = s["confidence"]
    except Exception:
        pass

    # Current signal banner
    sig_label = {1: “▲ LONG”, 0: “— FLAT”, -1: “▼ SHORT”}.get(live_sig, “—“)
    sig_color = {1: GREEN, 0: AMBER, -1: RED}.get(live_sig, AMBER)
    st.markdown(
        f'<div style="background:{SURFACE};border-left:4px solid {sig_color};'
        f'padding:8px 12px;margin-bottom:12px;font-family:Consolas;">'
        f'<span style="color:{sig_color};font-size:18px;font-weight:700">{sig_label}</span>'
        f'&nbsp;&nbsp;<span style="color:{AMBER};font-size:13px">confidence {live_conf:.1%}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        st.plotly_chart(signal_overlay(close, signals, title=f”{selected} — Price & Signals”),
                        use_container_width=True)
        st.plotly_chart(regime_timeline(regime), use_container_width=True)
    with col2:
        st.markdown(f'<div style="color:{AMBER};font-family:Consolas">SIGNAL STATS</div>',
                    unsafe_allow_html=True)
        longs  = int((signals == 1).sum())
        flats  = int((signals == 0).sum())
        shorts = int((signals == -1).sum())
        total  = len(signals)
        st.metric("Long",  f"{longs} ({longs/max(total,1):.0%})")
        st.metric("Flat",  f"{flats} ({flats/max(total,1):.0%})")
        st.metric("Short", f"{shorts} ({shorts/max(total,1):.0%})")

        # Compute realized hit rate vs close direction
        fwd_ret = close.pct_change().shift(-1)
        correct = (signals * fwd_ret.sign()).clip(lower=0)
        non_zero = signals[signals != 0]
        if len(non_zero) > 0:
            wr = float((correct[signals != 0] > 0).mean())
            st.metric("Hit Rate", f"{wr:.1%}")
        else:
            st.metric("Hit Rate", "N/A")

        st.metric("Data Points", f"{total:,}")


render()
