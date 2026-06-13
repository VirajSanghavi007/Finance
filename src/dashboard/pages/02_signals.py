"""Signals page -- real model predictions + regime overlay."""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_root = _Path(__file__).resolve().parents[3]
if str(_root) not in _sys.path:
    _sys.path.insert(0, str(_root))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.dashboard.theme import (
    AMBER, GREEN, RED, SURFACE, BORDER, MUTED, TEXT, PLOTLY_TEMPLATE,
)
from src.dashboard.components.charts import signal_overlay, regime_timeline
from src.config.assets import EQUITY_UNIVERSE


def _load_ticker_data(ticker: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (close, regime, target_direction) from disk features."""
    try:
        from src.data.pipeline.storage import load_features
        df = load_features(ticker)
        if df is not None and "close" in df.columns and len(df) > 60:
            close  = df["close"].dropna().tail(504)
            regime = pd.Series(1, index=close.index, dtype=int)
            if "reg_hmm_state" in df.columns:
                regime = df["reg_hmm_state"].reindex(close.index).fillna(1).astype(int)
            elif "vol_regime" in df.columns:
                regime = df["vol_regime"].reindex(close.index).fillna(1).astype(int)
            direction = close.pct_change().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
            return close, regime, direction
    except Exception:
        pass
    n = 252
    dates     = pd.date_range("2023-01-01", periods=n, freq="B")
    close     = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.5), index=dates)
    regime    = pd.Series(np.ones(n, dtype=int), index=dates)
    direction = pd.Series(np.zeros(n, dtype=int), index=dates)
    return close, regime, direction


def _get_model_predictions(ticker: str) -> pd.DataFrame | None:
    """Get real model predictions for the ticker."""
    try:
        from src.dashboard.predictions import predict_ticker
        return predict_ticker(ticker, n_rows=504)
    except Exception:
        return None


def _get_base_model_predictions(ticker: str) -> dict[str, np.ndarray]:
    """Get per-model predictions for comparison."""
    try:
        from src.models.registry import ModelRegistry
        from src.data.pipeline.storage import load_features
        from src.data.features.engineer import get_feature_columns

        df = load_features(ticker)
        if df is None or len(df) < 60:
            return {}
        feat_cols = get_feature_columns(df)
        X = df[feat_cols].select_dtypes(include=[np.number]).fillna(0).tail(504)

        reg = ModelRegistry()
        results = {}
        for suffix in ["rf", "xgb", "lgbm"]:
            m = reg.load_model(f"{ticker}_{suffix}")
            if m is not None:
                try:
                    preds = m.predict(X)
                    results[suffix.upper()] = preds
                except Exception:
                    pass
        return results
    except Exception:
        return {}


def _get_trained_tickers() -> list[str]:
    try:
        from src.dashboard.predictions import get_trained_tickers
        return get_trained_tickers()
    except Exception:
        return [t for t in EQUITY_UNIVERSE if not t.startswith("^")]


def _confidence_bar(confidence: float, sig: int) -> str:
    color = GREEN if sig == 1 else (RED if sig == -1 else AMBER)
    pct   = f"{confidence:.0%}"
    return (
        f'<div style="background:{BORDER};border-radius:2px;height:8px;width:100%">'
        f'<div style="background:{color};width:{confidence*100:.0f}%;height:8px;border-radius:2px"></div>'
        f'</div><span style="font-size:10px;color:{MUTED}">{pct}</span>'
    )


def render():
    st.markdown(
        f'<h2 style="color:{AMBER};font-family:Consolas;letter-spacing:.12em">SIGNALS</h2>',
        unsafe_allow_html=True,
    )

    tickers = _get_trained_tickers()
    all_tickers = list(EQUITY_UNIVERSE.keys())

    # Ticker selection — trained first, then untrained
    trained_set = set(tickers)
    ordered = tickers + [t for t in all_tickers if t not in trained_set]

    selected = st.selectbox("Ticker", ordered,
                            format_func=lambda t: f"{'✓ ' if t in trained_set else '○ '}{t}")

    close, regime, _ = _load_ticker_data(selected)

    # ── Model predictions ────────────────────────────────────────────────────
    preds = _get_model_predictions(selected)
    has_model = preds is not None and len(preds) > 0

    if has_model:
        signals    = preds["signal"].reindex(close.index).fillna(0).astype(int)
        confidence = preds["confidence"].reindex(close.index).fillna(0.0)
        source     = preds["source"].iloc[-1] if "source" in preds.columns else "model"
        live_sig   = int(signals.iloc[-1])
        live_conf  = float(confidence.iloc[-1])
    else:
        # Use target_1d as fallback
        try:
            from src.data.pipeline.storage import load_features
            df = load_features(selected)
            if df is not None and "target_1d" in df.columns:
                signals = df["target_1d"].reindex(close.index).fillna(0).astype(int)
            else:
                signals = pd.Series(0, index=close.index, dtype=int)
        except Exception:
            signals = pd.Series(0, index=close.index, dtype=int)
        confidence = pd.Series(0.0, index=close.index)
        source     = "none"
        live_sig   = 0
        live_conf  = 0.0

    # ── Current signal banner ────────────────────────────────────────────────
    sig_label = {1: "▲ LONG", 0: "— FLAT", -1: "▼ SHORT"}.get(live_sig, "— FLAT")
    sig_color = {1: GREEN, 0: AMBER, -1: RED}.get(live_sig, AMBER)
    model_src = source.upper() if has_model else "NO MODEL"

    st.markdown(
        f'<div style="background:{SURFACE};border:1px solid {BORDER};border-left:4px solid {sig_color};'
        f'padding:10px 16px;margin-bottom:12px;font-family:Consolas;display:flex;gap:24px;align-items:center">'
        f'<span style="color:{sig_color};font-size:20px;font-weight:700">{sig_label}</span>'
        f'<span style="color:{AMBER};font-size:13px">confidence {live_conf:.1%}</span>'
        f'<span style="color:{MUTED};font-size:11px">source: {model_src}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 1])

    with col1:
        st.plotly_chart(
            signal_overlay(close, signals, title=f"{selected} — Price & Model Signals"),
            use_container_width=True,
        )
        st.plotly_chart(regime_timeline(regime), use_container_width=True)

        # Confidence over time (if we have model predictions)
        if has_model and len(confidence) > 0:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=confidence.index,
                y=confidence.values,
                mode="lines",
                line=dict(color=AMBER, width=1),
                fill="tozeroy",
                fillcolor=f"rgba(255,179,0,0.08)",
                name="Confidence",
            ))
            fig.add_hline(y=0.62, line=dict(color=RED, width=1, dash="dash"),
                          annotation_text="0.62 threshold", annotation_font_color=RED)
            fig.update_layout(
                template=PLOTLY_TEMPLATE,
                title=f"{selected} — Model Confidence",
                height=180,
                showlegend=False,
                margin=dict(l=40, r=10, t=30, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(
            f'<div style="color:{AMBER};font-family:Consolas;font-weight:700">SIGNAL STATS</div>',
            unsafe_allow_html=True,
        )
        longs  = int((signals == 1).sum())
        flats  = int((signals == 0).sum())
        shorts = int((signals == -1).sum())
        total  = len(signals)

        st.metric("▲ Long",  f"{longs}  ({longs/max(total,1):.0%})")
        st.metric("— Flat",  f"{flats}  ({flats/max(total,1):.0%})")
        st.metric("▼ Short", f"{shorts} ({shorts/max(total,1):.0%})")

        # Realized hit rate
        fwd_ret = close.pct_change().shift(-1)
        fwd_dir = fwd_ret.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        non_zero_mask = signals != 0
        if non_zero_mask.sum() > 0:
            correct = (signals[non_zero_mask] == fwd_dir[non_zero_mask]).mean()
            st.metric("Hit Rate", f"{correct:.1%}")
        else:
            st.metric("Hit Rate", "N/A")

        avg_conf = confidence[non_zero_mask].mean() if non_zero_mask.sum() > 0 else 0.0
        st.metric("Avg Confidence", f"{avg_conf:.1%}" if has_model else "N/A")

        if has_model:
            high_conf = (confidence >= 0.62).sum()
            st.metric("High-Conf Bars", f"{high_conf}")

        st.metric("Data Points", f"{total:,}")

        # Per-model predictions comparison
        st.markdown("---")
        st.markdown(
            f'<div style="color:{AMBER};font-family:Consolas;font-weight:700;font-size:11px">MODEL CONSENSUS</div>',
            unsafe_allow_html=True,
        )
        base_preds = _get_base_model_predictions(selected)
        if base_preds:
            for model_name, preds_arr in base_preds.items():
                last_pred = int(preds_arr[-1]) if len(preds_arr) > 0 else 0
                label = {1: "▲", 0: "—", -1: "▼"}.get(last_pred, "—")
                color = {1: GREEN, 0: AMBER, -1: RED}.get(last_pred, MUTED)
                agree = sum(1 for p in [v[-1] if len(v) > 0 else 0
                                        for v in base_preds.values()] if p == last_pred)
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;font-family:Consolas;'
                    f'font-size:12px;padding:2px 0;border-bottom:1px solid {BORDER}">'
                    f'<span style="color:{MUTED}">{model_name}</span>'
                    f'<span style="color:{color};font-weight:700">{label}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                f'<div style="color:{MUTED};font-size:11px">'
                f'Train models first:<br/>'
                f'<code>python scripts/train_models.py --register</code>'
                f'</div>',
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    render()
