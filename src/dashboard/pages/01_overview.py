"""Overview page — portfolio summary + active signals."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.dashboard.theme import AMBER, BASE
from src.dashboard.components.metrics_bar import render_metrics_bar
from src.dashboard.components.signal_card import render_signal_card
from src.dashboard.components.charts import equity_curve, drawdown_chart


def _load_equity_curve(ticker: str = "SPY") -> pd.Series | None:
    """Load real close prices and convert to returns for equity curve."""
    try:
        from src.data.pipeline.storage import load_features
        df = load_features(ticker)
        if df is not None and "close" in df.columns and len(df) > 10:
            prices = df["close"].dropna()
            equity = (1 + prices.pct_change().fillna(0)).cumprod() * 100_000
            return equity.pct_change().fillna(0)
    except Exception:
        pass
    return None


def _get_live_signals() -> list[tuple]:
    """Fetch signals from AppState or fall back to neutral stubs."""
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
        from src.api.state import get_state
        state   = get_state()
        sigs    = state.get_all_signals()
        results = []
        for s in sigs[:6]:
            ticker = s["ticker"]
            sig    = s["signal"]
            conf   = s["confidence"]
            regime = s.get("regime", 1)
            feats  = s.get("top_features", {})
            results.append((ticker, sig, conf, regime, feats))
        if results:
            return results
    except Exception:
        pass
    return [
        ("SPY",  0, 0.0, 1, {}),
        ("QQQ",  0, 0.0, 1, {}),
        ("AAPL", 0, 0.0, 1, {}),
    ]


def _get_portfolio_metrics() -> dict:
    """Fetch portfolio metrics from AppState or return defaults."""
    try:
        from src.api.state import get_state
        p = get_state().get_portfolio()
        r = get_state().get_risk()
        return {
            "portfolio_value": p["portfolio_value"],
            "daily_pnl":       p["daily_pnl"],
            "daily_pnl_pct":   p["daily_pnl"] / max(p["portfolio_value"], 1) * 100,
            "drawdown":        abs(r["current_drawdown"]) * 100,
            "circuit_open":    r["circuit_open"],
            "n_positions":     len(p["positions"]),
        }
    except Exception:
        return {
            "portfolio_value": 100_000.0, "daily_pnl": 0.0,
            "daily_pnl_pct": 0.0, "drawdown": 0.0,
            "circuit_open": False, "n_positions": 0,
        }


def render():
    st.markdown(
        f'<h2 style="color:{AMBER};font-family:Consolas">PORTFOLIO OVERVIEW</h2>',
        unsafe_allow_html=True,
    )

    pm = _get_portfolio_metrics()

    # Try loading real equity data, fall back to flat line
    returns = _load_equity_curve("SPY")
    if returns is None or len(returns) < 10:
        np.random.seed(42)
        dates   = pd.date_range("2023-01-01", periods=252, freq="B")
        returns = pd.Series(np.zeros(252), index=dates)

    # Compute Sharpe from real returns
    sharpe = 0.0
    if len(returns) > 30:
        ann = returns.mean() * 252
        std = returns.std() * (252 ** 0.5)
        sharpe = ann / std if std > 1e-8 else 0.0

    render_metrics_bar(
        portfolio_value=pm["portfolio_value"],
        daily_pnl=pm["daily_pnl"],
        daily_pnl_pct=pm["daily_pnl_pct"],
        sharpe=round(sharpe, 2),
        drawdown=pm["drawdown"],
        n_positions=pm["n_positions"],
        circuit_open=pm["circuit_open"],
    )
    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(equity_curve(returns, title="Equity Curve (SPY benchmark)"),
                        use_container_width=True)
        st.plotly_chart(drawdown_chart(returns), use_container_width=True)

    with col2:
        st.markdown(f'<div style="color:{AMBER};font-family:Consolas;font-weight:700">ACTIVE SIGNALS</div>',
                    unsafe_allow_html=True)
        signals = _get_live_signals()
        for ticker, sig, conf, regime, feats in signals:
            # Get latest price from features if possible
            price = 0.0
            price_chg = 0.0
            try:
                from src.data.pipeline.storage import load_features
                df = load_features(ticker)
                if df is not None and "close" in df.columns and len(df) > 1:
                    price     = float(df["close"].iloc[-1])
                    price_chg = float(df["close"].pct_change().iloc[-1] * 100)
            except Exception:
                pass
            render_signal_card(ticker, sig, conf, regime, feats,
                               price=price, price_change_pct=price_chg)


render()
