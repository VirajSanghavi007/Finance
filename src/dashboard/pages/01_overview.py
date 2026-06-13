"""Overview page -- portfolio summary + active signals from trained models."""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_root = _Path(__file__).resolve().parents[3]
if str(_root) not in _sys.path:
    _sys.path.insert(0, str(_root))

import numpy as np
import pandas as pd
import streamlit as st

from src.dashboard.theme import AMBER, BASE, GREEN, RED, SURFACE, BORDER, MUTED, TEXT
from src.dashboard.components.metrics_bar import render_metrics_bar
from src.dashboard.components.signal_card import render_signal_card
from src.dashboard.components.charts import equity_curve, drawdown_chart


def _load_equity_curve(ticker: str = "SPY") -> pd.Series | None:
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


def _get_portfolio_metrics() -> dict:
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


def _load_trained_signals(max_tickers: int = 12) -> list[dict]:
    """Load live signals from all trained models."""
    try:
        from src.dashboard.predictions import get_trained_tickers, get_latest_signal
        tickers = get_trained_tickers()[:max_tickers]
        if not tickers:
            return []
        signals = []
        for t in tickers:
            s = get_latest_signal(t)
            # Get latest price
            try:
                from src.data.pipeline.storage import load_features
                df = load_features(t)
                if df is not None and "close" in df.columns and len(df) > 1:
                    s["price"]      = float(df["close"].iloc[-1])
                    s["price_chg"]  = float(df["close"].pct_change().iloc[-1] * 100)
                else:
                    s["price"] = 0.0; s["price_chg"] = 0.0
            except Exception:
                s["price"] = 0.0; s["price_chg"] = 0.0
            signals.append(s)
        return signals
    except Exception:
        return []


def _load_backtest_summary() -> dict | None:
    """Load latest backtest results summary."""
    try:
        import json
        from src.config.constants import PROJECT_ROOT
        results_dir = PROJECT_ROOT / "data" / "backtest_results"
        files = sorted(results_dir.glob("*.json")) if results_dir.exists() else []
        if files:
            with open(files[-1]) as f:
                data = json.load(f)
            folds = data.get("folds", [])
            if folds:
                import pandas as pd
                df = pd.DataFrame(folds)
                sharpes = pd.to_numeric(df.get("Sharpe", pd.Series([])), errors="coerce").dropna()
                return {
                    "n_tickers": df["Ticker"].nunique() if "Ticker" in df.columns else len(folds),
                    "mean_sharpe": float(sharpes.mean()) if len(sharpes) else 0.0,
                    "pct_positive": float((sharpes > 0).mean() * 100) if len(sharpes) else 0.0,
                }
    except Exception:
        pass
    return None


def render():
    st.markdown(
        f'<h2 style="color:{AMBER};font-family:Consolas;letter-spacing:.12em">PORTFOLIO OVERVIEW</h2>',
        unsafe_allow_html=True,
    )

    pm = _get_portfolio_metrics()
    returns = _load_equity_curve("SPY")
    if returns is None or len(returns) < 10:
        dates = pd.date_range("2023-01-01", periods=252, freq="B")
        returns = pd.Series(np.zeros(252), index=dates)

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

    # Backtest summary strip
    bt = _load_backtest_summary()
    if bt:
        c1, c2, c3 = st.columns(3)
        c1.metric("Tickers Backtested", bt["n_tickers"])
        c2.metric("Mean Sharpe (WFO)", f"{bt['mean_sharpe']:.2f}")
        c3.metric("% Positive Sharpe", f"{bt['pct_positive']:.0f}%")
        st.markdown("---")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown(
            f'<div style="color:{AMBER};font-family:Consolas;font-weight:700;margin-bottom:4px">'
            f'SPY BENCHMARK EQUITY CURVE</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(equity_curve(returns, title="SPY Buy-and-Hold"),
                        use_container_width=True)
        st.plotly_chart(drawdown_chart(returns), use_container_width=True)

    with col2:
        st.markdown(
            f'<div style="color:{AMBER};font-family:Consolas;font-weight:700;margin-bottom:8px">'
            f'LIVE MODEL SIGNALS</div>',
            unsafe_allow_html=True,
        )

        with st.spinner("Loading model predictions..."):
            signals = _load_trained_signals(max_tickers=12)

        if not signals:
            st.info("No trained models found. Run `python scripts/train_models.py --register` first.")
        else:
            # Summary: longs / flats / shorts
            longs  = sum(1 for s in signals if s["signal"] == 1)
            flats  = sum(1 for s in signals if s["signal"] == 0)
            shorts = sum(1 for s in signals if s["signal"] == -1)

            s1, s2, s3 = st.columns(3)
            s1.metric("▲ LONG",  longs,  delta=None)
            s2.metric("— FLAT",  flats,  delta=None)
            s3.metric("▼ SHORT", shorts, delta=None)
            st.markdown("")

            for sig in signals:
                render_signal_card(
                    ticker=sig["ticker"],
                    signal=sig["signal"],
                    confidence=sig["confidence"],
                    regime=1,
                    top_features={},
                    price=sig.get("price", 0.0),
                    price_change_pct=sig.get("price_chg", 0.0),
                )

        # Training progress
        st.markdown("---")
        try:
            from src.dashboard.predictions import get_trained_tickers, get_fully_trained_tickers
            all_t  = get_trained_tickers()
            full_t = get_fully_trained_tickers()
            st.markdown(
                f'<div style="color:{AMBER};font-family:Consolas;font-weight:700;margin-bottom:4px">'
                f'TRAINING STATUS</div>',
                unsafe_allow_html=True,
            )
            total_universe = 25
            st.progress(len(full_t) / total_universe,
                        text=f"{len(full_t)}/{total_universe} tickers fully trained")
        except Exception:
            pass


if __name__ == "__main__":
    render()
