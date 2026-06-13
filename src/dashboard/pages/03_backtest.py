“””Backtest results page — WFO fold performance, metrics table.”””
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_root = _Path(__file__).resolve().parents[3]
if str(_root) not in _sys.path:
    _sys.path.insert(0, str(_root))



import numpy as np
import pandas as pd
import streamlit as st

from src.dashboard.theme import AMBER, GREEN, RED, SURFACE
from src.dashboard.components.charts import equity_curve


def _load_backtest_results() -> pd.DataFrame | None:
    """Try to load saved backtest results from disk."""
    try:
        from src.config.constants import PROJECT_ROOT
        import json
        results_dir = PROJECT_ROOT / "data" / "backtest_results"
        files = list(results_dir.glob("*.json")) if results_dir.exists() else []
        if files:
            latest = sorted(files)[-1]
            with open(latest) as f:
                data = json.load(f)
            if "folds" in data:
                return pd.DataFrame(data["folds"])
    except Exception:
        pass
    return None


def _load_equity_for_ticker(ticker: str) -> pd.Series | None:
    try:
        from src.data.pipeline.storage import load_features
        df = load_features(ticker)
        if df is not None and "close" in df.columns and len(df) > 20:
            prices  = df["close"].dropna()
            returns = prices.pct_change().fillna(0)
            return returns
    except Exception:
        pass
    return None


def render():
    st.markdown(
        f'<h2 style="color:{AMBER};font-family:Consolas">BACKTEST RESULTS</h2>',
        unsafe_allow_html=True,
    )

    saved = _load_backtest_results()

    if saved is not None and len(saved) > 0:
        st.info("Loaded saved backtest results from disk.")
        fold_data = saved
    else:
        st.info("No saved backtest results. Run `python scripts/run_backtest.py` to generate them. "
                "Showing SPY buy-and-hold benchmark as placeholder.")

        # Load real SPY data as benchmark
        spy_returns = _load_equity_for_ticker("SPY")
        if spy_returns is not None and len(spy_returns) > 100:
            # Build fold summary from SPY data (monthly)
            spy_monthly = spy_returns.resample("ME").sum()
            spy_cum     = (1 + spy_returns).cumprod()
            n_folds     = min(8, len(spy_monthly) // 3)
            chunk       = len(spy_returns) // max(n_folds, 1)

            rows = []
            for i in range(n_folds):
                s = spy_returns.iloc[i*chunk:(i+1)*chunk]
                if len(s) < 10:
                    continue
                ann  = s.mean() * 252
                std  = s.std() * (252**0.5)
                sh   = ann / std if std > 1e-8 else 0.0
                dd   = ((1+s).cumprod() / (1+s).cumprod().cummax() - 1).min() * 100
                rows.append({
                    "Fold":        i + 1,
                    "Start":       s.index[0].strftime("%Y-%m"),
                    "Sharpe":      round(sh, 2),
                    "Total Ret %": round(ann * (len(s)/252) * 100, 2),
                    "Max DD %":    round(abs(dd), 2),
                    "N Trades":    0,  # placeholder
                })
            fold_data = pd.DataFrame(rows) if rows else pd.DataFrame()
        else:
            fold_data = pd.DataFrame()

    if len(fold_data) > 0:
        def color_sharpe(val):
            try:
                v = float(val)
                if v >= 1.5:
                    return f"color:{GREEN}"
                if v >= 1.0:
                    return f"color:{AMBER}"
                return f"color:{RED}"
            except Exception:
                return ""

        sharpe_col = "Sharpe" if "Sharpe" in fold_data.columns else fold_data.columns[2]
        try:
            styled = fold_data.style.map(color_sharpe, subset=[sharpe_col])
        except AttributeError:
            styled = fold_data.style.applymap(color_sharpe, subset=[sharpe_col])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        if sharpe_col in fold_data.columns:
            sharpes = pd.to_numeric(fold_data[sharpe_col], errors="coerce").dropna()
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Avg Sharpe", f"{sharpes.mean():.2f}")
            ret_col = "Total Ret %" if "Total Ret %" in fold_data.columns else None
            if ret_col:
                rets = pd.to_numeric(fold_data[ret_col], errors="coerce").dropna()
                col2.metric("Avg Return", f"{rets.mean():.1f}%")
            dd_col = "Max DD %" if "Max DD %" in fold_data.columns else None
            if dd_col:
                dds = pd.to_numeric(fold_data[dd_col], errors="coerce").dropna()
                col3.metric("Avg Max DD", f"{dds.mean():.1f}%")
            n = len(fold_data)
            col4.metric("Win Folds", f"{(sharpes > 1.0).sum()}/{n}")

    # Equity curve chart — real benchmark data
    st.subheader("SPY Benchmark Equity Curve")
    spy_r = _load_equity_for_ticker("SPY")
    if spy_r is not None and len(spy_r) > 10:
        st.plotly_chart(equity_curve(spy_r, title="SPY Buy-and-Hold"),
                        use_container_width=True)
    else:
        st.warning("Download SPY data first: `python scripts/fetch_data.py`")


render()
