"""Backtest results page -- WFO fold performance, per-ticker breakdown."""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_root = _Path(__file__).resolve().parents[3]
if str(_root) not in _sys.path:
    _sys.path.insert(0, str(_root))

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.theme import AMBER, GREEN, RED, SURFACE, BORDER, MUTED, TEXT, PLOTLY_TEMPLATE
from src.dashboard.components.charts import equity_curve


def _load_all_backtest_results() -> pd.DataFrame | None:
    """Load the latest saved backtest JSON and return as DataFrame."""
    try:
        from src.config.constants import PROJECT_ROOT
        results_dir = PROJECT_ROOT / "data" / "backtest_results"
        files = sorted(results_dir.glob("*.json")) if results_dir.exists() else []
        if not files:
            return None
        with open(files[-1]) as f:
            data = json.load(f)
        folds = data.get("folds", [])
        if folds:
            df = pd.DataFrame(folds)
            # Coerce numeric
            for col in ["Sharpe", "CAGR %", "Max DD %", "Win Rate", "N Trades"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            return df
    except Exception:
        pass
    return None


def _load_benchmark_curve() -> pd.Series | None:
    try:
        from src.data.pipeline.storage import load_features
        df = load_features("SPY")
        if df is not None and "close" in df.columns and len(df) > 20:
            return df["close"].pct_change().fillna(0).tail(504)
    except Exception:
        pass
    return None


def _sharpe_color(val: float) -> str:
    if val >= 1.5:
        return f"color:{GREEN};font-weight:700"
    if val >= 0.5:
        return f"color:{AMBER}"
    return f"color:{RED}"


def render():
    st.markdown(
        f'<h2 style="color:{AMBER};font-family:Consolas;letter-spacing:.12em">BACKTEST RESULTS</h2>',
        unsafe_allow_html=True,
    )

    fold_df = _load_all_backtest_results()

    if fold_df is None or len(fold_df) == 0:
        st.info(
            "No saved backtest results. Run:\n"
            "`python scripts/run_full_backtest.py`\n\n"
            "Showing SPY buy-and-hold benchmark as placeholder."
        )
        fold_df = None

    if fold_df is not None:
        # ── Aggregate KPIs ───────────────────────────────────────────────────
        sharpes    = fold_df["Sharpe"].dropna()
        cagr_col   = "CAGR %" if "CAGR %" in fold_df.columns else "Total Ret %"
        dd_col     = "Max DD %" if "Max DD %" in fold_df.columns else None
        n_tickers  = fold_df["Ticker"].nunique() if "Ticker" in fold_df.columns else len(fold_df)
        pct_pos    = (sharpes > 0).mean() * 100
        pct_good   = (sharpes >= 1.0).mean() * 100

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Tickers", n_tickers)
        c2.metric("Mean Sharpe", f"{sharpes.mean():.2f}")
        c3.metric("Median Sharpe", f"{sharpes.median():.2f}")
        c4.metric("% Positive", f"{pct_pos:.0f}%")
        c5.metric("% Sharpe ≥ 1.0", f"{pct_good:.0f}%")
        st.markdown("---")

        # ── Ticker summary table ─────────────────────────────────────────────
        st.markdown(
            f'<div style="color:{AMBER};font-family:Consolas;font-weight:700;margin-bottom:6px">'
            f'TICKER BREAKDOWN</div>',
            unsafe_allow_html=True,
        )

        if "Ticker" in fold_df.columns:
            ticker_grp = fold_df.groupby("Ticker").agg(
                Sharpe=("Sharpe", "mean"),
                **({cagr_col: (cagr_col, "mean")} if cagr_col in fold_df.columns else {}),
                **({dd_col:    (dd_col,   "mean")} if dd_col in fold_df.columns else {}),
                **({  "Win Rate": ("Win Rate", "mean")} if "Win Rate" in fold_df.columns else {}),
                **({  "N Trades":("N Trades",  "sum")} if "N Trades" in fold_df.columns else {}),
            ).reset_index().sort_values("Sharpe", ascending=False)

            def color_sharpe_cell(val):
                try:
                    return _sharpe_color(float(val))
                except Exception:
                    return ""

            try:
                styled = ticker_grp.style.map(color_sharpe_cell, subset=["Sharpe"])
            except AttributeError:
                styled = ticker_grp.style.applymap(color_sharpe_cell, subset=["Sharpe"])
            st.dataframe(styled, use_container_width=True, hide_index=True)

        # ── Sharpe bar chart ─────────────────────────────────────────────────
        if "Ticker" in fold_df.columns:
            grp      = fold_df.groupby("Ticker")["Sharpe"].mean().sort_values(ascending=True)
            bar_clrs = [GREEN if v >= 1.0 else (AMBER if v >= 0 else RED) for v in grp.values]
            fig = go.Figure(go.Bar(
                x=grp.values,
                y=grp.index,
                orientation="h",
                marker_color=bar_clrs,
                text=[f"{v:.2f}" for v in grp.values],
                textposition="outside",
                textfont=dict(color=AMBER, size=11),
            ))
            fig.add_vline(x=1.0, line=dict(color=AMBER, width=1, dash="dot"),
                          annotation_text="1.0 target", annotation_font_color=AMBER)
            fig.update_layout(
                template=PLOTLY_TEMPLATE,
                title="Sharpe Ratio by Ticker (WFO backtest)",
                height=max(300, len(grp) * 30 + 60),
                xaxis_title="Sharpe ratio",
                margin=dict(l=80, r=60, t=40, b=30),
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Detailed folds table ─────────────────────────────────────────────
        with st.expander("All Fold Results (detailed)"):
            display_cols = ["Ticker", "Sharpe", cagr_col, dd_col, "Win Rate", "N Trades", "Model"]
            display_cols = [c for c in display_cols if c and c in fold_df.columns]
            try:
                styled2 = fold_df[display_cols].style.map(
                    color_sharpe_cell, subset=["Sharpe"]
                )
            except AttributeError:
                styled2 = fold_df[display_cols].style.applymap(
                    color_sharpe_cell, subset=["Sharpe"]
                )
            st.dataframe(styled2, use_container_width=True, hide_index=True)

    # ── SPY Benchmark Equity Curve ───────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        f'<div style="color:{AMBER};font-family:Consolas;font-weight:700;margin-bottom:6px">'
        f'SPY BENCHMARK EQUITY CURVE</div>',
        unsafe_allow_html=True,
    )
    spy_r = _load_benchmark_curve()
    if spy_r is not None and len(spy_r) > 10:
        st.plotly_chart(equity_curve(spy_r, title="SPY Buy-and-Hold (last 2 years)"),
                        use_container_width=True)
    else:
        st.warning("Download SPY data first: `python scripts/fetch_data.py`")

    # ── Run buttons ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        f'<div style="color:{MUTED};font-family:Consolas;font-size:12px">'
        f'To refresh results: <code>python scripts/run_full_backtest.py</code><br/>'
        f'To train more tickers: <code>python scripts/train_models.py --register --tickers JPM GS XOM</code>'
        f'</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    render()
