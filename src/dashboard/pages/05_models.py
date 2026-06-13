"""Models page -- per-model performance, feature importance, backtest Sharpe."""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_root = _Path(__file__).resolve().parents[3]
if str(_root) not in _sys.path:
    _sys.path.insert(0, str(_root))

import json
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.theme import AMBER, GREEN, RED, SURFACE, BORDER, MUTED, TEXT, PLOTLY_TEMPLATE


def _load_registry_data() -> tuple[list[dict], dict]:
    """Load all registered models and group by ticker."""
    try:
        from src.models.registry import ModelRegistry
        reg   = ModelRegistry()
        names = reg.list_models()
        rows  = []
        by_ticker: dict = defaultdict(list)
        for name in names:
            info = reg.get_latest_info(name)
            parts = name.rsplit("_", 1)
            ticker   = parts[0] if len(parts) == 2 else name
            model_tp = parts[1].upper() if len(parts) == 2 else "?"
            by_ticker[ticker].append(model_tp)
            if info:
                m = info.get("metrics", {})
                rows.append({
                    "Ticker":   ticker,
                    "Model":    model_tp,
                    "Name":     name,
                    "Version":  info.get("version", "v1"),
                    "Val Acc":  round(m.get("val_acc", 0.0), 3),
                    "Status":   "champion" if info.get("is_champion") else "registered",
                })
        return rows, dict(by_ticker)
    except Exception:
        return [], {}


def _load_backtest_sharpes() -> dict[str, float]:
    """Load per-ticker Sharpe from latest backtest results."""
    try:
        from src.config.constants import PROJECT_ROOT
        results_dir = PROJECT_ROOT / "data" / "backtest_results"
        files = sorted(results_dir.glob("*.json")) if results_dir.exists() else []
        if not files:
            return {}
        with open(files[-1]) as f:
            data = json.load(f)
        folds = data.get("folds", [])
        if not folds:
            return {}
        df = pd.DataFrame(folds)
        if "Ticker" not in df.columns or "Sharpe" not in df.columns:
            return {}
        df["Sharpe"] = pd.to_numeric(df["Sharpe"], errors="coerce")
        return df.groupby("Ticker")["Sharpe"].mean().to_dict()
    except Exception:
        return {}


def _get_feature_importance(model_name: str) -> pd.Series | None:
    try:
        from src.models.registry import ModelRegistry
        reg   = ModelRegistry()
        model = reg.load_model(model_name)
        if model is not None:
            fi = model.get_feature_importance()
            if fi is not None and len(fi) > 0:
                return fi.nlargest(20)
    except Exception:
        pass
    return None


def render():
    st.markdown(
        f'<h2 style="color:{AMBER};font-family:Consolas;letter-spacing:.12em">MODEL PERFORMANCE</h2>',
        unsafe_allow_html=True,
    )

    rows, by_ticker = _load_registry_data()
    sharpes = _load_backtest_sharpes()

    if not rows:
        st.info("No models trained yet. Run `python scripts/train_models.py --register` to train.")
        return

    model_df = pd.DataFrame(rows)
    # Enrich with backtest Sharpe
    model_df["Sharpe (WFO)"] = model_df["Ticker"].map(sharpes).round(3)

    # ── Training status summary ──────────────────────────────────────────────
    total_universe = 25
    complete = [t for t, mods in by_ticker.items() if len(mods) >= 3]
    partial  = [t for t, mods in by_ticker.items() if 0 < len(mods) < 3]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fully Trained",   len(complete))
    c2.metric("In Progress",     len(partial))
    c3.metric("Total Models",    len(rows))
    c4.metric("Universe",        f"{len(complete)}/{total_universe}")
    st.progress(len(complete) / total_universe,
                text=f"Training progress: {len(complete)}/{total_universe} tickers complete")
    st.markdown("---")

    # ── Ticker-level summary ─────────────────────────────────────────────────
    st.markdown(
        f'<div style="color:{AMBER};font-family:Consolas;font-weight:700;margin-bottom:6px">'
        f'TICKER SUMMARY</div>',
        unsafe_allow_html=True,
    )

    ticker_rows = []
    for ticker in sorted(by_ticker.keys()):
        mods = by_ticker[ticker]
        ticker_rows.append({
            "Ticker":    ticker,
            "Models":    " / ".join(sorted(mods)),
            "Complete":  "✓" if len(mods) >= 3 else f"{len(mods)}/3",
            "Sharpe (WFO)": round(sharpes.get(ticker, float("nan")), 2) if ticker in sharpes else "—",
        })

    ticker_df = pd.DataFrame(ticker_rows)

    def _color_sharpe(val):
        try:
            v = float(val)
            if v >= 1.5:
                return f"color:{GREEN}"
            if v >= 0.5:
                return f"color:{AMBER}"
            return f"color:{RED}"
        except Exception:
            return f"color:{MUTED}"

    styled = ticker_df.style.map(_color_sharpe, subset=["Sharpe (WFO)"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Per-model detail + Feature Importance ────────────────────────────────
    st.markdown(
        f'<div style="color:{AMBER};font-family:Consolas;font-weight:700;margin-bottom:6px">'
        f'FEATURE IMPORTANCE</div>',
        unsafe_allow_html=True,
    )

    ticker_filter = st.selectbox("Select Ticker", sorted(by_ticker.keys()))
    available_models = [r["Name"] for r in rows if r["Ticker"] == ticker_filter]
    selected_model   = st.selectbox("Select Model", available_models)

    with st.spinner(f"Loading feature importance for {selected_model}..."):
        fi = _get_feature_importance(selected_model)

    if fi is not None and len(fi) > 0:
        from src.dashboard.components.charts import feature_importance_bar
        st.plotly_chart(
            feature_importance_bar(fi, title=f"Top 20 Features — {selected_model}"),
            use_container_width=True,
        )
    else:
        st.info("Feature importance not available for this model.")

    # ── All models table ─────────────────────────────────────────────────────
    with st.expander("All Registered Models (detailed)"):
        display_cols = ["Name", "Version", "Val Acc", "Sharpe (WFO)", "Status"]
        available = [c for c in display_cols if c in model_df.columns]
        st.dataframe(model_df[available], use_container_width=True, hide_index=True)


if __name__ == "__main__":
    render()
