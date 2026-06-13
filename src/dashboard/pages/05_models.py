"""Models page â€” per-model performance, feature importance."""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_root = _Path(__file__).resolve().parents[3]
if str(_root) not in _sys.path:
    _sys.path.insert(0, str(_root))



import numpy as np
import pandas as pd
import streamlit as st

from src.dashboard.theme import AMBER, GREEN, RED
from src.dashboard.components.charts import feature_importance_bar


def _load_registered_models() -> list[dict]:
    """Try to load model registry entries."""
    try:
        from src.models.registry import ModelRegistry
        reg   = ModelRegistry()
        names = reg.list_models()
        rows  = []
        for name in names:
            info = reg.get_latest_info(name)
            if info:
                m = info.get("metrics", {})
                rows.append({
                    "Model":   name,
                    "Version": info.get("version", "v1"),
                    "Sharpe":  round(m.get("sharpe", m.get("val_acc", 0.0)), 3),
                    "Status":  "champion" if info.get("is_champion") else "registered",
                })
        return rows
    except Exception:
        return []


def _get_feature_importance(model_name: str) -> pd.Series | None:
    """Try to load feature importance for a registered model."""
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


def _get_feature_importance_from_rf(ticker: str = "SPY") -> pd.Series | None:
    """Train a quick RF and get feature importance from real data."""
    try:
        from src.data.pipeline.storage import load_features
        from src.data.features.engineer import get_feature_columns
        from src.models.classical.random_forest import RandomForestModel

        df = load_features(ticker)
        if df is None or len(df) < 200:
            return None

        feat_cols = get_feature_columns(df)
        X = df[feat_cols].fillna(0)
        y = df["target_1d"].fillna(0).astype(int) if "target_1d" in df.columns else pd.Series(0, index=X.index)

        split = int(len(X) * 0.8)
        rf    = RandomForestModel(n_estimators=100, max_depth=6)
        rf.fit(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])
        return rf.get_feature_importance().nlargest(20)
    except Exception:
        return None


def render():
    st.markdown(
        f'<h2 style="color:{AMBER};font-family:Consolas">MODEL PERFORMANCE</h2>',
        unsafe_allow_html=True,
    )

    # Try real registry data first
    reg_rows = _load_registered_models()

    if reg_rows:
        model_df = pd.DataFrame(reg_rows)
        st.success(f"Loaded {len(reg_rows)} registered model(s) from ModelRegistry.")
        st.dataframe(model_df, use_container_width=True, hide_index=True)

        selected_model = st.selectbox("View Feature Importance", model_df["Model"].tolist())
        fi = _get_feature_importance(selected_model)
    else:
        st.info("No models trained yet. Run `python scripts/train_models.py --register` to train models.")
        # Show known model names as placeholders
        model_stats = pd.DataFrame({
            "Model":   ["XGBoost", "LightGBM", "RandomForest", "LSTM", "TCN", "PatchTST", "PPO"],
            "Sharpe":  ["â€”"] * 7,
            "Status":  ["not trained"] * 7,
        })
        st.dataframe(model_stats, use_container_width=True, hide_index=True)

        selected_model = st.selectbox("Compute Feature Importance From",
                                      [t for t in ["SPY", "QQQ", "AAPL", "GLD", "TLT"]])
        fi = None

    # Feature importance
    st.markdown(f'<div style="color:{AMBER};font-family:Consolas;margin-top:12px">FEATURE IMPORTANCE</div>',
                unsafe_allow_html=True)

    if fi is None:
        # Try computing from disk data
        ticker = selected_model if selected_model in ["SPY", "QQQ", "AAPL", "GLD", "TLT"] else "SPY"
        with st.spinner("Computing feature importance from disk data..."):
            fi = _get_feature_importance_from_rf(ticker)

    if fi is not None and len(fi) > 0:
        st.plotly_chart(
            feature_importance_bar(fi, title=f"Feature Importance (RandomForest on {selected_model})"),
            use_container_width=True,
        )
    else:
        st.info("Feature importance will be available after data is downloaded and models are trained.")


render()
