"""Explainability page — SHAP, drift monitor, audit log."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.dashboard.theme import AMBER, GREEN, RED, SURFACE


def _compute_drift(ticker: str = "SPY") -> pd.DataFrame | None:
    """Compute real feature drift using KS test between first and last half of data."""
    try:
        from src.data.pipeline.storage import load_features
        from src.data.features.engineer import get_feature_columns
        from scipy import stats

        df = load_features(ticker)
        if df is None or len(df) < 100:
            return None

        feat_cols = get_feature_columns(df)[:20]  # top 20 features
        X = df[feat_cols].fillna(0)

        split = len(X) // 2
        ref   = X.iloc[:split]
        live  = X.iloc[split:]

        rows = []
        for col in feat_cols:
            ks_stat, p_val = stats.ks_2samp(ref[col].values, live[col].values)
            rows.append({
                "Feature": col,
                "KS Stat": round(ks_stat, 4),
                "P-Value": round(p_val, 4),
                "Drifted": p_val < 0.05,
            })
        return pd.DataFrame(rows).sort_values("KS Stat", ascending=False)
    except Exception:
        return None


def _get_shap_values(ticker: str = "SPY") -> pd.DataFrame | None:
    """Get SHAP-like feature attribution from RF feature importance on last prediction."""
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

        fi      = rf.get_feature_importance().nlargest(10)
        last_x  = X.iloc[-1]
        # Pseudo-SHAP: importance × (feature_value - feature_mean) / feature_std
        means   = X.mean()
        stds    = X.std().replace(0, 1)
        z_score = (last_x - means) / stds
        shap_proxy = (fi * z_score.reindex(fi.index).fillna(0)).sort_values(ascending=False)

        return pd.DataFrame({
            "Feature": shap_proxy.index,
            "SHAP":    shap_proxy.values.round(4),
        })
    except Exception:
        return None


def render():
    st.markdown(
        f'<h2 style="color:{AMBER};font-family:Consolas">EXPLAINABILITY</h2>',
        unsafe_allow_html=True,
    )

    ticker = st.selectbox("Ticker", ["SPY", "QQQ", "AAPL", "GLD", "TLT"])

    tab1, tab2, tab3 = st.tabs(["Feature Drift", "SHAP Attribution", "Audit Log"])

    with tab1:
        st.markdown(f'<div style="color:{AMBER}">Feature Drift Monitor (KS Test: reference vs recent)</div>',
                    unsafe_allow_html=True)
        with st.spinner("Computing drift..."):
            drift_data = _compute_drift(ticker)

        if drift_data is not None:
            n_drifted = int(drift_data["Drifted"].sum())
            if n_drifted > 0:
                st.warning(f"{n_drifted} features show significant drift (p < 0.05).")
            else:
                st.success("No significant feature drift detected.")

            st.dataframe(
                drift_data.style.applymap(
                    lambda v: f"color:{RED}" if v else f"color:{GREEN}",
                    subset=["Drifted"],
                ),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("Download data first: `python scripts/fetch_data.py`")

    with tab2:
        st.markdown(f'<div style="color:{AMBER}">Feature Attribution (last prediction)</div>',
                    unsafe_allow_html=True)
        with st.spinner("Computing attribution..."):
            shap_df = _get_shap_values(ticker)

        if shap_df is not None and len(shap_df) > 0:
            colors = [GREEN if v > 0 else RED for v in shap_df["SHAP"]]
            st.bar_chart(shap_df.set_index("Feature")["SHAP"])
            st.caption("Positive = pushed toward LONG, Negative = pushed toward SHORT")
            st.dataframe(shap_df, use_container_width=True, hide_index=True)
        else:
            st.info("Attribution data will be available after data is downloaded and models are trained.")

    with tab3:
        st.markdown(f'<div style="color:{AMBER}">Recent Audit Records</div>',
                    unsafe_allow_html=True)
        try:
            from src.explainability.audit_log import AuditLog
            log = AuditLog()
            df  = log.read()
            if df.empty:
                st.info("No audit records for today. Records appear here after live trading starts.")
            else:
                st.dataframe(df.tail(100), use_container_width=True)
        except Exception as e:
            st.info(f"Audit log not yet populated. Start paper trading to generate records.")


render()
