"""Explainability page -- real SHAP from trained models, drift monitor, audit log."""
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


def _get_trained_tickers() -> list[str]:
    try:
        from src.dashboard.predictions import get_trained_tickers
        return get_trained_tickers()
    except Exception:
        return ["SPY", "QQQ", "AAPL"]


def _compute_real_shap(ticker: str) -> tuple[pd.DataFrame | None, str]:
    """
    Compute real SHAP values using the trained XGB model from the registry.
    Falls back to RF feature importance if SHAP fails.
    Returns (DataFrame with Feature/SHAP columns, method_name).
    """
    try:
        import shap
        from src.models.registry import ModelRegistry
        from src.data.pipeline.storage import load_features
        from src.data.features.engineer import get_feature_columns

        df = load_features(ticker)
        if df is None or len(df) < 60:
            return None, "no data"

        feat_cols = get_feature_columns(df)
        X = df[feat_cols].select_dtypes(include=[np.number]).fillna(0)
        X_sample = X.tail(100)  # use last 100 bars for SHAP background

        reg = ModelRegistry()
        # Prefer XGB (TreeExplainer is exact + fast), then LGBM, then RF
        for suffix in ["xgb", "lgbm", "rf"]:
            model = reg.load_model(f"{ticker}_{suffix}")
            if model is None:
                continue
            try:
                raw = model._model  # underlying sklearn/xgb estimator
                explainer = shap.TreeExplainer(raw)
                shap_vals = explainer.shap_values(X_sample.values)
                sv_arr = np.array(shap_vals)

                # Shape possibilities:
                # (n_samples, n_features, n_classes) — XGBoost 3D
                # (n_classes, n_samples, n_features) — list-style legacy
                # (n_samples, n_features) — binary
                if sv_arr.ndim == 3:
                    # (n_samples, n_features, n_classes)
                    sv = sv_arr[:, :, 2] - sv_arr[:, :, 0]  # LONG - SHORT
                elif isinstance(shap_vals, list):
                    sv = np.array(shap_vals[2]) - np.array(shap_vals[0])
                else:
                    sv = sv_arr  # binary: (n_samples, n_features)

                mean_abs_shap = np.abs(sv).mean(axis=0)
                last_shap     = sv[-1]  # SHAP for most recent bar

                feat_names = list(X_sample.columns)
                result = pd.DataFrame({
                    "Feature":       feat_names,
                    "SHAP (latest)": last_shap.round(5),
                    "Mean |SHAP|":   mean_abs_shap.round(5),
                })
                result = result.reindex(result["Mean |SHAP|"].abs().sort_values(ascending=False).index)
                return result.head(20), suffix.upper()
            except Exception:
                continue

        # Last resort: feature importance × z-score proxy
        model = reg.load_model(f"{ticker}_rf")
        if model is not None:
            fi      = model.get_feature_importance()
            last_x  = X.iloc[-1]
            means   = X.mean()
            stds    = X.std().replace(0, 1)
            z       = (last_x - means) / stds
            proxy   = (fi * z.reindex(fi.index).fillna(0))
            top     = proxy.abs().nlargest(20).index
            proxy   = proxy.loc[top].sort_values(ascending=False)
            return pd.DataFrame({
                "Feature":       proxy.index,
                "SHAP (latest)": proxy.values.round(5),
                "Mean |SHAP|":   fi.reindex(top).fillna(0).values.round(5),
            }), "RF proxy"

    except Exception as e:
        return None, f"error: {e}"

    return None, "no model"


def _compute_drift(ticker: str, n_features: int = 20) -> pd.DataFrame | None:
    """KS test: first half vs second half of feature data."""
    try:
        from src.data.pipeline.storage import load_features
        from src.data.features.engineer import get_feature_columns
        from scipy import stats

        df = load_features(ticker)
        if df is None or len(df) < 100:
            return None

        feat_cols = get_feature_columns(df)[:n_features]
        X     = df[feat_cols].fillna(0)
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
                "Status":  "DRIFT" if p_val < 0.05 else "stable",
            })
        return pd.DataFrame(rows).sort_values("KS Stat", ascending=False)
    except Exception:
        return None


def render():
    st.markdown(
        f'<h2 style="color:{AMBER};font-family:Consolas;letter-spacing:.12em">EXPLAINABILITY</h2>',
        unsafe_allow_html=True,
    )

    tickers = _get_trained_tickers()
    ticker  = st.selectbox("Ticker", tickers if tickers else ["SPY"])

    tab1, tab2, tab3 = st.tabs(["SHAP Attribution", "Feature Drift", "Audit Log"])

    # ── Tab 1: SHAP ──────────────────────────────────────────────────────────
    with tab1:
        st.markdown(
            f'<div style="color:{AMBER};font-family:Consolas;margin-bottom:6px">'
            f'SHAP VALUES — LAST PREDICTION (using trained model from registry)</div>',
            unsafe_allow_html=True,
        )

        with st.spinner(f"Computing SHAP for {ticker}..."):
            shap_df, method = _compute_real_shap(ticker)

        if shap_df is not None and len(shap_df) > 0:
            st.caption(f"Source: {method} — positive SHAP → pushes toward LONG, negative → SHORT")

            # Waterfall-style bar chart
            vals   = shap_df["SHAP (latest)"].values
            feats  = shap_df["Feature"].values
            colors = [GREEN if v > 0 else RED for v in vals]

            fig = go.Figure(go.Bar(
                x=vals,
                y=feats,
                orientation="h",
                marker_color=colors,
                text=[f"{v:+.4f}" for v in vals],
                textposition="outside",
                textfont=dict(color=AMBER, size=10),
            ))
            fig.update_layout(
                template=PLOTLY_TEMPLATE,
                title=f"SHAP Waterfall — {ticker} (last bar)",
                height=max(400, len(feats) * 22),
                xaxis_title="SHAP value (impact on model output)",
                yaxis=dict(autorange="reversed"),
                margin=dict(l=180, r=60, t=40, b=30),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Mean |SHAP| bar (global importance)
            fig2 = go.Figure(go.Bar(
                x=shap_df["Mean |SHAP|"].values,
                y=feats,
                orientation="h",
                marker_color=AMBER,
            ))
            fig2.update_layout(
                template=PLOTLY_TEMPLATE,
                title=f"Mean |SHAP| — {ticker} (global importance, last 100 bars)",
                height=max(350, len(feats) * 20),
                xaxis_title="Mean absolute SHAP",
                yaxis=dict(autorange="reversed"),
                margin=dict(l=180, r=40, t=40, b=30),
            )
            st.plotly_chart(fig2, use_container_width=True)

            st.dataframe(shap_df.reset_index(drop=True), use_container_width=True, hide_index=True)
        else:
            st.info(f"No trained model for {ticker}. "
                    f"Run: `python scripts/train_models.py --register --tickers {ticker}`")

    # ── Tab 2: Feature Drift ─────────────────────────────────────────────────
    with tab2:
        st.markdown(
            f'<div style="color:{AMBER};font-family:Consolas;margin-bottom:6px">'
            f'FEATURE DRIFT MONITOR (KS test: first half vs recent half)</div>',
            unsafe_allow_html=True,
        )

        with st.spinner("Computing drift statistics..."):
            drift_df = _compute_drift(ticker)

        if drift_df is not None:
            n_drifted = int((drift_df["Status"] == "DRIFT").sum())
            pct_drift = n_drifted / len(drift_df) * 100
            col1, col2 = st.columns(2)
            col1.metric("Features Drifted", f"{n_drifted}/{len(drift_df)}")
            col2.metric("Drift Rate", f"{pct_drift:.0f}%",
                        delta=f"{'⚠ High' if pct_drift > 40 else 'OK'}")

            def _color_status(val: str) -> str:
                return f"color:{RED}" if val == "DRIFT" else f"color:{GREEN}"

            try:
                styled = drift_df.style.map(_color_status, subset=["Status"])
            except AttributeError:
                styled = drift_df.style.applymap(_color_status, subset=["Status"])
            st.dataframe(styled, use_container_width=True, hide_index=True)
        else:
            st.info("Download data first: `python scripts/fetch_data.py`")

    # ── Tab 3: Audit Log ─────────────────────────────────────────────────────
    with tab3:
        st.markdown(
            f'<div style="color:{AMBER};font-family:Consolas;margin-bottom:6px">'
            f'TRADE DECISION AUDIT LOG</div>',
            unsafe_allow_html=True,
        )
        try:
            from src.explainability.audit_log import AuditLog
            log = AuditLog()
            df  = log.read()
            if df.empty:
                st.info("No audit records yet. Records appear here after paper trading starts.\n\n"
                        "Run: `python scripts/start_paper_trading.py`")
            else:
                st.success(f"{len(df)} audit records loaded.")
                st.dataframe(df.tail(200), use_container_width=True)
        except Exception as e:
            st.info("Audit log not yet populated. Start paper trading to generate records.")


if __name__ == "__main__":
    render()
