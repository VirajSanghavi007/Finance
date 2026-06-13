"""Settings page -- configuration, training status, live signal summary."""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_root = _Path(__file__).resolve().parents[3]
if str(_root) not in _sys.path:
    _sys.path.insert(0, str(_root))

from collections import defaultdict

import pandas as pd
import streamlit as st

from src.dashboard.theme import AMBER, GREEN, RED, SURFACE, BORDER, MUTED


def _get_api_status() -> dict[str, bool]:
    try:
        from src.config.settings import Settings
        settings = Settings()
        sources  = settings.available_sources()
        return {
            "Alpaca (Paper Trading)": "alpaca" in sources,
            "NewsAPI (Sentiment)":    "newsapi" in sources,
            "FRED (Macro)":           "fred" in sources,
            "Alpha Vantage":          "alpha_vantage" in sources,
            "Finnhub (Sentiment)":    "finnhub" in sources,
        }
    except Exception:
        return {}


def _get_training_status() -> tuple[list[str], dict]:
    try:
        from src.models.registry import ModelRegistry
        reg   = ModelRegistry()
        names = reg.list_models()
        by_t: dict = defaultdict(list)
        for name in names:
            parts = name.rsplit("_", 1)
            if len(parts) == 2:
                by_t[parts[0]].append(parts[1])
        complete = sorted([t for t, v in by_t.items() if len(v) >= 3])
        return complete, dict(by_t)
    except Exception:
        return [], {}


def _get_latest_signals() -> list[dict]:
    try:
        from src.dashboard.predictions import get_trained_tickers, get_latest_signal
        return [get_latest_signal(t) for t in get_trained_tickers()]
    except Exception:
        return []


def render():
    st.markdown(
        f'<h2 style="color:{AMBER};font-family:Consolas;letter-spacing:.12em">SETTINGS</h2>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["API Keys", "Training Status", "Live Signals"])

    # ── Tab 1: API Keys ──────────────────────────────────────────────────────
    with tab1:
        st.markdown(
            f'<div style="color:{AMBER};font-family:Consolas;font-weight:700;margin-bottom:8px">'
            f'API KEY STATUS</div>',
            unsafe_allow_html=True,
        )
        api_status = _get_api_status()
        key_notes = {
            "Alpaca": "ALPACA_API_KEY, ALPACA_SECRET_KEY",
            "NewsAPI": "NEWS_API_KEY",
            "Alpha": "ALPHA_VANTAGE_KEY",
            "Finnhub": "FINNHUB_KEY",
        }
        for source, has_key in api_status.items():
            color = GREEN if has_key else RED
            label = "✓ Configured" if has_key else "✗ Missing"
            note  = ""
            if not has_key:
                for k, v in key_notes.items():
                    if k in source:
                        note = f" → add {v} to .env"
                        break
            st.markdown(
                f'<div style="background:{SURFACE};border:1px solid {BORDER};'
                f'padding:8px 12px;margin-bottom:4px;font-family:Consolas;font-size:12px">'
                f'<span style="color:{AMBER}">{source}</span>  '
                f'<span style="color:{color}">{label}</span>'
                f'<span style="color:{MUTED}">{note}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown(
            f'<div style="color:{AMBER};font-family:Consolas;font-weight:700;margin-bottom:8px">'
            f'RISK LIMITS</div>',
            unsafe_allow_html=True,
        )
        try:
            from src.config import constants as C
            limits = {
                "MAX_SINGLE_POSITION":    f"{C.MAX_SINGLE_POSITION:.0%}",
                "MAX_PORTFOLIO_DRAWDOWN": f"{C.MAX_PORTFOLIO_DRAWDOWN:.0%}",
                "MAX_DAILY_LOSS":         f"{C.MAX_DAILY_LOSS:.0%}",
                "MAX_GROSS_EXPOSURE":     f"{C.MAX_GROSS_EXPOSURE:.1f}×",
                "VOL_TARGET":             f"{C.VOL_TARGET:.0%}",
                "TRADING_DAYS":           "252",
            }
            for k, v in limits.items():
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'font-family:Consolas;font-size:12px;padding:3px 0;'
                    f'border-bottom:1px solid {BORDER}">'
                    f'<span style="color:{MUTED}">{k}</span>'
                    f'<span style="color:#E0E0E0">{v}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        except Exception as e:
            st.warning(f"Could not load risk constants: {e}")

    # ── Tab 2: Training Status ───────────────────────────────────────────────
    with tab2:
        complete, by_t = _get_training_status()
        total = 25
        st.progress(len(complete) / total,
                    text=f"{len(complete)}/{total} tickers fully trained")

        try:
            from src.config.constants import PROJECT_ROOT
            stacker_dir = PROJECT_ROOT / "data" / "models"
            stackers = {f.stem.replace("_stacker", "") for f in stacker_dir.glob("*_stacker.pkl")}
        except Exception:
            stackers = set()

        rows = []
        for t in sorted(by_t.keys()):
            mods = by_t[t]
            rows.append({
                "Ticker":   t,
                "RF":       "✓" if "rf"   in mods else "—",
                "XGB":      "✓" if "xgb"  in mods else "⏳",
                "LGBM":     "✓" if "lgbm" in mods else ("⏳" if len(mods) >= 2 else "—"),
                "Stacker":  "✓" if t in stackers else "—",
                "Done":     "✓" if len(mods) >= 3 else f"{len(mods)}/3",
            })

        def _color_done(val):
            if "✓" in str(val) and "/" not in str(val):
                return f"color:{GREEN};font-weight:700"
            if "/" in str(val):
                return f"color:{AMBER}"
            return ""

        try:
            styled = pd.DataFrame(rows).style.map(_color_done, subset=["Done"])
        except AttributeError:
            styled = pd.DataFrame(rows).style.applymap(_color_done, subset=["Done"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        st.markdown(
            f'<div style="color:{MUTED};font-family:Consolas;font-size:11px;margin-top:8px">'
            f'Refresh with: <code>python scripts/train_models.py --register</code> → '
            f'<code>python scripts/build_ensemble.py</code> → '
            f'<code>python scripts/run_full_backtest.py</code>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Tab 3: Live Signals ──────────────────────────────────────────────────
    with tab3:
        st.markdown(
            f'<div style="color:{AMBER};font-family:Consolas;font-weight:700;margin-bottom:8px">'
            f'CURRENT MODEL SIGNALS</div>',
            unsafe_allow_html=True,
        )
        with st.spinner("Loading predictions..."):
            sigs = _get_latest_signals()

        if not sigs:
            st.info("No trained models yet.")
        else:
            sig_df = pd.DataFrame(sigs)
            sig_df["Direction"]  = sig_df["signal"].map({1: "▲ LONG", 0: "— FLAT", -1: "▼ SHORT"})
            sig_df["Confidence"] = sig_df["confidence"].map(lambda v: f"{v:.1%}")
            sig_df["HiConf"]     = sig_df["confidence"].map(lambda v: "✓" if v >= 0.62 else "")
            display = sig_df[["ticker", "Direction", "Confidence", "HiConf", "source"]].rename(
                columns={"ticker": "Ticker", "source": "Model"}
            )

            def _color_dir(val):
                if "LONG"  in str(val): return f"color:{GREEN};font-weight:700"
                if "SHORT" in str(val): return f"color:{RED};font-weight:700"
                return f"color:{AMBER}"

            try:
                styled2 = display.style.map(_color_dir, subset=["Direction"])
            except AttributeError:
                styled2 = display.style.applymap(_color_dir, subset=["Direction"])
            st.dataframe(styled2, use_container_width=True, hide_index=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("▲ LONG",  int((sig_df["signal"] == 1).sum()))
            c2.metric("— FLAT",  int((sig_df["signal"] == 0).sum()))
            c3.metric("▼ SHORT", int((sig_df["signal"] == -1).sum()))
            c4.metric("High-Conf", int((sig_df["confidence"] >= 0.62).sum()))


if __name__ == "__main__":
    render()
