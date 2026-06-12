from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config.constants import RAW_EQUITIES_DIR, RAW_MACRO_DIR
from src.config.logging_config import get_logger

logger = get_logger(__name__)


def _load_close(ticker: str) -> Optional[pd.Series]:
    safe = ticker.replace("^", "IDX_").replace("/", "_")
    path = RAW_EQUITIES_DIR / f"{safe}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "close" not in df.columns:
        return None
    return df["close"].rename(ticker)


def _load_macro_series(series_id: str) -> Optional[pd.Series]:
    path = RAW_MACRO_DIR / f"{series_id}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if df.empty:
        return None
    return df.iloc[:, 0].rename(series_id)


def compute_cross_asset_features(
    df: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    close = df["close"]
    ret   = np.log(close / close.shift(1))
    out   = pd.DataFrame(index=df.index)

    # Load reference assets
    spy_close = _load_close("SPY")
    vix_close = _load_close("^VIX")
    gld_close = _load_close("GLD")
    tlt_close = _load_close("TLT")

    # Correlations
    if spy_close is not None:
        spy_ret = np.log(spy_close / spy_close.shift(1)).reindex(df.index)
        for w in [21, 63]:
            out[f"ca_corr_spy_{w}"] = ret.rolling(w).corr(spy_ret)
        # Beta to SPY (60-day)
        cov60  = ret.rolling(60).cov(spy_ret)
        var60  = spy_ret.rolling(60).var().replace(0, np.nan)
        out["ca_beta_spy_60"] = cov60 / var60
    else:
        for w in [21, 63]:
            out[f"ca_corr_spy_{w}"] = np.nan
        out["ca_beta_spy_60"] = np.nan

    if vix_close is not None:
        vix_ret = np.log(vix_close / vix_close.shift(1)).reindex(df.index)
        out["ca_corr_vix_21"] = ret.rolling(21).corr(vix_ret)
        out["ca_vix_level"]   = vix_close.reindex(df.index).ffill()
        # VIX regime
        vix = out["ca_vix_level"]
        out["ca_vix_regime"] = pd.cut(
            vix,
            bins=[0, 15, 25, 35, np.inf],
            labels=["low", "mid", "high", "spike"],
        ).astype(str)
    else:
        out["ca_corr_vix_21"] = np.nan
        out["ca_vix_level"]   = np.nan
        out["ca_vix_regime"]  = "unknown"

    if gld_close is not None:
        gld_ret = np.log(gld_close / gld_close.shift(1)).reindex(df.index)
        out["ca_corr_gold_21"] = ret.rolling(21).corr(gld_ret)
    else:
        out["ca_corr_gold_21"] = np.nan

    if tlt_close is not None:
        tlt_ret = np.log(tlt_close / tlt_close.shift(1)).reindex(df.index)
        out["ca_corr_tlt_21"] = ret.rolling(21).corr(tlt_ret)
    else:
        out["ca_corr_tlt_21"] = np.nan

    # Relative strength vs sector ETF
    from src.config.assets import SECTOR_MAP
    sector = SECTOR_MAP.get(ticker)
    sector_etf_map = {
        "tech": "XLK", "financials": "XLF", "energy": "XLE",
        "healthcare": "XLV", "industrials": "XLI",
    }
    sec_etf = sector_etf_map.get(sector or "", None) if sector else None
    if sec_etf and sec_etf != ticker:
        sec_close = _load_close(sec_etf)
        if sec_close is not None:
            sec_ret = np.log(sec_close / sec_close.shift(1)).reindex(df.index)
            out["ca_rel_strength_sector"] = ret.rolling(21).mean() - sec_ret.rolling(21).mean()
        else:
            out["ca_rel_strength_sector"] = np.nan
    else:
        out["ca_rel_strength_sector"] = np.nan

    # Yield curve (10Y - 2Y)
    gs10 = _load_macro_series("GS10")
    gs2  = _load_macro_series("GS2")
    if gs10 is not None and gs2 is not None:
        yc = (gs10 - gs2).reindex(df.index).ffill()
        out["ca_yield_curve"] = yc
    else:
        out["ca_yield_curve"] = np.nan

    # Dollar index proxy (USD/EUR inverse)
    dex = _load_macro_series("DEXUSEU")
    if dex is not None:
        out["ca_dollar_index"] = (1 / dex.reindex(df.index).ffill())
    else:
        out["ca_dollar_index"] = np.nan

    return out
