"""
Options-Derived Features

Options markets are forward-looking — "smart money" trades here first.
These features extract the option market's view on a stock:

  opt_iv_atm        — At-the-money implied volatility (30-day)
  opt_iv_skew       — IV skew = IV(put 10-delta) − IV(call 10-delta)
                      Positive skew = market pricing in downside risk
  opt_pcr_vol       — Put-Call Ratio by volume (>1 = bearish, <1 = bullish)
  opt_pcr_oi        — Put-Call Ratio by open interest
  opt_iv_rv_spread  — IV − Realised Vol: positive = expensive options (mean-revert)
  opt_term_spread   — IV(30d) − IV(7d): negative = inverted term structure (fear)

Data source: yfinance options chain (free, no API key needed)
Rate limit: ~2 calls/min per ticker (cached to disk)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config.logging_config import get_logger
from src.config.constants import RAW_DIR

logger = get_logger(__name__)

_OPTIONS_CACHE_DIR = RAW_DIR / "options"
_OPTIONS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_TTL_HOURS   = 4   # re-fetch if older than this


def _cache_path(ticker: str, expiry: str) -> Path:
    return _OPTIONS_CACHE_DIR / f"{ticker}_{expiry}.json"


def _fetch_option_chain(ticker: str) -> Optional[dict]:
    """Fetch nearest two expiries from yfinance with disk caching."""
    try:
        import yfinance as yf
        t      = yf.Ticker(ticker)
        expiry_dates = t.options
        if not expiry_dates:
            return None

        result = {}
        for expiry in expiry_dates[:2]:  # nearest two expiries only
            cpath = _cache_path(ticker, expiry)
            if cpath.exists():
                age_h = (time.time() - cpath.stat().st_mtime) / 3600
                if age_h < _CACHE_TTL_HOURS:
                    with open(cpath) as f:
                        result[expiry] = json.load(f)
                    continue
            try:
                chain = t.option_chain(expiry)
                data  = {
                    "calls": chain.calls[["strike", "impliedVolatility", "volume", "openInterest"]].to_dict("records"),
                    "puts":  chain.puts [["strike", "impliedVolatility", "volume", "openInterest"]].to_dict("records"),
                }
                with open(cpath, "w") as f:
                    json.dump(data, f)
                result[expiry] = data
                time.sleep(0.5)   # gentle rate limit
            except Exception as e:
                logger.warning("options_fetch_failed", ticker=ticker, expiry=expiry, error=str(e))
        return result if result else None
    except Exception as e:
        logger.warning("yfinance_options_unavailable", ticker=ticker, error=str(e))
        return None


def _compute_chain_features(chain: dict, current_price: float) -> dict:
    """Extract IV skew, PCR, ATM-IV from a single expiry chain dict."""
    calls = pd.DataFrame(chain.get("calls", []))
    puts  = pd.DataFrame(chain.get("puts",  []))

    if calls.empty or puts.empty:
        return {}

    features: dict = {}

    # ── ATM implied volatility ─────────────────────────────────────────────
    calls["moneyness"] = (calls["strike"] - current_price).abs()
    puts ["moneyness"] = (puts ["strike"] - current_price).abs()

    atm_call = calls.sort_values("moneyness").iloc[0]
    atm_put  = puts .sort_values("moneyness").iloc[0]
    features["opt_iv_atm"] = float(
        np.nanmean([atm_call["impliedVolatility"], atm_put["impliedVolatility"]])
    )

    # ── IV Skew: OTM put IV − OTM call IV ─────────────────────────────────
    # 10% OTM puts and calls
    otm_put_strike  = current_price * 0.90
    otm_call_strike = current_price * 1.10

    otm_put  = puts [(puts ["strike"] - otm_put_strike ).abs().idxmin()] \
               if len(puts)  > 0 else None
    otm_call = calls[(calls["strike"] - otm_call_strike).abs().idxmin()] \
               if len(calls) > 0 else None

    if otm_put is not None and otm_call is not None:
        skew = float(otm_put["impliedVolatility"]) - float(otm_call["impliedVolatility"])
        features["opt_iv_skew"] = skew
    else:
        features["opt_iv_skew"] = 0.0

    # ── Put-Call Ratio (volume) ────────────────────────────────────────────
    total_call_vol = calls["volume"].fillna(0).sum()
    total_put_vol  = puts ["volume"].fillna(0).sum()
    if total_call_vol > 0:
        features["opt_pcr_vol"] = float(total_put_vol / total_call_vol)
    else:
        features["opt_pcr_vol"] = 1.0

    # ── Put-Call Ratio (open interest) ────────────────────────────────────
    total_call_oi = calls["openInterest"].fillna(0).sum()
    total_put_oi  = puts ["openInterest"].fillna(0).sum()
    if total_call_oi > 0:
        features["opt_pcr_oi"] = float(total_put_oi / total_call_oi)
    else:
        features["opt_pcr_oi"] = 1.0

    return features


def compute_options_features(
    df: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    """
    Compute options-derived features and broadcast to the price DataFrame index.

    Options data updates daily (roughly), so we fetch current snapshot and
    broadcast it as a constant for recent bars. For historical bars we return
    NaN (options history isn't free).

    Returned columns:
      opt_iv_atm, opt_iv_skew, opt_pcr_vol, opt_pcr_oi,
      opt_iv_rv_spread, opt_term_spread
    """
    # Only meaningful for equities with options (skip crypto/macro)
    _options_tickers = {
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META",
        "JPM", "GS", "XOM", "SPY", "QQQ", "GLD", "TLT",
    }
    result = pd.DataFrame(index=df.index)
    opt_cols = ["opt_iv_atm", "opt_iv_skew", "opt_pcr_vol", "opt_pcr_oi",
                "opt_iv_rv_spread", "opt_term_spread"]
    for col in opt_cols:
        result[col] = np.nan

    if ticker not in _options_tickers:
        return result

    chain_data = _fetch_option_chain(ticker)
    if not chain_data:
        return result

    expiries = sorted(chain_data.keys())
    current_price = float(df["close"].iloc[-1]) if "close" in df.columns else 100.0

    # Near-term expiry
    near_feats = _compute_chain_features(chain_data[expiries[0]], current_price) \
                 if len(expiries) >= 1 else {}

    # Far-term expiry (for term structure)
    far_feats  = _compute_chain_features(chain_data[expiries[1]], current_price) \
                 if len(expiries) >= 2 else {}

    # Broadcast to last 5 bars (latest data — older bars get NaN)
    n_fill = min(5, len(result))
    for col, val in near_feats.items():
        result.loc[result.index[-n_fill:], col] = val

    # Term spread = near IV − far IV (inverted = fear)
    if "opt_iv_atm" in near_feats and "opt_iv_atm" in far_feats:
        spread = near_feats["opt_iv_atm"] - far_feats["opt_iv_atm"]
        result.loc[result.index[-n_fill:], "opt_term_spread"] = spread

    # IV vs Realised Vol spread (expensive options = mean-revert)
    if "opt_iv_atm" in near_feats and "close" in df.columns:
        log_ret  = np.log(df["close"] / df["close"].shift(1))
        rv_30    = float(log_ret.tail(30).std() * np.sqrt(252))
        result.loc[result.index[-n_fill:], "opt_iv_rv_spread"] = \
            near_feats["opt_iv_atm"] - rv_30

    logger.info("options_features_computed", ticker=ticker,
                n_features=result.notna().any(axis=0).sum())
    return result
