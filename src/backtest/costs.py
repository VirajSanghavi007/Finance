from __future__ import annotations

import math


def transaction_cost(
    trade_value: float,
    asset_type: str,
    realized_vol: float,
    avg_daily_volume: float,
    trade_size_shares: float = 0.0,
) -> float:
    """
    Realistic transaction cost model for a retail algorithmic trader.

    Parameters
    ----------
    trade_value       : dollar value of the trade
    asset_type        : "equity" or "crypto"
    realized_vol      : recent realized daily volatility (annualised)
    avg_daily_volume  : average daily dollar volume of the asset
    trade_size_shares : number of shares (for FINRA TAF, equity only)

    Returns
    -------
    Total cost in dollars. Always > 0 for any non-zero trade_value.
    """
    if trade_value <= 0:
        return 0.0

    if asset_type == "equity":
        commission    = max(1.00, trade_value * 0.0005)
        slippage      = trade_value * 0.0003 * (1 + realized_vol / 0.015)
        mkt_impact    = (trade_value * 0.0001 *
                         math.sqrt(trade_value / max(avg_daily_volume, 1.0)))
        sec_fee       = trade_value * 0.0000229
        finra_taf     = min(5.95, trade_size_shares * 0.000119)

    elif asset_type == "crypto":
        commission    = trade_value * 0.001
        slippage      = trade_value * 0.0005 * (1 + realized_vol / 0.025)
        mkt_impact    = (trade_value * 0.0002 *
                         math.sqrt(trade_value / max(avg_daily_volume, 1.0)))
        sec_fee       = 0.0
        finra_taf     = 0.0

    else:
        # Generic fallback
        commission    = trade_value * 0.001
        slippage      = trade_value * 0.0003
        mkt_impact    = 0.0
        sec_fee       = 0.0
        finra_taf     = 0.0

    total = commission + slippage + mkt_impact + sec_fee + finra_taf
    # Guarantee cost > 0 for any real trade
    return max(total, trade_value * 0.0001)
