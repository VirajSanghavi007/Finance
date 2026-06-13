from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from src.backtest.costs    import transaction_cost
from src.backtest.order_book import SimOrderBook
from src.backtest.portfolio  import Portfolio
from src.config.constants    import MAX_SINGLE_POSITION, MIN_TRADE_SIZE
from src.config.logging_config import get_logger

logger = get_logger(__name__)

SignalFn = Callable[[pd.DataFrame, pd.Timestamp], dict[str, tuple[int, float]]]
# SignalFn: (feature_df_up_to_t, date) → {ticker: (signal, confidence)}


class BacktestEngine:
    """
    Event-driven backtester.

    Rules enforced here:
      - Signal at close[t] → fill at open[t+1]  (no bar-of-signal fills)
      - Transaction costs on every fill
      - Position limits from risk manager
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        risk_free_rate: float = 0.045,
        max_positions: int = 10,
    ) -> None:
        self.initial_capital = initial_capital
        self.risk_free_rate  = risk_free_rate
        self.max_positions   = max_positions
        self.portfolio        = Portfolio(initial_capital)
        self.order_book       = SimOrderBook()

    def _position_size(
        self, signal: int, confidence: float,
        asset_type: str, realized_vol: float,
        portfolio_value: float,
    ) -> float:
        """Kelly-inspired vol-targeted position size, clipped to limits."""
        vol_target = 0.15
        vol_scalar = vol_target / max(realized_vol, 0.05)
        raw_size   = 0.5 * vol_scalar * confidence  # half-Kelly × confidence
        clipped    = min(raw_size, MAX_SINGLE_POSITION)
        return max(clipped, MIN_TRADE_SIZE) * portfolio_value

    def run(
        self,
        price_data: dict[str, pd.DataFrame],
        feature_data: dict[str, pd.DataFrame],
        signal_fn: SignalFn,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Run a backtest.

        Parameters
        ----------
        price_data   : {ticker: OHLCV DataFrame}
        feature_data : {ticker: features DataFrame (no future data)}
        signal_fn    : function that generates signals given features up to date t
        """
        # Build aligned date index
        all_dates = sorted(set.union(*[
            set(df.index) for df in price_data.values()
        ]))
        if start:
            all_dates = [d for d in all_dates if d >= pd.Timestamp(start)]
        if end:
            all_dates = [d for d in all_dates if d <= pd.Timestamp(end)]

        logger.info("backtest_start", n_dates=len(all_dates),
                    tickers=list(price_data.keys()))

        # Build vol and ADV maps (rolling 21-day)
        vol_map: dict[str, float] = {}
        adv_map: dict[str, float] = {}

        for i, date in enumerate(all_dates):
            # ── 1. Process overnight fills (at this bar's open) ───────────────
            opens = {
                t: float(df.loc[date, "open"])
                for t, df in price_data.items()
                if date in df.index and not np.isnan(df.loc[date, "open"])
            }

            fills = self.order_book.process_bar(
                date=date,
                open_prices=opens,
                cost_fn=lambda tv, at, vol, adv: transaction_cost(tv, at, vol, adv),
                vol_map=vol_map,
                adv_map=adv_map,
            )

            # Apply fills to portfolio
            for fill in fills:
                t = fill.ticker
                if fill.side == "buy":
                    qty = fill.quantity
                else:
                    qty = -fill.quantity
                port_val = self.portfolio.equity_curve[-1][1] if self.portfolio.equity_curve else self.initial_capital
                self.portfolio.open_position(
                    ticker=t, quantity=qty, price=fill.fill_price,
                    date=date, cost=fill.cost,
                )

            # Close positions that have a sell order
            # (handled by order_book -- in this simplified version we close
            # positions where a new opposing signal was submitted)

            # ── 2. Mark to market at close ────────────────────────────────────
            closes = {
                t: float(df.loc[date, "close"])
                for t, df in price_data.items()
                if date in df.index and not np.isnan(df.loc[date, "close"])
            }
            portfolio_value = self.portfolio.mark_to_market(closes, date)

            # Update rolling vol / ADV
            for t, df in price_data.items():
                if date in df.index:
                    end_idx = df.index.get_loc(date)
                    start_idx = max(0, end_idx - 21)
                    window = df["close"].iloc[start_idx:end_idx + 1]
                    if len(window) > 2:
                        log_rets = np.log(window / window.shift(1)).dropna()
                        vol_map[t] = float(log_rets.std() * np.sqrt(252))
                    vol_avg_vol = df["volume"].iloc[start_idx:end_idx + 1].mean()
                    mid_price   = float(df["close"].iloc[end_idx])
                    adv_map[t]  = float(vol_avg_vol * mid_price)

            # ── 3. Generate signals using features up to (and including) date ─
            signals = signal_fn(feature_data, date)

            # ── 4. Submit orders (fill at next open) ──────────────────────────
            for ticker, (signal, confidence) in signals.items():
                current_pos = self.portfolio.positions.get(ticker)
                current_signal = 1 if (current_pos and current_pos.quantity > 0) else (
                    -1 if (current_pos and current_pos.quantity < 0) else 0
                )

                # Close if signal changed
                if current_pos is not None and signal != current_signal:
                    close_price = closes.get(ticker, current_pos.entry_price)
                    cost = transaction_cost(
                        abs(current_pos.quantity) * close_price,
                        current_pos.asset_type,
                        vol_map.get(ticker, 0.02),
                        adv_map.get(ticker, 1e6),
                    )
                    self.portfolio.close_position(ticker, close_price, date, cost)

                # Open new position if signal is directional
                if signal != 0 and ticker not in self.portfolio.positions:
                    if len(self.portfolio.positions) < self.max_positions:
                        size = self._position_size(
                            signal, confidence,
                            asset_type="equity",
                            realized_vol=vol_map.get(ticker, 0.02),
                            portfolio_value=portfolio_value,
                        )
                        price = closes.get(ticker)
                        if price and price > 0:
                            qty = (size / price) * signal
                            side = "buy" if signal > 0 else "sell"
                            self.order_book.submit_market_order(
                                ticker=ticker, side=side,
                                quantity=abs(qty), created_at=date,
                                signal=signal, confidence=confidence,
                            )

        # Close all remaining positions at last close
        if all_dates:
            last_date = all_dates[-1]
            last_closes = {
                t: float(df.loc[last_date, "close"])
                for t, df in price_data.items()
                if last_date in df.index
            }
            for ticker in list(self.portfolio.positions.keys()):
                price = last_closes.get(ticker, self.portfolio.positions[ticker].entry_price)
                cost  = transaction_cost(
                    abs(self.portfolio.positions[ticker].quantity) * price,
                    "equity",
                    vol_map.get(ticker, 0.02),
                    adv_map.get(ticker, 1e6),
                )
                self.portfolio.close_position(ticker, price, last_date, cost)

        equity_curve = self.portfolio.get_equity_series()
        trades_df    = self.portfolio.get_trades_df()

        logger.info(
            "backtest_complete",
            final_value=float(equity_curve.iloc[-1]) if not equity_curve.empty else 0,
            n_trades=len(trades_df),
        )

        return {
            "equity_curve": equity_curve,
            "trades":       trades_df,
            "portfolio":    self.portfolio,
        }
