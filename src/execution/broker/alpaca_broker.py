from __future__ import annotations

from src.execution.broker.base_broker import BaseBroker, Order, Position
from src.config.logging_config import get_logger

logger = get_logger(__name__)


class AlpacaBroker(BaseBroker):
    """
    Live broker adapter for Alpaca paper/live trading API.
    Requires ALPACA_API_KEY and ALPACA_SECRET_KEY in .env.
    """

    def __init__(self, api_key: str, secret_key: str, paper: bool = True) -> None:
        try:
            import alpaca_trade_api as tradeapi
            base_url = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
            self._api = tradeapi.REST(api_key, secret_key, base_url)
            logger.info("alpaca_connected", paper=paper)
        except ImportError:
            logger.warning("alpaca_sdk_not_installed")
            self._api = None

    def submit_order(self, order: Order) -> Order:
        if self._api is None:
            order.status = "rejected"
            return order
        try:
            ao = self._api.submit_order(
                symbol=order.ticker,
                qty=int(order.qty),
                side=order.side,
                type=order.order_type,
                time_in_force="day",
            )
            order.order_id = ao.id
            order.status   = "pending"
        except Exception as e:
            logger.error("alpaca_order_failed", error=str(e))
            order.status = "rejected"
        return order

    def cancel_order(self, order_id: str) -> bool:
        if self._api is None:
            return False
        try:
            self._api.cancel_order(order_id)
            return True
        except Exception:
            return False

    def get_positions(self) -> dict[str, Position]:
        if self._api is None:
            return {}
        try:
            positions = {}
            for p in self._api.list_positions():
                positions[p.symbol] = Position(
                    ticker=p.symbol,
                    qty=float(p.qty),
                    avg_price=float(p.avg_entry_price),
                    market_value=float(p.market_value),
                    unrealized_pnl=float(p.unrealized_pl),
                )
            return positions
        except Exception:
            return {}

    def get_portfolio_value(self) -> float:
        if self._api is None:
            return 0.0
        try:
            return float(self._api.get_account().portfolio_value)
        except Exception:
            return 0.0

    def get_cash(self) -> float:
        if self._api is None:
            return 0.0
        try:
            return float(self._api.get_account().cash)
        except Exception:
            return 0.0
