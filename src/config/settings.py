from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .constants import PROJECT_ROOT


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API Keys (all optional — system degrades gracefully without them)
    alpha_vantage_key: Optional[str] = Field(default=None, alias="ALPHA_VANTAGE_KEY")
    finnhub_key:       Optional[str] = Field(default=None, alias="FINNHUB_KEY")
    fred_api_key:      Optional[str] = Field(default=None, alias="FRED_API_KEY")
    news_api_key:      Optional[str] = Field(default=None, alias="NEWS_API_KEY")
    alpaca_api_key:    Optional[str] = Field(default=None, alias="ALPACA_API_KEY")
    alpaca_secret_key: Optional[str] = Field(default=None, alias="ALPACA_SECRET_KEY")

    # Alpaca endpoints
    alpaca_base_url: str = Field(
        default="https://paper-api.alpaca.markets",
        alias="ALPACA_BASE_URL",
    )

    # App config
    log_level:     str = Field(default="INFO",  alias="LOG_LEVEL")
    redis_url:     str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    api_host:      str = Field(default="0.0.0.0", alias="API_HOST")
    api_port:      int = Field(default=8000,      alias="API_PORT")
    dashboard_port: int = Field(default=8501,     alias="DASHBOARD_PORT")

    # SEC EDGAR user-agent (required by SEC)
    sec_user_agent: str = Field(
        default="AlgoTradeX contact@algotradex.local",
        alias="SEC_USER_AGENT",
    )

    # Backtest initial capital
    initial_capital: float = Field(default=100_000.0, alias="INITIAL_CAPITAL")

    # Risk-free rate for Sharpe (annualised)
    risk_free_rate: float = Field(default=0.045, alias="RISK_FREE_RATE")

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    def available_sources(self) -> list[str]:
        sources = ["yfinance", "fred_basic", "binance_public", "coingecko", "sec_edgar"]
        if self.alpha_vantage_key:
            sources.append("alpha_vantage")
        if self.finnhub_key:
            sources.append("finnhub")
        if self.news_api_key:
            sources.append("newsapi")
        if self.alpaca_api_key and self.alpaca_secret_key:
            sources.append("alpaca")
        return sources


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
