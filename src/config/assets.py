from __future__ import annotations

EQUITY_UNIVERSE: dict[str, str] = {
    # US Large Cap
    "SPY":   "S&P 500 ETF — benchmark",
    "QQQ":   "Nasdaq 100 ETF",
    "AAPL":  "Apple",
    "MSFT":  "Microsoft",
    "NVDA":  "NVIDIA",
    "GOOGL": "Alphabet",
    "AMZN":  "Amazon",
    "META":  "Meta",
    "JPM":   "JPMorgan Chase",
    "GS":    "Goldman Sachs",
    "XOM":   "ExxonMobil",
    "BRK-B": "Berkshire Hathaway",
    # Sector ETFs
    "XLF":   "Financials",
    "XLK":   "Technology",
    "XLE":   "Energy",
    "XLV":   "Healthcare",
    "XLI":   "Industrials",
    # Volatility / Macro
    "^VIX":  "CBOE Volatility Index",
    "^TNX":  "10-Year Treasury Yield",
    "GLD":   "Gold ETF",
    "TLT":   "20-Year Treasury ETF",
}

# Tickers safe for OHLCV trading (excludes index-only tickers)
TRADEABLE_EQUITIES: list[str] = [
    t for t in EQUITY_UNIVERSE if not t.startswith("^")
]

CRYPTO_UNIVERSE: dict[str, str] = {
    "BTC/USDT": "Bitcoin",
    "ETH/USDT": "Ethereum",
    "SOL/USDT": "Solana",
    "BNB/USDT": "Binance Coin",
}

MACRO_FRED_SERIES: dict[str, str] = {
    "VIXCLS":   "VIX Close",
    "DFF":      "Fed Funds Rate",
    "GS10":     "10Y Treasury",
    "GS2":      "2Y Treasury",
    "CPIAUCSL": "CPI",
    "UNRATE":   "Unemployment Rate",
    "GDP":      "GDP",
    "T10YIE":   "10Y Breakeven Inflation",
    "DEXUSEU":  "USD/EUR Exchange Rate",
}

# Sector mapping for concentration limits
SECTOR_MAP: dict[str, str] = {
    "SPY":   "broad",
    "QQQ":   "broad",
    "AAPL":  "tech",
    "MSFT":  "tech",
    "NVDA":  "tech",
    "GOOGL": "tech",
    "AMZN":  "tech",
    "META":  "tech",
    "JPM":   "financials",
    "GS":    "financials",
    "XOM":   "energy",
    "BRK-B": "financials",
    "XLF":   "financials",
    "XLK":   "tech",
    "XLE":   "energy",
    "XLV":   "healthcare",
    "XLI":   "industrials",
    "GLD":   "commodities",
    "TLT":   "bonds",
}

ALL_TICKERS: list[str] = list(EQUITY_UNIVERSE.keys()) + list(CRYPTO_UNIVERSE.keys())
