#!/usr/bin/env python3
"""
AlgoTrade-X Bootstrap Entry Point
===================================
Run this first. It will:
  1. Check / prompt for API keys
  2. Download all data (equities, crypto, macro, news, SEC)
  3. Run data quality validation
  4. Print readiness report

Usage:
    python bootstrap.py
    python bootstrap.py --skip-news --skip-sec   # faster first run
    python bootstrap.py --force-refresh          # re-download everything
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from src.config.logging_config import configure_logging, get_logger
from src.config.settings import get_settings
from src.config.constants import DATA_DIR


def check_and_prompt_keys(settings) -> None:
    """Prompt user to add any missing (but useful) API keys."""
    key_info = [
        ("FINNHUB_KEY",       settings.finnhub_key,       "https://finnhub.io/register",            "news & earnings"),
        ("NEWS_API_KEY",      settings.news_api_key,       "https://newsapi.org/register",           "headlines for sentiment"),
        ("ALPHA_VANTAGE_KEY", settings.alpha_vantage_key,  "https://www.alphavantage.co/support/#api-key", "additional indicators"),
        ("ALPACA_API_KEY",    settings.alpaca_api_key,     "https://app.alpaca.markets/signup",      "paper trading"),
    ]
    missing = [(name, url, desc) for name, val, url, desc in key_info if not val]

    if not missing:
        print("  [OK] All optional API keys found in .env")
        return

    print("\n  Optional API keys (all free, takes ~30s each to register):")
    print("  -" * 35)
    env_path = Path(".env")

    for name, url, desc in missing:
        print(f"  [SKIP] {name} not set -- {desc} will be disabled")
        print(f"         Add it later: {url}")

    # Reload settings after updates
    get_settings.cache_clear()  # type: ignore[attr-defined]


def main() -> None:
    parser = argparse.ArgumentParser(description="AlgoTrade-X Bootstrap")
    parser.add_argument("--skip-news",     action="store_true", help="Skip NewsAPI download")
    parser.add_argument("--skip-sec",      action="store_true", help="Skip SEC EDGAR download")
    parser.add_argument("--skip-crypto",   action="store_true", help="Skip crypto download (requires ccxt)")
    parser.add_argument("--force-refresh", action="store_true", help="Re-download all data")
    parser.add_argument("--log-level",     default="INFO")
    args = parser.parse_args()

    configure_logging(log_level=args.log_level, log_file=DATA_DIR / "logs" / "bootstrap.log")
    logger = get_logger("bootstrap")

    print("\n" + "=" * 60)
    print("  AlgoTrade-X  --  Bootstrap")
    print("=" * 60)

    settings = get_settings()
    available = settings.available_sources()
    print(f"\n  Active data sources: {', '.join(available)}")

    # Prompt for missing keys
    check_and_prompt_keys(settings)

    # Run download
    from src.data.bootstrap.downloader import run_full_download, print_summary

    print("\n  Starting data download...")
    summary = run_full_download(
        skip_news=args.skip_news or not settings.news_api_key,
        skip_sec=args.skip_sec,
        force_refresh=args.force_refresh,
    )

    print_summary(summary)

    # Run feature engineering if we have enough data
    eq_ok = sum(1 for v in summary["equities"].values() if v > 0)
    if eq_ok >= 5:
        print("  Running feature engineering on downloaded data...")
        try:
            from src.data.features.engineer import run_all_features
            feat_results = run_all_features()
            print(f"  [OK] Features computed for {len(feat_results)} assets")
        except ImportError:
            print("  Feature engineering modules not yet built -- skipping")
        except Exception as exc:
            logger.warning("feature_eng_failed", error=str(exc))
            print(f"  [WARN] Feature engineering failed: {exc}")
    else:
        print("  [WARN] Not enough equity data for feature engineering -- check API access")

    print("\n  Bootstrap complete. Run:")
    print("    docker-compose up           # full stack")
    print("    streamlit run src/dashboard/app.py  # dashboard only")
    print("    uvicorn src.api.main:app    # API only\n")


if __name__ == "__main__":
    main()
