"""
Fetch and cache market data for all assets in EQUITY_UNIVERSE + CRYPTO_UNIVERSE.
Usage: python scripts/fetch_data.py [--skip-crypto] [--skip-news] [--force-refresh]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.logging_config import configure_logging, get_logger
from src.config.constants import DATA_DIR

configure_logging()
logger = get_logger("fetch_data")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-crypto",  action="store_true")
    parser.add_argument("--skip-news",    action="store_true")
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args()

    from src.data.bootstrap.downloader import run_full_download, print_summary
    from src.data.pipeline.storage import init_db

    init_db()
    print("Fetching all market data...")
    summary = run_full_download(
        skip_news=args.skip_news,
        skip_sec=True,        # SEC is slow — skip by default
        force_refresh=args.force_refresh,
    )
    print_summary(summary)

    # Run feature engineering on everything downloaded
    ok_tickers = [t for t, n in summary["equities"].items() if n > 0]
    if ok_tickers:
        print(f"\nRunning feature engineering on {len(ok_tickers)} tickers...")
        from src.data.features.engineer import run_all_features
        results = run_all_features(equity_tickers=ok_tickers, save=True)
        print(f"Features saved for {len(results)} assets.")
    else:
        print("No equity data downloaded — check internet connection.")


if __name__ == "__main__":
    main()
