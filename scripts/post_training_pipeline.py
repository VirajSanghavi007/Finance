"""
Post-training automation pipeline.
Run after all models are trained to:
  1. Build stacker ensembles for all newly-trained tickers
  2. Run WFO backtest across all registered tickers
  3. Print consolidated report

Usage: python scripts/post_training_pipeline.py
       python scripts/post_training_pipeline.py --min-models 2   (build ensemble if >= 2 base models)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.logging_config import configure_logging, get_logger
from src.models.registry import ModelRegistry

configure_logging()
logger = get_logger("post_training")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-models", type=int, default=2,
                        help="Min base models per ticker to build ensemble")
    args = parser.parse_args()

    registry = ModelRegistry()
    models   = registry.list_models()

    # Group by ticker
    by_ticker: dict = defaultdict(list)
    for name in models:
        parts = name.rsplit("_", 1)
        if len(parts) == 2:
            by_ticker[parts[0]].append(parts[1])

    complete = [t for t, v in by_ticker.items() if len(v) >= 3]
    partial  = [(t, len(v)) for t, v in by_ticker.items() if 0 < len(v) < 3]

    print(f"\nAlgoTrade-X Post-Training Pipeline")
    print(f"  Tickers with all 3 models: {len(complete)}")
    print(f"  Tickers in progress: {len(partial)}")
    if partial:
        for t, n in partial:
            print(f"    {t}: {n}/3 models")
    print()

    if not complete:
        print("No fully trained tickers. Run train_models.py first.")
        return

    # 1. Build ensembles
    print(f"Step 1: Building stacker ensembles for {len(complete)} tickers...")
    ensemble_tickers = [
        t for t, v in by_ticker.items()
        if len(v) >= args.min_models
    ]
    result = subprocess.run(
        [sys.executable, "scripts/build_ensemble.py",
         "--tickers"] + ensemble_tickers,
        capture_output=False,
        text=True,
    )
    if result.returncode != 0:
        print("  Ensemble build had errors but continuing...")

    # 2. Run WFO backtest
    print(f"\nStep 2: Running WFO backtest for {len(complete)} tickers...")
    result = subprocess.run(
        [sys.executable, "scripts/run_full_backtest.py",
         "--tickers"] + complete,
        capture_output=False,
        text=True,
    )
    if result.returncode != 0:
        print("  Backtest had errors.")
        return

    print("\nPost-training pipeline complete.")
    print("  - Ensemble stackers: data/models/*_stacker.pkl")
    print("  - Backtest results:  data/backtest_results/wfo_*.json")
    print("  - Dashboard: streamlit run src/dashboard/app.py")


if __name__ == "__main__":
    main()
