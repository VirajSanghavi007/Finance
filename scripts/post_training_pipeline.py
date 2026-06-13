"""
Post-training automation pipeline.
Run after all models are trained to:
  1. Build stacker ensembles for all newly-trained tickers
  2. Run WFO backtest across all registered tickers
  3. Print consolidated report

Usage:
  python scripts/post_training_pipeline.py              # run once
  python scripts/post_training_pipeline.py --watch      # poll every 60s, auto-process new tickers
  python scripts/post_training_pipeline.py --min-models 2  # ensemble with >= 2 base models
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.logging_config import configure_logging, get_logger
from src.models.registry import ModelRegistry

configure_logging()
logger = get_logger("post_training")


def _get_complete_tickers(min_models: int = 3) -> list[str]:
    """Return tickers with >= min_models base models registered."""
    registry = ModelRegistry()
    models   = registry.list_models()
    by_ticker: dict = defaultdict(list)
    for name in models:
        parts = name.rsplit("_", 1)
        if len(parts) == 2:
            by_ticker[parts[0]].append(parts[1])
    return sorted(t for t, v in by_ticker.items() if len(v) >= min_models)


def _run_pipeline_for_tickers(tickers: list[str], min_models: int) -> None:
    """Build ensembles + run backtest for given tickers."""
    if not tickers:
        return

    print(f"\nStep 1: Building stacker ensembles for {tickers}...")
    subprocess.run(
        [sys.executable, "scripts/build_ensemble.py", "--tickers"] + tickers,
        capture_output=False, text=True,
    )

    print(f"\nStep 2: Running WFO backtest for {tickers}...")
    subprocess.run(
        [sys.executable, "scripts/run_full_backtest.py", "--tickers"] + tickers,
        capture_output=False, text=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-models", type=int, default=2,
                        help="Min base models per ticker to build ensemble")
    parser.add_argument("--watch", action="store_true",
                        help="Poll registry every 60s and auto-process new complete tickers")
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

    if args.watch:
        # Watch mode: poll every 60s and process new tickers as they complete
        print(f"\nWatch mode: polling registry every 60s for new complete tickers...")
        processed: set[str] = set()
        TOTAL_EXPECTED = 25

        while True:
            all_complete = _get_complete_tickers(min_models=3)
            new_tickers  = [t for t in all_complete if t not in processed]

            if new_tickers:
                print(f"\n[{time.strftime('%H:%M:%S')}] New tickers complete: {new_tickers}")
                _run_pipeline_for_tickers(new_tickers, args.min_models)
                processed.update(new_tickers)
                print(f"  Progress: {len(processed)}/{TOTAL_EXPECTED} tickers processed")

            if len(processed) >= TOTAL_EXPECTED:
                print(f"\nAll {TOTAL_EXPECTED} tickers processed. Watch mode complete.")
                break

            print(f"[{time.strftime('%H:%M:%S')}] Waiting 60s... ({len(processed)}/{TOTAL_EXPECTED} done)", end="\r")
            time.sleep(60)
        return

    # One-shot mode
    if not complete:
        print("No fully trained tickers. Run train_models.py first.")
        return

    # Build ensembles + run backtest
    ensemble_tickers = [
        t for t, v in by_ticker.items()
        if len(v) >= args.min_models
    ]
    _run_pipeline_for_tickers(ensemble_tickers, args.min_models)

    print("\nPost-training pipeline complete.")
    print("  - Ensemble stackers: data/models/*_stacker.pkl")
    print("  - Backtest results:  data/backtest_results/wfo_*.json")
    print("  - Dashboard: streamlit run src/dashboard/app.py")


if __name__ == "__main__":
    main()
