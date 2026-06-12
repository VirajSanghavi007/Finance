from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from src.backtest.engine  import BacktestEngine
from src.backtest.metrics import compute_all_metrics
from src.config.constants import (
    WFO_TRAIN_DAYS, WFO_TEST_DAYS, WFO_STEP_DAYS, WFO_MIN_HISTORY, TRADING_DAYS
)
from src.config.logging_config import get_logger

logger = get_logger(__name__)

TrainFn  = Callable[[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DatetimeIndex], Any]
SignalFn = Callable[[dict[str, pd.DataFrame], pd.Timestamp], dict[str, tuple[int, float]]]


@dataclass
class WFOFold:
    fold_id:      int
    train_start:  pd.Timestamp
    train_end:    pd.Timestamp
    test_start:   pd.Timestamp
    test_end:     pd.Timestamp
    metrics:      dict = field(default_factory=dict)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    trades:       pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class WFOResult:
    folds:        list[WFOFold]
    aggregated:   dict
    full_equity:  pd.Series


def _generate_folds(
    all_dates: pd.DatetimeIndex,
    train_days: int  = WFO_TRAIN_DAYS,
    test_days:  int  = WFO_TEST_DAYS,
    step_days:  int  = WFO_STEP_DAYS,
    min_history: int = WFO_MIN_HISTORY,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    folds = []
    start = all_dates[0]
    i = 0

    while True:
        train_start = start
        # Find index after min_history days from start
        train_end_idx = None
        for j, d in enumerate(all_dates):
            if (d - train_start).days >= train_days:
                train_end_idx = j
                break
        if train_end_idx is None:
            break

        train_end  = all_dates[train_end_idx - 1]
        test_start = all_dates[train_end_idx]

        test_end_idx = None
        for j in range(train_end_idx, len(all_dates)):
            if (all_dates[j] - test_start).days >= test_days:
                test_end_idx = j
                break
        if test_end_idx is None:
            test_end = all_dates[-1]
        else:
            test_end = all_dates[test_end_idx]

        # Require minimum history before first fold
        if (train_end - all_dates[0]).days >= min_history:
            folds.append((train_start, train_end, test_start, test_end))

        # Advance by step_days
        next_start_candidates = [d for d in all_dates if (d - start).days >= step_days]
        if not next_start_candidates:
            break
        start = next_start_candidates[0]
        i += 1

        if test_end >= all_dates[-1]:
            break

    return folds


class WalkForwardOptimizer:
    """
    Walk-Forward Optimization orchestrator.

    Protocol:
      1. Split full history into expanding/rolling train + fixed test windows
      2. For each fold:
         a. Train all models on train window
         b. Run backtest on test window (no data from outside train window)
         c. Record metrics
      3. Aggregate across all folds
    """

    def __init__(
        self,
        train_fn:  TrainFn,
        signal_fn_factory: Callable[[Any], SignalFn],
        initial_capital: float = 100_000.0,
        risk_free_rate:  float = 0.045,
        train_days:  int = WFO_TRAIN_DAYS,
        test_days:   int = WFO_TEST_DAYS,
        step_days:   int = WFO_STEP_DAYS,
        min_history: int = WFO_MIN_HISTORY,
    ) -> None:
        self.train_fn            = train_fn
        self.signal_fn_factory   = signal_fn_factory
        self.initial_capital     = initial_capital
        self.risk_free_rate      = risk_free_rate
        self.train_days          = train_days
        self.test_days           = test_days
        self.step_days           = step_days
        self.min_history         = min_history

    def run(
        self,
        price_data:   dict[str, pd.DataFrame],
        feature_data: dict[str, pd.DataFrame],
    ) -> WFOResult:
        # Build common date index
        all_dates = sorted(set.union(*[set(df.index) for df in price_data.values()]))
        date_idx  = pd.DatetimeIndex(all_dates)

        fold_specs = _generate_folds(
            date_idx, self.train_days, self.test_days,
            self.step_days, self.min_history,
        )

        if not fold_specs:
            raise ValueError(
                f"Not enough history to generate WFO folds. "
                f"Need at least {self.min_history} days, "
                f"got {(date_idx[-1] - date_idx[0]).days}."
            )

        logger.info("wfo_start", n_folds=len(fold_specs))
        folds: list[WFOFold] = []
        all_equity_pieces:    list[pd.Series] = []

        for fold_id, (train_start, train_end, test_start, test_end) in enumerate(fold_specs):
            logger.info(
                "wfo_fold", fold=fold_id,
                train=f"{train_start.date()} → {train_end.date()}",
                test=f"{test_start.date()} → {test_end.date()}",
            )

            # ── Slice to train window (strict: no data outside train) ─────────
            train_mask = lambda df: df[(df.index >= train_start) & (df.index <= train_end)]
            price_train = {t: train_mask(df) for t, df in price_data.items()}
            feat_train  = {t: train_mask(df) for t, df in feature_data.items()}

            # ── Normalise features: compute scaler on TRAIN only ──────────────
            scalers = {}
            for t, df in feat_train.items():
                num_cols = df.select_dtypes(include="number").columns.tolist()
                mean = df[num_cols].mean()
                std  = df[num_cols].std().replace(0, 1)
                scalers[t] = (mean, std)

            def _scale(feat_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
                if ticker not in scalers:
                    return feat_df
                mean, std = scalers[ticker]
                num_cols  = [c for c in mean.index if c in feat_df.columns]
                out = feat_df.copy()
                out[num_cols] = (out[num_cols] - mean[num_cols]) / std[num_cols]
                return out

            feat_train_scaled = {t: _scale(df, t) for t, df in feat_train.items()}

            # ── Train models ──────────────────────────────────────────────────
            train_date_idx = date_idx[(date_idx >= train_start) & (date_idx <= train_end)]
            model_artifacts = self.train_fn(price_train, feat_train_scaled, train_date_idx)
            signal_fn = self.signal_fn_factory(model_artifacts)

            # ── Backtest on test window using TRAIN scalers ───────────────────
            price_test = {t: df[(df.index >= test_start) & (df.index <= test_end)]
                          for t, df in price_data.items()}
            feat_test  = {t: _scale(
                              df[(df.index >= test_start) & (df.index <= test_end)], t
                          )
                          for t, df in feature_data.items()}

            engine = BacktestEngine(
                initial_capital=self.initial_capital,
                risk_free_rate=self.risk_free_rate,
            )
            result = engine.run(price_test, feat_test, signal_fn,
                                start=str(test_start.date()),
                                end=str(test_end.date()))

            equity = result["equity_curve"]
            trades = result["trades"]
            metrics = compute_all_metrics(equity, trades, self.risk_free_rate)

            fold = WFOFold(
                fold_id=fold_id,
                train_start=train_start, train_end=train_end,
                test_start=test_start, test_end=test_end,
                metrics=metrics,
                equity_curve=equity,
                trades=trades,
            )
            folds.append(fold)
            if not equity.empty:
                all_equity_pieces.append(equity)

            logger.info(
                "wfo_fold_done", fold=fold_id,
                sharpe=f"{metrics.get('sharpe_ratio', 0):.2f}",
                max_dd=f"{metrics.get('max_drawdown', 0):.2%}",
                trades=metrics.get("total_trades", 0),
            )

        # ── Aggregate ─────────────────────────────────────────────────────────
        agg = _aggregate_metrics(folds)

        # Stitch equity curves
        full_equity = pd.concat(all_equity_pieces) if all_equity_pieces else pd.Series()

        logger.info(
            "wfo_complete",
            mean_sharpe=f"{agg.get('mean_sharpe', 0):.2f}",
            mean_max_dd=f"{agg.get('mean_max_dd', 0):.2%}",
        )

        return WFOResult(folds=folds, aggregated=agg, full_equity=full_equity)


def _aggregate_metrics(folds: list[WFOFold]) -> dict:
    key_metrics = [
        "sharpe_ratio", "sortino_ratio", "calmar_ratio",
        "max_drawdown", "win_rate", "total_trades", "total_return",
        "profit_factor", "cagr",
    ]
    agg: dict = {}
    for key in key_metrics:
        vals = [f.metrics.get(key) for f in folds if f.metrics.get(key) is not None]
        if vals:
            agg[f"mean_{key.replace('_ratio', '')}"] = float(np.mean(vals))
            agg[f"std_{key.replace('_ratio', '')}"]  = float(np.std(vals))
    agg["n_folds"] = len(folds)
    return agg
