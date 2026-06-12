from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.config.logging_config import get_logger

logger = get_logger(__name__)

OHLCV_COLS = ["open", "high", "low", "close", "volume"]


@dataclass
class ValidationReport:
    ticker: str
    rows: int
    date_start: Optional[str]
    date_end: Optional[str]
    missing_pct: float
    gaps: list[str]
    ohlc_errors: int
    zero_volume_days: int
    passed: bool
    issues: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.ticker}: {self.rows} rows "
            f"{self.date_start} → {self.date_end} | "
            f"missing={self.missing_pct:.1%} gaps={len(self.gaps)} "
            f"ohlc_errors={self.ohlc_errors}"
        )


def validate_ohlcv(df: pd.DataFrame, ticker: str = "", min_rows: int = 252) -> ValidationReport:
    issues: list[str] = []
    gaps: list[str] = []

    if df.empty:
        return ValidationReport(
            ticker=ticker, rows=0, date_start=None, date_end=None,
            missing_pct=1.0, gaps=[], ohlc_errors=0, zero_volume_days=0,
            passed=False, issues=["empty dataframe"],
        )

    rows = len(df)
    date_start = str(df.index[0].date())
    date_end = str(df.index[-1].date())

    # Missing values
    missing_pct = df[OHLCV_COLS].isna().mean().mean() if all(c in df.columns for c in OHLCV_COLS) else 1.0
    if missing_pct > 0.05:
        issues.append(f"high missing rate: {missing_pct:.1%}")

    # Detect gaps > 5 business days
    bday_diff = df.index.to_series().diff().dt.days
    big_gaps = bday_diff[bday_diff > 7]
    for gap_end, gap_days in big_gaps.items():
        gap_str = f"{gap_end.date()} ({int(gap_days)} days)"
        gaps.append(gap_str)

    # OHLC consistency errors
    ohlc_errors = 0
    if all(c in df.columns for c in ["open", "high", "low", "close"]):
        bad = (
            (df["high"] < df["low"]) |
            (df["close"] > df["high"] * 1.01) |
            (df["close"] < df["low"] * 0.99)
        )
        ohlc_errors = int(bad.sum())
        if ohlc_errors > 0:
            issues.append(f"ohlc_inconsistency: {ohlc_errors} rows")

    # Zero volume days
    zero_vol = 0
    if "volume" in df.columns:
        zero_vol = int((df["volume"] == 0).sum())
        if zero_vol / rows > 0.10:
            issues.append(f"high zero_volume rate: {zero_vol}/{rows}")

    # Minimum rows
    if rows < min_rows:
        issues.append(f"too few rows: {rows} < {min_rows}")

    passed = (
        rows >= min_rows and
        missing_pct < 0.05 and
        ohlc_errors == 0 and
        len(gaps) < 10
    )

    return ValidationReport(
        ticker=ticker,
        rows=rows,
        date_start=date_start,
        date_end=date_end,
        missing_pct=missing_pct,
        gaps=gaps,
        ohlc_errors=ohlc_errors,
        zero_volume_days=zero_vol,
        passed=passed,
        issues=issues,
    )


def validate_all(data_map: dict[str, pd.DataFrame]) -> dict[str, ValidationReport]:
    reports: dict[str, ValidationReport] = {}
    for ticker, df in data_map.items():
        report = validate_ohlcv(df, ticker=ticker)
        reports[ticker] = report
        if not report.passed:
            logger.warning("validation_failed", ticker=ticker, issues=report.issues)
        else:
            logger.debug("validation_passed", ticker=ticker, rows=report.rows)
    return reports
