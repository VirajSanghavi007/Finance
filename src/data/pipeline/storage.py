from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config.constants import PROCESSED_DIR, DATA_DIR
from src.config.logging_config import get_logger

logger = get_logger(__name__)

DB_PATH = DATA_DIR / "metadata.db"


# ─── Parquet helpers ─────────────────────────────────────────────────────────

def save_processed(df: pd.DataFrame, name: str, subdir: str = "") -> Path:
    out_dir = PROCESSED_DIR / subdir if subdir else PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.parquet"
    df.to_parquet(path, compression="snappy")
    logger.debug("saved_parquet", path=str(path), rows=len(df))
    return path


def load_processed(name: str, subdir: str = "") -> Optional[pd.DataFrame]:
    in_dir = PROCESSED_DIR / subdir if subdir else PROCESSED_DIR
    path = in_dir / f"{name}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def save_features(df: pd.DataFrame, ticker: str) -> Path:
    return save_processed(df, ticker, subdir="features")


def load_features(ticker: str) -> Optional[pd.DataFrame]:
    return load_processed(ticker, subdir="features")


# ─── SQLite metadata store ────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS download_status (
            ticker      TEXT PRIMARY KEY,
            asset_type  TEXT,
            rows        INTEGER,
            date_start  TEXT,
            date_end    TEXT,
            updated_at  TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS live_state (
            key         TEXT PRIMARY KEY,
            value       TEXT,
            updated_at  TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS trade_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT,
            ticker      TEXT,
            direction   TEXT,
            quantity    REAL,
            price       REAL,
            cost        REAL,
            model       TEXT,
            signal      INTEGER,
            confidence  REAL
        );
    """)
    conn.commit()
    conn.close()


def upsert_download_status(
    ticker: str, asset_type: str, rows: int,
    date_start: str, date_end: str,
) -> None:
    conn = _get_conn()
    conn.execute("""
        INSERT INTO download_status (ticker, asset_type, rows, date_start, date_end)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(ticker) DO UPDATE SET
            rows=excluded.rows, date_start=excluded.date_start,
            date_end=excluded.date_end, updated_at=datetime('now')
    """, (ticker, asset_type, rows, date_start, date_end))
    conn.commit()
    conn.close()


def get_live_state(key: str) -> Optional[str]:
    conn = _get_conn()
    row = conn.execute("SELECT value FROM live_state WHERE key=?", (key,)).fetchone()
    conn.close()
    return row["value"] if row else None


def set_live_state(key: str, value: str) -> None:
    conn = _get_conn()
    conn.execute("""
        INSERT INTO live_state (key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=datetime('now')
    """, (key, value))
    conn.commit()
    conn.close()


def log_trade(
    timestamp: str, ticker: str, direction: str,
    quantity: float, price: float, cost: float,
    model: str, signal: int, confidence: float,
) -> None:
    conn = _get_conn()
    conn.execute("""
        INSERT INTO trade_log
        (timestamp, ticker, direction, quantity, price, cost, model, signal, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (timestamp, ticker, direction, quantity, price, cost, model, signal, confidence))
    conn.commit()
    conn.close()


def load_trade_log() -> pd.DataFrame:
    conn = _get_conn()
    df = pd.read_sql("SELECT * FROM trade_log ORDER BY timestamp", conn)
    conn.close()
    return df
