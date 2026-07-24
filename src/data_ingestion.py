"""Download and validate the reproducible market-data snapshot used by the project."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf


DEFAULT_START = "2016-07-01"
USD_IDR_TICKER = "IDR=X"
US_10Y_TICKER = "^TNX"


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a predictable column layout for yfinance's one-ticker output."""
    if isinstance(frame.columns, pd.MultiIndex):
        frame = frame.copy()
        frame.columns = frame.columns.get_level_values(0)
    return frame


def _download_one(ticker: str, start: str, end: str) -> pd.DataFrame:
    frame = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    frame = _flatten_columns(frame)
    if frame.empty or "Close" not in frame:
        raise ValueError(f"Tidak ada data Close untuk ticker {ticker}.")
    return frame[["Close"]].rename(columns={"Close": ticker})


def download_market_snapshot(start: str = DEFAULT_START, end: str | None = None) -> pd.DataFrame:
    """Download daily USD/IDR and US 10Y yield proxy, then align by date.

    `end` is exclusive, matching yfinance.  The market-rate source is Yahoo
    Finance and is stored as a versioned snapshot for notebook reproducibility.
    """
    if end is None:
        end = (date.today() + timedelta(days=1)).isoformat()
    usd_idr = _download_one(USD_IDR_TICKER, start, end)
    us_10y = _download_one(US_10Y_TICKER, start, end)
    panel = usd_idr.join(us_10y, how="left").reset_index().rename(columns={"Date": "date"})
    panel["date"] = pd.to_datetime(panel["date"]).dt.tz_localize(None)
    panel = panel.rename(columns={USD_IDR_TICKER: "usd_idr", US_10Y_TICKER: "us10y_yield"})
    return validate_market_data(panel)


def validate_market_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply essential grain and domain checks for the daily market panel."""
    expected = {"date", "usd_idr", "us10y_yield"}
    missing = expected.difference(frame.columns)
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {sorted(missing)}")
    frame = frame.copy().sort_values("date").reset_index(drop=True)
    if frame["date"].isna().any() or frame["date"].duplicated().any():
        raise ValueError("Kolom date harus terisi dan unik pada grain harian.")
    if frame["usd_idr"].isna().any() or (frame["usd_idr"] <= 0).any():
        raise ValueError("USD/IDR harus terisi dan bernilai positif.")
    if len(frame) < 1_000:
        raise ValueError("Data terlalu pendek; diperlukan sedikitnya 1.000 observasi harian.")
    return frame


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_snapshot(frame: pd.DataFrame, raw_dir: str | Path) -> tuple[Path, Path]:
    """Persist CSV plus compact provenance manifest, returning both paths."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / "yahoo_usd_idr_us10y.csv"
    frame.to_csv(csv_path, index=False)
    manifest = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Yahoo Finance via yfinance",
        "tickers": {"usd_idr": USD_IDR_TICKER, "us10y_yield": US_10Y_TICKER},
        "date_start": str(frame["date"].min().date()),
        "date_end": str(frame["date"].max().date()),
        "rows": int(len(frame)),
        "sha256": _sha256(csv_path),
        "notes": [
            "^TNX is used as a US 10-year yield proxy.",
            "Indonesia 10-year yield is intentionally not inferred from an unverified Yahoo ticker.",
        ],
    }
    manifest_path = raw_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return csv_path, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Unduh snapshot pasar USD/IDR untuk proyek.")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=None, help="Tanggal eksklusif YYYY-MM-DD")
    parser.add_argument("--output-dir", default="data/raw")
    args = parser.parse_args()
    frame = download_market_snapshot(args.start, args.end)
    csv_path, manifest_path = save_snapshot(frame, args.output_dir)
    print(f"Tersimpan: {csv_path} ({len(frame):,} baris)")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
