"""Feature preparation with explicit information-availability rules."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data_ingestion import validate_market_data


def load_market_snapshot(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"])
    return validate_market_data(frame)


def build_daily_features(market_data: pd.DataFrame) -> pd.DataFrame:
    """Create features available before the return dated at row *t* is known.

    The return at date *t* is the outcome.  All predictive columns are shifted
    by at least one market date, including the change in US 10-year yield.
    """
    df = validate_market_data(market_data).copy()
    # US market holidays create sparse ^TNX values.  The last published yield
    # is the only yield observable on the next Indonesian market date.
    df["us10y_yield"] = df["us10y_yield"].ffill()
    if df["us10y_yield"].isna().any():
        raise ValueError("us10y_yield tidak tersedia pada awal sampel.")
    df["log_return_pct"] = 100 * np.log(df["usd_idr"] / df["usd_idr"].shift(1))
    df["us10y_change_bp"] = 100 * df["us10y_yield"].diff()

    for lag in (1, 2, 5, 10, 20):
        df[f"return_lag_{lag}"] = df["log_return_pct"].shift(lag)
        df[f"abs_return_lag_{lag}"] = df["log_return_pct"].abs().shift(lag)
    df["rolling_vol_5"] = df["log_return_pct"].shift(1).rolling(5).std()
    df["rolling_vol_20"] = df["log_return_pct"].shift(1).rolling(20).std()
    df["us10y_change_bp_lag_1"] = df["us10y_change_bp"].shift(1)
    df["direction_up"] = (df["log_return_pct"] > 0).astype(int)

    return df.dropna(subset=["log_return_pct", "return_lag_20", "rolling_vol_20"]).reset_index(drop=True)


OPTIONAL_MACRO_FILES = {
    "bi_rate.csv": ("effective_date", "bi_rate_pct"),
    "inflation_yoy.csv": ("available_date", "inflation_yoy_pct"),
    "id10y_yield.csv": ("date", "id10y_yield_pct"),
}


def add_optional_macro_features(frame: pd.DataFrame, macro_dir: str | Path) -> pd.DataFrame:
    """As-of join validated macro snapshots when the documented CSVs exist.

    The date field must indicate *public availability*, not merely the economic
    reference month.  This prevents monthly variables from being backfilled into
    daily dates where a forecaster could not yet have observed them.
    """
    result = frame.copy().sort_values("date")
    macro_dir = Path(macro_dir)
    for file_name, (date_column, value_column) in OPTIONAL_MACRO_FILES.items():
        path = macro_dir / file_name
        if not path.exists():
            continue
        macro = pd.read_csv(path, parse_dates=[date_column])
        required = {date_column, value_column}
        if missing := required.difference(macro.columns):
            raise ValueError(f"{path} tidak memiliki kolom wajib: {sorted(missing)}")
        macro = macro[[date_column, value_column]].dropna().sort_values(date_column)
        if macro[date_column].duplicated().any() or (macro[value_column] <= 0).any():
            raise ValueError(f"{path} berisi tanggal ganda atau nilai non-positif.")
        result = pd.merge_asof(result, macro, left_on="date", right_on=date_column, direction="backward")
        result = result.drop(columns=[date_column])
    if {"id10y_yield_pct", "us10y_yield"}.issubset(result.columns):
        result["bond_spread_pct_point"] = result["id10y_yield_pct"] - result["us10y_yield"]
        result["bond_spread_pct_point_lag_1"] = result["bond_spread_pct_point"].shift(1)
    return result


def chronological_split(frame: pd.DataFrame, test_fraction: float = 0.20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split frame into train and test sets while preserving chronological order.
    
    Ensures minimum viable sizes for time-series evaluation: at least 500 training
    observations and 150 test observations.
    """
    if not 0 < test_fraction < 0.5:
        raise ValueError("test_fraction harus berada di antara 0 dan 0.5.")
    
    split = int(len(frame) * (1 - test_fraction))
    train_size = split
    test_size = len(frame) - split
    
    # Ensure minimum viable sizes for time-series evaluation
    if train_size < 500 or test_size < 150:
        raise ValueError("Observasi train/test tidak cukup untuk evaluasi time-series.")
    
    return frame.iloc[:split].copy(), frame.iloc[split:].copy()
