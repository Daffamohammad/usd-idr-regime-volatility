from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import add_optional_macro_features, build_daily_features
from src.garch_modeling import ewma_one_step_variance_forecasts, qlike


def _market_frame(rows: int = 1_050) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=rows)
    return pd.DataFrame(
        {
            "date": dates,
            "usd_idr": 14_000 + np.arange(rows, dtype=float),
            "us10y_yield": 1.5 + np.arange(rows, dtype=float) / 100,
        }
    )


def test_return_lag_one_is_prior_observation():
    feature_frame = build_daily_features(_market_frame())
    first = feature_frame.iloc[0]
    source = _market_frame()
    current_position = source.index[source["date"] == first["date"]][0]
    expected = 100 * np.log(source.loc[current_position - 1, "usd_idr"] / source.loc[current_position - 2, "usd_idr"])
    assert first["return_lag_1"] == pytest.approx(expected)


def test_asof_macro_join_never_uses_future_observation(tmp_path: Path):
    base = build_daily_features(_market_frame())
    macro = pd.DataFrame(
        {
            "effective_date": [base.loc[100, "date"]],
            "bi_rate_pct": [5.0],
        }
    )
    macro.to_csv(tmp_path / "bi_rate.csv", index=False)
    joined = add_optional_macro_features(base, tmp_path)
    effective_date = base.loc[100, "date"]
    assert joined.loc[joined["date"] < effective_date, "bi_rate_pct"].isna().all()
    assert joined.loc[joined["date"] >= effective_date, "bi_rate_pct"].eq(5.0).all()


def test_qlike_is_finite_for_positive_variances():
    actual = pd.Series([0.01, 0.02, 0.03])
    forecast = pd.Series([0.02, 0.02, 0.02])
    assert np.isfinite(qlike(actual, forecast))


def test_ewma_forecast_is_available_before_the_target_return():
    returns = pd.Series(
        [1.0, 2.0, 3.0, 10.0],
        index=pd.date_range("2024-01-01", periods=4, freq="B"),
    )
    forecasts = ewma_one_step_variance_forecasts(returns, split_index=3, decay=0.5)
    # The first test forecast uses only 1%, 2%, and 3% training returns.
    expected = 0.5 * (0.5 * 1.0**2 + 0.5 * 2.0**2) + 0.5 * 3.0**2
    assert forecasts.iloc[0] == pytest.approx(expected)
