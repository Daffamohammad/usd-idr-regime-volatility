from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import add_optional_macro_features, build_daily_features, chronological_split
from src.garch_modeling import GarchSpec, qlike, rolling_one_step_variance_forecasts
from src.hamilton_regime import fit_hamilton_smoothed
from src.forecasting import evaluate_direction


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


def test_qlike_raises_on_zero_or_negative():
    actual = pd.Series([0.01, 0.02, 0.03])
    forecast = pd.Series([0.02, 0.02, 0.02])
    with pytest.raises(ValueError, match="strictly positive"):
        qlike(pd.Series([0.0, 0.02, 0.03]), forecast)
    with pytest.raises(ValueError, match="strictly positive"):
        qlike(actual, pd.Series([-0.01, 0.02, 0.02]))


def test_build_daily_features_minimal_rows():
    frame = _market_frame(1_050)
    result = build_daily_features(frame)
    assert len(result) > 0
    assert "log_return_pct" in result.columns
    assert "direction_up" in result.columns


def test_rolling_one_step_variance_forecasts_boundary():
    frame = _market_frame(1_100)
    features = build_daily_features(frame)
    returns = features.set_index("date")["log_return_pct"]
    spec = GarchSpec("GARCH(1,1)", "GARCH", 0)
    forecasts = rolling_one_step_variance_forecasts(returns, 500, spec)
    assert len(forecasts) == 100
    assert forecasts.notna().all()
    assert (forecasts >= 0).all()


def test_fit_hamilton_smoothed_output_shape():
    frame = _market_frame(1_050)
    features = build_daily_features(frame)
    returns = features.set_index("date")["log_return_pct"]
    result, smoothed, high_state = fit_hamilton_smoothed(returns)
    assert isinstance(result, object)
    assert isinstance(smoothed, pd.Series)
    assert len(smoothed) == len(returns)
    assert high_state in (0, 1)
    assert smoothed.between(0, 1).all()


def test_chronological_split_boundaries():
    frame = _market_frame(1050)
    train, test = chronological_split(frame, test_fraction=0.01)
    assert len(train) >= 500
    assert len(test) >= 150
    assert train["date"].max() < test["date"].min()

    train2, test2 = chronological_split(frame, test_fraction=0.49)
    assert len(train2) >= 500
    assert len(test2) >= 150
    assert train2["date"].max() < test2["date"].min()


def test_evaluate_direction_synthetic():
    dates = pd.bdate_range("2020-01-01", periods=1_050)
    np.random.seed(42)
    noise = np.random.randn(1_050) * 0.5
    up_signal = (np.arange(1_050) % 10) > 4
    returns = np.where(up_signal, 0.5 + noise, -0.5 + noise)
    frame = pd.DataFrame({"date": dates, "usd_idr": 14_000 + np.cumsum(returns), "us10y_yield": 1.5})
    features = build_daily_features(frame)
    features = features.dropna(subset=["return_lag_20", "rolling_vol_20"]).reset_index(drop=True)
    train, test = chronological_split(features, test_fraction=0.20)
    metrics, predictions = evaluate_direction(train, test)
    assert len(metrics) == 2
    assert set(metrics["model"]).issubset({"Persistence sign baseline", "Logistic direction model"})
    assert (predictions["direction_up"].isin([0, 1])).all()
    assert len(predictions) == len(test)
