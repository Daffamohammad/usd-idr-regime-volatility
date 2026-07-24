"""Leakage-aware rolling one-step forecasts for GARCH-family models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model


@dataclass(frozen=True)
class GarchSpec:
    name: str
    vol: str
    asymmetric_order: int


SPECS = (
    GarchSpec("GARCH(1,1)", "GARCH", 0),
    GarchSpec("EGARCH(1,1)", "EGARCH", 1),
    GarchSpec("GJR-GARCH(1,1)", "GARCH", 1),
)


def _make_model(returns: pd.Series, spec: GarchSpec):
    return arch_model(
        returns,
        mean="Constant",
        vol=spec.vol,
        p=1,
        o=spec.asymmetric_order,
        q=1,
        dist="t",
        rescale=False,
    )


def rolling_one_step_variance_forecasts(
    returns_pct: pd.Series, split_index: int, spec: GarchSpec, refit_every: int = 63
) -> pd.Series:
    """Forecast every test day using only returns known strictly beforehand.

    Parameters are re-estimated every `refit_every` observations.  Between
    refits, `fix` updates the conditional variance with newly observed returns
    while retaining the last legitimately estimated parameters.
    """
    if split_index < 500 or split_index >= len(returns_pct):
        raise ValueError("split_index tidak valid untuk rolling forecast.")
    predictions: dict[pd.Timestamp, float] = {}
    params = None
    for position in range(split_index, len(returns_pct)):
        history = returns_pct.iloc[:position]
        if params is None or (position - split_index) % refit_every == 0:
            fitted = _make_model(history, spec).fit(disp="off", show_warning=False)
            params = fitted.params
        fixed = _make_model(history, spec).fix(params)
        variance = float(fixed.forecast(horizon=1, reindex=False).variance.iloc[-1, 0])
        predictions[returns_pct.index[position]] = variance
    result = pd.Series(predictions, name=f"variance_{spec.name}")
    result.index.name = returns_pct.index.name
    return result


def qlike(realized_variance: pd.Series, forecast_variance: pd.Series) -> float:
    """QLIKE loss; lower is better, and both inputs must be strictly positive."""
    aligned = pd.concat([realized_variance, forecast_variance], axis=1).dropna()
    actual = aligned.iloc[:, 0].clip(lower=1e-12)
    forecast = aligned.iloc[:, 1].clip(lower=1e-12)
    return float((np.log(forecast) + actual / forecast).mean())


def evaluate_variance_forecasts(realized_returns_pct: pd.Series, forecasts: pd.DataFrame) -> pd.DataFrame:
    realized = realized_returns_pct.pow(2).rename("realized_variance")
    rows = []
    for column in forecasts:
        aligned = pd.concat([realized, forecasts[column]], axis=1).dropna()
        error = aligned.iloc[:, 1] - aligned.iloc[:, 0]
        rows.append(
            {
                "model": column.removeprefix("variance_"),
                "observations": int(len(aligned)),
                "mae_variance_pct2": float(error.abs().mean()),
                "rmse_variance_pct2": float(np.sqrt(np.mean(error.pow(2)))),
                "qlike": qlike(aligned.iloc[:, 0], aligned.iloc[:, 1]),
            }
        )
    return pd.DataFrame(rows).sort_values("qlike").reset_index(drop=True)
