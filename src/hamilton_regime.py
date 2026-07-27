"""Hamilton-style two-state Markov switching regime diagnostics."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


def _state_variances(returns: pd.Series, probabilities: pd.DataFrame) -> pd.Series:
    squared = returns.reindex(probabilities.index).pow(2)
    return probabilities.mul(squared, axis=0).sum(axis=0) / probabilities.sum(axis=0)


def fit_hamilton_train_and_filter(
    train_returns: pd.Series, all_returns: pd.Series
) -> tuple[object, pd.Series, int]:
    """Fit only train data then filter all dates using fixed train parameters.

    This creates a predictive filtered probability: at date *t* it uses returns
    through *t*, never future returns.  It is safe to lag once before direction
    modelling.
    """
    np.random.seed(42)
    train_model = MarkovRegression(
        train_returns,
        k_regimes=2,
        trend="c",
        switching_variance=True,
    )
    train_result = train_model.fit(search_reps=3, em_iter=5, disp=False)
    train_probs = train_result.smoothed_marginal_probabilities
    variances = _state_variances(train_returns, train_probs)
    high_state = int(variances.idxmax())
    low_state = 1 - high_state
    ratio = variances[high_state] / variances[low_state]
    if ratio < 1.5:
        warnings.warn(f"High/low variance ratio ({ratio:.2f}) < 1.5 — state labeling may be unstable.")

    full_model = MarkovRegression(all_returns, k_regimes=2, trend="c", switching_variance=True)
    filtered = full_model.filter(train_result.params).filtered_marginal_probabilities[high_state]
    filtered.name = "high_vol_probability_filtered"
    return train_result, filtered, high_state


def fit_hamilton_smoothed(all_returns: pd.Series) -> tuple[object, pd.Series, int]:
    """Fit a full-sample model for descriptive, retrospective visualization only."""
    np.random.seed(42)
    model = MarkovRegression(all_returns, k_regimes=2, trend="c", switching_variance=True)
    result = model.fit(search_reps=3, em_iter=5, disp=False)
    variances = _state_variances(all_returns, result.smoothed_marginal_probabilities)
    high_state = int(variances.idxmax())
    low_state = 1 - high_state
    ratio = variances[high_state] / variances[low_state]
    if ratio < 1.5:
        warnings.warn(f"High/low variance ratio ({ratio:.2f}) < 1.5 — state labeling may be unstable.")
    smoothed = result.smoothed_marginal_probabilities[high_state]
    smoothed.name = "high_vol_probability_smoothed"
    return result, smoothed, high_state


def regime_summary(returns: pd.Series, probabilities: pd.Series) -> pd.DataFrame:
    """Provide transparent state labels based on conditional realized variance."""
    high = probabilities >= 0.5
    rows = []
    for label, mask in [("Low probability high-vol", ~high), ("High probability high-vol", high)]:
        subset = returns.loc[mask]
        rows.append(
            {
                "state_label": label,
                "observations": int(mask.sum()),
                "mean_abs_return_pct": float(subset.abs().mean()),
                "realized_variance_pct2": float(subset.pow(2).mean()),
            }
        )
    return pd.DataFrame(rows)
