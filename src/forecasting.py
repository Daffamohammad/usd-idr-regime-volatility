"""End-to-end experiment: regime diagnosis, volatility and direction forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.nonparametric.smoothers_lowess import lowess

from .feature_engineering import add_optional_macro_features, build_daily_features, chronological_split, load_market_snapshot
from .garch_modeling import SPECS, evaluate_variance_forecasts, rolling_one_step_variance_forecasts
from .hamilton_regime import fit_hamilton_smoothed, fit_hamilton_train_and_filter, regime_summary


@dataclass
class ExperimentResult:
    features: pd.DataFrame
    train: pd.DataFrame
    test: pd.DataFrame
    volatility_metrics: pd.DataFrame
    direction_metrics: pd.DataFrame
    volatility_forecasts: pd.DataFrame
    direction_predictions: pd.DataFrame
    regime_summary: pd.DataFrame


sns.set_theme(style="whitegrid")

EVENTS = {
    "Perang dagang AS–China\n(Mar 2018)": "2018-03-01",
    "Pandemi COVID-19\n(Mar 2020)": "2020-03-11",
    "Siklus kenaikan The Fed\n(Mar 2022)": "2022-03-16",
}


def _direction_features(frame: pd.DataFrame) -> list[str]:
    base = [
        "return_lag_1",
        "return_lag_2",
        "return_lag_5",
        "abs_return_lag_1",
        "abs_return_lag_5",
        "rolling_vol_5",
        "rolling_vol_20",
        "us10y_change_bp_lag_1",
        "high_vol_probability_filtered_lag_1",
    ]
    optional = ["bi_rate_pct", "inflation_yoy_pct", "bond_spread_pct_point_lag_1"]
    return base + [column for column in optional if column in frame.columns]


def evaluate_direction(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare a persistence-sign baseline to a train-only logistic model."""
    features = _direction_features(train)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2_000, random_state=42))
    model.fit(train[features], train["direction_up"])
    logistic = model.predict(test[features])
    persistence = (test["return_lag_1"] > 0).astype(int).to_numpy()
    actual = test["direction_up"].to_numpy()
    rows = []
    for name, prediction in [("Persistence sign baseline", persistence), ("Logistic direction model", logistic)]:
        cm = confusion_matrix(actual, prediction, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        rows.append(
            {
                "model": name,
                "observations": int(len(actual)),
                "accuracy_pct": 100 * accuracy_score(actual, prediction),
                "up_prediction_share_pct": 100 * float(np.mean(prediction)),
                "true_up_share_pct": 100 * float(np.mean(actual)),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            }
        )
    predictions = test[["date", "log_return_pct", "direction_up"]].copy()
    predictions["persistence_direction_up"] = persistence
    predictions["logistic_direction_up"] = logistic
    predictions["logistic_probability_up"] = model.predict_proba(test[features])[:, 1]
    return pd.DataFrame(rows), predictions


def plot_regime_probability(regime_frame: pd.DataFrame, output_path: str | Path) -> None:
    fig, axis = plt.subplots(figsize=(13, 5.5))
    raw = regime_frame["high_vol_probability_smoothed"].values
    smooth_indices = np.arange(len(raw))
    smoothed = lowess(raw, smooth_indices, frac=0.05, return_sorted=False)[:, 1]
    axis.plot(
        regime_frame["date"], regime_frame["high_vol_probability_smoothed"],
        color="#b22222", alpha=0.16, linewidth=0.65, label="Probabilitas smoothed harian",
    )
    axis.plot(
        regime_frame["date"], smoothed,
        color="#b22222",
        linewidth=1.35,
        label="LOWESS (frac=0.05) dari probabilitas smoothed",
    )
    for label, event_date in EVENTS.items():
        timestamp = pd.Timestamp(event_date)
        axis.axvline(timestamp, color="#4c566a", linestyle="--", linewidth=0.9, alpha=0.8)
        axis.annotate(label, xy=(timestamp, 0.98), xycoords=("data", "axes fraction"), xytext=(4, -4), textcoords="offset points", va="top", fontsize=8)
    axis.set(title="Hamilton Markov Switching: probabilitas high-volatility regime", xlabel="Tanggal", ylabel="Probabilitas")
    axis.set_ylim(-0.02, 1.05)
    axis.xaxis.set_major_locator(mdates.YearLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axis.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_volatility_forecasts(volatility_frame: pd.DataFrame, output_path: str | Path) -> None:
    fig, axis = plt.subplots(figsize=(13, 5.5))
    # Plot daily standard deviation rather than variance for reader clarity.
    axis.plot(volatility_frame["date"], np.sqrt(volatility_frame["realized_variance"]), color="black", alpha=0.45, linewidth=0.8, label="|return aktual| (proxy volatilitas harian)")
    for column, color in zip([c for c in volatility_frame if c.startswith("variance_")], cycle(["#1f77b4", "#ff7f0e", "#2ca02c"])):
        axis.plot(volatility_frame["date"], np.sqrt(volatility_frame[column]), linewidth=1.1, color=color, label=column.removeprefix("variance_"))
    axis.set(title="Forecast one-step volatilitas USD/IDR pada test set", xlabel="Tanggal", ylabel="Volatilitas harian (%)")
    axis.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axis.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_experiment(
    raw_path: str | Path,
    output_dir: str | Path = "outputs",
    optional_macro_dir: str | Path | None = None,
) -> ExperimentResult:
    """Execute and persist the complete v2 experiment from a frozen raw CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features = build_daily_features(load_market_snapshot(raw_path))
    if optional_macro_dir is not None:
        features = add_optional_macro_features(features, optional_macro_dir)
    train, test = chronological_split(features)
    all_returns = features.set_index("date")["log_return_pct"]
    train_returns = train.set_index("date")["log_return_pct"]

    _, filtered_probability, _ = fit_hamilton_train_and_filter(train_returns, all_returns)
    _, smoothed_probability, _ = fit_hamilton_smoothed(all_returns)
    features = features.set_index("date")
    features["high_vol_probability_filtered"] = filtered_probability
    features["high_vol_probability_filtered_lag_1"] = features["high_vol_probability_filtered"].shift(1)
    features["high_vol_probability_smoothed"] = smoothed_probability
    features = features.dropna(subset=["high_vol_probability_filtered_lag_1"] + _direction_features(features)).reset_index()
    train, test = chronological_split(features)

    variance_columns = {}
    full_returns = features.set_index("date")["log_return_pct"]
    for spec in SPECS:
        variance_columns[f"variance_{spec.name}"] = rolling_one_step_variance_forecasts(full_returns, len(train), spec)
    forecasts = pd.DataFrame(variance_columns)
    volatility_metrics = evaluate_variance_forecasts(test.set_index("date")["log_return_pct"], forecasts)
    volatility_frame = pd.concat([test.set_index("date")[["log_return_pct"]].pow(2).rename(columns={"log_return_pct": "realized_variance"}), forecasts], axis=1).reset_index()
    direction_metrics, direction_predictions = evaluate_direction(train, test)
    regime_frame = features[["date", "log_return_pct", "high_vol_probability_filtered", "high_vol_probability_smoothed"]].copy()
    regime_table = regime_summary(features.set_index("date")["log_return_pct"], features.set_index("date")["high_vol_probability_smoothed"])

    processed_dir = Path(raw_path).resolve().parents[1] / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(processed_dir / "daily_features.csv", index=False)
    volatility_metrics.to_csv(output_dir / "volatility_metrics.csv", index=False)
    direction_metrics.to_csv(output_dir / "direction_metrics.csv", index=False)
    volatility_frame.to_csv(output_dir / "volatility_forecasts.csv", index=False)
    direction_predictions.to_csv(output_dir / "direction_predictions.csv", index=False)
    regime_frame.to_csv(output_dir / "regime_probabilities.csv", index=False)
    regime_table.to_csv(output_dir / "regime_summary.csv", index=False)
    plot_regime_probability(regime_frame, output_dir / "regime_probability.png")
    plot_volatility_forecasts(volatility_frame, output_dir / "volatility_forecasts.png")

    return ExperimentResult(
        features=features,
        train=train,
        test=test,
        volatility_metrics=volatility_metrics,
        direction_metrics=direction_metrics,
        volatility_forecasts=volatility_frame,
        direction_predictions=direction_predictions,
        regime_summary=regime_table,
    )


if __name__ == "__main__":
    result = run_experiment("data/raw/yahoo_usd_idr_us10y.csv")
    print("Volatility metrics")
    print(result.volatility_metrics.to_string(index=False))
    print("\nDirection metrics")
    print(result.direction_metrics.to_string(index=False))
