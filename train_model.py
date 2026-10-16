"""Rebuild the AirSense same-hour CO model from the UCI Air Quality dataset.

Provide the source as CSV, XLSX, or a ZIP containing the UCI Air Quality files.
This script keeps the final test period out of search, model choice, and fitting.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import zipfile
from pathlib import Path
from datetime import time as datetime_time

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SENSOR_FEATURES = [
    "PT08.S1(CO)",
    "PT08.S2(NMHC)",
    "PT08.S3(NOx)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH",
]
FEATURE_COLS = SENSOR_FEATURES + ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos"]
TARGET = "CO(GT)"
REFERENCE_TARGETS = ["NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]
DATASET_MEASURES = [TARGET, *SENSOR_FEATURES, *REFERENCE_TARGETS]
RANDOM_STATE = 42


def read_source(source: Path) -> pd.DataFrame:
    """Read the UCI semicolon CSV, either directly or from a ZIP archive."""
    def read_member(stream: object, name: str) -> pd.DataFrame:
        if Path(name).suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(stream)
        return pd.read_csv(
            stream,
            sep=";",
            decimal=",",
        )

    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            names = archive.namelist()
            preferred = {"airquality.csv", "airqualityuci.csv", "airqualityuci.xlsx", "airquality.xlsx"}
            matches = [name for name in names if Path(name).name.lower() in preferred]
            if not matches:
                matches = [name for name in names if Path(name).suffix.lower() in {".csv", ".xlsx", ".xls"}]
            if not matches:
                raise ValueError("The ZIP must contain the UCI Air Quality CSV or Excel file.")
            with archive.open(matches[0]) as stream:
                return read_member(stream, matches[0])
    with source.open("rb") as stream:
        return read_member(stream, source.name)


def clean_and_audit(source: Path) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    """Normalize the source and return auditable counts before model fitting."""
    frame = read_source(source)
    source_rows = len(frame)
    source_columns = [str(column) for column in frame.columns]
    empty_source_columns = [str(column) for column in frame.columns if frame[column].isna().all()]
    keep = ["Date", "Time", *DATASET_MEASURES]
    frame = frame[[column for column in keep if column in frame.columns]].copy()
    required = {"Date", "Time", TARGET, *SENSOR_FEATURES}
    if not required.issubset(frame.columns):
        raise ValueError("The source file is missing one or more required Air Quality dataset columns.")

    sentinel_counts: dict[str, int] = {}
    for column in DATASET_MEASURES:
        if column not in frame.columns:
            continue
        raw_values = frame[column]
        if not pd.api.types.is_numeric_dtype(raw_values):
            raw_values = raw_values.astype(str).str.replace(",", ".", regex=False)
        numeric_values = pd.to_numeric(raw_values, errors="coerce")
        sentinel_counts[column] = int(numeric_values.eq(-200).sum())

    frame["Date"] = pd.to_datetime(frame["Date"], dayfirst=True, errors="coerce")
    def parse_time(value: object) -> pd.Timestamp | pd.NaT:
        if pd.isna(value):
            return pd.NaT
        if isinstance(value, datetime_time):
            return pd.Timestamp("1970-01-01") + pd.to_timedelta(
                value.hour * 3600 + value.minute * 60 + value.second, unit="s"
            )
        if isinstance(value, (int, float, np.number)) and 0 <= float(value) < 1:
            return pd.Timestamp("1970-01-01") + pd.to_timedelta(float(value), unit="D")
        normalized = str(value).strip().replace(".", ":")
        return pd.to_datetime(normalized, format="%H:%M:%S", errors="coerce")

    parsed_time = pd.to_datetime([parse_time(value) for value in frame["Time"]], errors="coerce")
    frame["timestamp"] = frame["Date"] + pd.to_timedelta(parsed_time.hour, unit="h") + pd.to_timedelta(
        parsed_time.minute, unit="m"
    ) + pd.to_timedelta(parsed_time.second, unit="s")
    invalid_timestamp_rows = int(frame["timestamp"].isna().sum())
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp", kind="stable").reset_index(drop=True)
    duplicate_timestamp_rows = int(frame.duplicated(subset=["timestamp"], keep="first").sum())
    frame = frame.drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)
    for column in DATASET_MEASURES:
        if column not in frame.columns:
            continue
        values = frame[column]
        if not pd.api.types.is_numeric_dtype(values):
            values = values.astype(str).str.replace(",", ".", regex=False)
        frame[column] = pd.to_numeric(values, errors="coerce").replace(-200, np.nan)

    hour = frame["timestamp"].dt.hour
    weekday = frame["timestamp"].dt.dayofweek
    frame["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    frame["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    frame["weekday_sin"] = np.sin(2 * np.pi * weekday / 7.0)
    frame["weekday_cos"] = np.cos(2 * np.pi * weekday / 7.0)

    role_by_column = {
        TARGET: "Prediction target",
        **{column: "Model input" for column in SENSOR_FEATURES},
        **{column: "Reference pollutant excluded from model inputs" for column in REFERENCE_TARGETS},
    }
    missingness = pd.DataFrame(
        [
            {
                "variable": column,
                "role": role_by_column.get(column, "Source measurement"),
                "missing_count": int(frame[column].isna().sum()),
                "missing_percent": float(frame[column].isna().mean() * 100.0),
                "observed_count": int(frame[column].notna().sum()),
                "sentinel_minus_200_count": int(sentinel_counts.get(column, 0)),
            }
            for column in DATASET_MEASURES
            if column in frame.columns
        ]
    )
    target_missing_count = int(frame[TARGET].isna().sum())
    quality = {
        "source_file": source.name if source.is_file() else source.name,
        "source_rows": int(source_rows),
        "source_columns": source_columns,
        "empty_source_columns": empty_source_columns,
        "invalid_timestamp_rows_removed": invalid_timestamp_rows,
        "duplicate_timestamp_rows_removed": duplicate_timestamp_rows,
        "usable_timestamp_rows": int(len(frame)),
        "timestamp_start": frame["timestamp"].min().strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp_end": frame["timestamp"].max().strftime("%Y-%m-%d %H:%M:%S"),
        "missing_sentinel": -200,
        "sentinel_counts_by_variable": sentinel_counts,
        "target_missing_rows_excluded_from_scoring": target_missing_count,
        "target_observed_rows": int(frame[TARGET].notna().sum()),
        "outlier_rows_removed": 0,
        "outlier_policy": "No rows were removed by an automatic outlier rule; no validated physical bounds were supplied.",
        "feature_imputation": "Median imputation is fitted within each training fold and on development data only.",
    }
    return frame, quality, missingness


def prepare_data(source: Path) -> pd.DataFrame:
    """Backward-compatible helper returning the normalized modeling frame."""
    return clean_and_audit(source)[0]


def write_data_profile(
    frame: pd.DataFrame,
    quality: dict[str, object],
    missingness: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save cleaned-data summaries and charts without copying the source file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data_quality_summary.json").write_text(
        json.dumps(quality, indent=2), encoding="utf-8"
    )
    missingness.to_csv(output_dir / "cleaned_missingness.csv", index=False)

    numeric = frame[DATASET_MEASURES].copy()
    numeric.describe(percentiles=[0.25, 0.5, 0.75]).T.rename_axis("variable").reset_index().to_csv(
        output_dir / "cleaned_feature_summary.csv", index=False
    )

    measured_co = frame.dropna(subset=[TARGET]).copy()
    monthly = (
        measured_co.assign(month=measured_co["timestamp"].dt.to_period("M").astype(str))
        .groupby("month", as_index=False)[TARGET]
        .agg(mean_CO_mg_m3="mean", median_CO_mg_m3="median", records="count")
    )
    monthly.to_csv(output_dir / "cleaned_co_by_month.csv", index=False)

    hourly = (
        measured_co.assign(hour=measured_co["timestamp"].dt.hour)
        .groupby("hour", as_index=False)[TARGET]
        .agg(mean_CO_mg_m3="mean", median_CO_mg_m3="median", records="count")
    )
    hourly.to_csv(output_dir / "cleaned_co_by_hour.csv", index=False)

    counts, edges = np.histogram(measured_co[TARGET].to_numpy(), bins=20)
    distribution = pd.DataFrame(
        {
            "bin_start_mg_m3": edges[:-1],
            "bin_end_mg_m3": edges[1:],
            "bin_center_mg_m3": (edges[:-1] + edges[1:]) / 2.0,
            "records": counts,
        }
    )
    distribution.to_csv(output_dir / "cleaned_co_distribution.csv", index=False)

    correlation_columns = [TARGET, *SENSOR_FEATURES]
    frame[correlation_columns].corr(method="pearson").to_csv(
        output_dir / "cleaned_correlation.csv", index=True, index_label="variable"
    )
    write_eda_svg(
        output_dir / "cleaned_data_eda.svg",
        monthly=monthly,
        hourly=hourly,
        distribution=distribution,
        missingness=missingness,
        measured_target_rows=int(frame[TARGET].notna().sum()),
    )


def write_eda_svg(
    path: Path,
    monthly: pd.DataFrame,
    hourly: pd.DataFrame,
    distribution: pd.DataFrame,
    missingness: pd.DataFrame,
    measured_target_rows: int,
) -> None:
    """Render a compact, dependency-free figure for the README."""
    width, height = 1200, 860
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">AirSense cleaned data overview</title>',
        '<desc id="desc">Four charts show CO concentration distribution, monthly mean CO, average CO by hour, and missing measurements after replacing the dataset missing marker.</desc>',
        '<rect width="100%" height="100%" fill="#f3f6f4"/>',
        '<text x="52" y="52" fill="#18312e" font-family="Segoe UI,Arial,sans-serif" font-size="27" font-weight="700">Cleaned data · UCI Air Quality</text>',
        '<text x="52" y="80" fill="#60736f" font-family="Segoe UI,Arial,sans-serif" font-size="14">-200 converted to missing · incomplete timestamps removed · target measured rows only</text>',
    ]

    def panel(x: int, y: int, title: str, subtitle: str) -> None:
        parts.append(f'<rect x="{x}" y="{y}" width="536" height="330" rx="14" fill="#ffffff" stroke="#dce5e1"/>')
        parts.append(f'<text x="{x+24}" y="{y+34}" fill="#18312e" font-family="Segoe UI,Arial,sans-serif" font-size="16" font-weight="700">{html.escape(title)}</text>')
        parts.append(f'<text x="{x+24}" y="{y+57}" fill="#60736f" font-family="Segoe UI,Arial,sans-serif" font-size="12">{html.escape(subtitle)}</text>')

    def chart_axes(x: int, y: int, max_value: float, label: str) -> tuple[float, float, float, float]:
        left, top, plot_w, plot_h = x + 66, y + 88, 432.0, 194.0
        for tick in range(5):
            yy = top + plot_h - tick * plot_h / 4
            val = max_value * tick / 4
            parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{left+plot_w}" y2="{yy:.1f}" stroke="#e7eeea" stroke-width="1"/>')
            parts.append(f'<text x="{left-10}" y="{yy+4:.1f}" text-anchor="end" fill="#71817d" font-family="Segoe UI,Arial,sans-serif" font-size="10">{val:.0f}</text>')
        parts.append(f'<text x="{left+plot_w/2:.1f}" y="{top+plot_h+35}" text-anchor="middle" fill="#60736f" font-family="Segoe UI,Arial,sans-serif" font-size="11">{html.escape(label)}</text>')
        return left, top, plot_w, plot_h

    panel(40, 108, "Measured CO distribution", f"Hourly reference CO · mg/m³ · {measured_target_rows:,} measured records")
    left, top, plot_w, plot_h = chart_axes(40, 108, max(float(distribution["records"].max()), 1.0), "CO concentration bin (mg/m³)")
    bar_w = plot_w / max(len(distribution), 1)
    for index, row in distribution.reset_index(drop=True).iterrows():
        bar_h = float(row["records"]) / max(float(distribution["records"].max()), 1.0) * plot_h
        parts.append(f'<rect x="{left+index*bar_w+1:.1f}" y="{top+plot_h-bar_h:.1f}" width="{max(bar_w-2, 1):.1f}" height="{bar_h:.1f}" fill="#2f8273"/>')
    if len(distribution):
        for index in [0, len(distribution)//2, len(distribution)-1]:
            value = float(distribution.iloc[index]["bin_center_mg_m3"])
            xx = left + (index + 0.5) * bar_w
            parts.append(f'<text x="{xx:.1f}" y="{top+plot_h+17}" text-anchor="middle" fill="#71817d" font-family="Segoe UI,Arial,sans-serif" font-size="10">{value:.1f}</text>')

    panel(624, 108, "Monthly mean CO", "Reference concentration · mg/m³ · chronological view")
    left, top, plot_w, plot_h = chart_axes(624, 108, max(float(monthly["mean_CO_mg_m3"].max()) * 1.1, 1.0), "Month")
    vals = monthly["mean_CO_mg_m3"].to_numpy(dtype=float)
    if len(vals):
        hi = max(float(monthly["mean_CO_mg_m3"].max()) * 1.1, 1.0)
        points = []
        for index, value in enumerate(vals):
            xx = left + (index / max(len(vals)-1, 1)) * plot_w
            yy = top + plot_h - (value / hi) * plot_h
            points.append(f"{xx:.1f},{yy:.1f}")
            parts.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="3.5" fill="#c47a27"/>')
        parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="#c47a27" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>')
        for index in [0, len(vals)//2, len(vals)-1]:
            xx = left + (index / max(len(vals)-1, 1)) * plot_w
            label = str(monthly.iloc[index]["month"])
            parts.append(f'<text x="{xx:.1f}" y="{top+plot_h+17}" text-anchor="middle" fill="#71817d" font-family="Segoe UI,Arial,sans-serif" font-size="10">{html.escape(label)}</text>')

    panel(40, 470, "Average CO by hour", "Grouped by clock hour · measured target values only")
    left, top, plot_w, plot_h = chart_axes(40, 470, max(float(hourly["mean_CO_mg_m3"].max()) * 1.1, 1.0), "Hour of day")
    hour_hi = max(float(hourly["mean_CO_mg_m3"].max()) * 1.1, 1.0)
    bar_w = plot_w / max(len(hourly), 1)
    for _, row in hourly.iterrows():
        index = int(row["hour"])
        bar_h = float(row["mean_CO_mg_m3"]) / hour_hi * plot_h
        parts.append(f'<rect x="{left+index*bar_w+2:.1f}" y="{top+plot_h-bar_h:.1f}" width="{max(bar_w-4, 1):.1f}" height="{bar_h:.1f}" rx="2" fill="#2f8273"/>')
    for hour_mark in [0, 6, 12, 18, 23]:
        xx = left + (hour_mark + 0.5) * bar_w
        parts.append(f'<text x="{xx:.1f}" y="{top+plot_h+17}" text-anchor="middle" fill="#71817d" font-family="Segoe UI,Arial,sans-serif" font-size="10">{hour_mark:02d}:00</text>')

    panel(624, 470, "Missing measurements after cleaning", "-200 sentinel converted to missing · counts in timestamped rows")
    nonzero = missingness.loc[missingness["missing_count"] > 0].sort_values("missing_count", ascending=True).tail(8)
    left, top, plot_w, plot_h = 716, 558, 410.0, 194.0
    for tick in range(5):
        xx = left + tick * plot_w / 4
        val = float(nonzero["missing_count"].max()) * tick / 4 if not nonzero.empty else 0.0
        parts.append(f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{top+plot_h}" stroke="#e7eeea" stroke-width="1"/>')
        parts.append(f'<text x="{xx:.1f}" y="{top+plot_h+17}" text-anchor="middle" fill="#71817d" font-family="Segoe UI,Arial,sans-serif" font-size="10">{val:.0f}</text>')
    bar_h = plot_h / max(len(nonzero), 1)
    for index, (_, row) in enumerate(nonzero.iterrows()):
        yy = top + plot_h - (index + 1) * bar_h
        bar_w = float(row["missing_count"]) / max(float(nonzero["missing_count"].max()), 1.0) * plot_w
        parts.append(f'<rect x="{left:.1f}" y="{yy+2:.1f}" width="{bar_w:.1f}" height="{max(bar_h-4, 1):.1f}" fill="#c47a27"/>')
        variable = str(row["variable"]).replace("(GT)", "").replace("PT08.", "")
        parts.append(f'<text x="{left-8:.1f}" y="{yy+bar_h/2+3:.1f}" text-anchor="end" fill="#60736f" font-family="Segoe UI,Arial,sans-serif" font-size="9">{html.escape(variable)}</text>')
        parts.append(f'<text x="{left+bar_w+6:.1f}" y="{yy+bar_h/2+4:.1f}" fill="#60736f" font-family="Segoe UI,Arial,sans-serif" font-size="10">{int(row["missing_count"]):,}</text>')

    parts.append('<text x="52" y="836" fill="#71817d" font-family="Segoe UI,Arial,sans-serif" font-size="11">Source: UCI Machine Learning Repository · Air Quality · data rows with a measured CO target appear in the first three charts.</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def ridge_pipeline(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=float(alpha))),
        ]
    )


def extra_trees_pipeline(**parameters: object) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesRegressor(
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                    **parameters,
                ),
            ),
        ]
    )


def bayesian_ridge_search(
    x: pd.DataFrame,
    y: pd.Series,
    cv: TimeSeriesSplit,
    iterations: int = 21,
) -> tuple[float, pd.DataFrame]:
    """Use a Gaussian-process surrogate and expected improvement over log10(alpha)."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

    rng = np.random.default_rng(RANDOM_STATE)
    observed_x: list[float] = []
    observed_score: list[float] = []
    history: list[dict[str, float | int]] = []
    grid = np.linspace(-3.0, 3.0, 801)
    initial = [-3.0, -1.5, 0.0, 1.5, 3.0]

    for iteration in range(iterations):
        if iteration < len(initial):
            log_alpha = initial[iteration]
        else:
            kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(
                noise_level=1e-5, noise_level_bounds="fixed"
            )
            surrogate = GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                n_restarts_optimizer=1,
                random_state=RANDOM_STATE,
            )
            surrogate.fit(np.asarray(observed_x).reshape(-1, 1), np.asarray(observed_score))
            mean, deviation = surrogate.predict(grid.reshape(-1, 1), return_std=True)
            best = max(observed_score)
            improvement = mean - best - 0.005
            z = np.divide(improvement, deviation, out=np.zeros_like(improvement), where=deviation > 1e-12)
            expected_improvement = improvement * norm.cdf(z) + deviation * norm.pdf(z)
            expected_improvement[deviation <= 1e-12] = 0.0
            for tried in observed_x:
                expected_improvement[np.abs(grid - tried) < 0.015] = -np.inf
            choice = int(np.argmax(expected_improvement))
            if not np.isfinite(expected_improvement[choice]):
                log_alpha = float(rng.uniform(-3.0, 3.0))
            else:
                log_alpha = float(grid[choice])

        score = float(
            cross_val_score(
                ridge_pipeline(10.0**log_alpha),
                x,
                y,
                scoring="neg_mean_absolute_error",
                cv=cv,
                n_jobs=1,
            ).mean()
        )
        observed_x.append(float(log_alpha))
        observed_score.append(score)
        history.append(
            {
                "iteration": iteration + 1,
                "alpha": float(10.0**log_alpha),
                "cv_mae_mg_m3": -score,
            }
        )

    best_index = int(np.argmax(observed_score))
    best_alpha = float(10.0 ** observed_x[best_index])
    return best_alpha, pd.DataFrame(history)


def randomized_extra_trees_search(
    x: pd.DataFrame,
    y: pd.Series,
    cv: TimeSeriesSplit,
    iterations: int = 12,
) -> tuple[Pipeline, pd.DataFrame]:
    estimator = extra_trees_pipeline()
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions={
            "model__n_estimators": [120, 180, 240, 300],
            "model__max_depth": [None, 8, 12, 18],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__max_features": [0.7, 0.85, 1.0],
        },
        n_iter=iterations,
        scoring="neg_mean_absolute_error",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )
    search.fit(x, y)
    results = pd.DataFrame(
        {
            "iteration": np.arange(1, len(search.cv_results_["params"]) + 1),
            "cv_mae_mg_m3": -search.cv_results_["mean_test_score"],
            "parameters": [json.dumps(params, sort_keys=True) for params in search.cv_results_["params"]],
        }
    ).sort_values("cv_mae_mg_m3", ignore_index=True)
    return search.best_estimator_, results


def fit_expanding_window_stack(
    base_estimators: dict[str, Pipeline], x: pd.DataFrame, y: pd.Series, cv: TimeSeriesSplit
) -> tuple[dict[str, Pipeline], Ridge]:
    """Fit a stack from expanding-window OOF predictions.

    Base-estimator hyperparameters are selected on the full training partition
    before this function runs, so this comparison is not a fully nested stack.
    """
    oof = np.full((len(x), len(base_estimators)), np.nan, dtype=float)
    split_list = list(cv.split(x))
    for train_indices, valid_indices in split_list:
        for model_index, estimator in enumerate(base_estimators.values()):
            fitted = clone(estimator)
            fitted.fit(x.iloc[train_indices], y.iloc[train_indices])
            oof[valid_indices, model_index] = fitted.predict(x.iloc[valid_indices])
    valid_mask = np.isfinite(oof).all(axis=1)
    if valid_mask.sum() < 20:
        raise ValueError("Not enough expanding-window out-of-fold predictions to fit the stack.")
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(oof[valid_mask], y.iloc[np.flatnonzero(valid_mask)])
    fitted_base: dict[str, Pipeline] = {}
    for name, estimator in base_estimators.items():
        fitted_base[name] = clone(estimator).fit(x, y)
    return fitted_base, meta_model


def raw_predictions(model: object, x: pd.DataFrame) -> np.ndarray:
    if isinstance(model, dict):
        base = np.column_stack([model["models"][name].predict(x) for name in model["base_names"]])
        return np.asarray(model["meta_model"].predict(base), dtype=float)
    return np.asarray(model.predict(x), dtype=float)


def regression_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    return {
        "R²": float(r2_score(actual, predicted)),
        "MAE": float(mean_absolute_error(actual, predicted)),
        "MSE": float(mean_squared_error(actual, predicted)),
        "RMSE": float(math.sqrt(mean_squared_error(actual, predicted))),
    }


def evaluate_candidates(
    candidates: dict[str, object], x: pd.DataFrame, y: pd.Series
) -> tuple[str, pd.DataFrame]:
    records: list[dict[str, object]] = []
    for name, model in candidates.items():
        predicted = np.maximum(raw_predictions(model, x), 0.0)
        records.append({"Model": name, **regression_metrics(y, predicted)})
    comparison = pd.DataFrame(records).sort_values("MAE", ignore_index=True)
    return str(comparison.iloc[0]["Model"]), comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="UCI Air Quality CSV, XLSX, or ZIP archive")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--bayes-iterations", type=int, default=21)
    parser.add_argument("--random-iterations", type=int, default=12)
    args = parser.parse_args()

    frame, data_quality, missingness = clean_and_audit(args.data)
    write_data_profile(frame, data_quality, missingness, args.output)
    raw_count = len(frame)
    train_end = int(raw_count * 0.70)
    validation_end = int(raw_count * 0.85)
    windows = {
        "train": frame.iloc[:train_end].copy(),
        "validation": frame.iloc[train_end:validation_end].copy(),
        "test": frame.iloc[validation_end:].copy(),
    }
    for split in windows.values():
        split.dropna(subset=[TARGET], inplace=True)
    train = windows["train"]
    validation = windows["validation"]
    test = windows["test"]
    if min(map(len, (train, validation, test))) == 0:
        raise ValueError("A chronological split has no measured CO targets.")

    x_train, y_train = train[FEATURE_COLS], train[TARGET]
    cv = TimeSeriesSplit(n_splits=4)
    best_alpha, bayes_history = bayesian_ridge_search(x_train, y_train, cv, args.bayes_iterations)
    ridge = ridge_pipeline(best_alpha)
    ridge.fit(x_train, y_train)
    extra_trees, random_results = randomized_extra_trees_search(
        x_train, y_train, cv, args.random_iterations
    )

    stack_base, stack_meta = fit_expanding_window_stack(
        {"Ridge baseline": ridge, "Extra Trees": extra_trees}, x_train, y_train, cv
    )
    validation_stack = {"models": stack_base, "meta_model": stack_meta, "base_names": list(stack_base)}
    candidates: dict[str, object] = {
        "Ridge (Bayesian search)": ridge,
        "Extra Trees (randomized search)": extra_trees,
        "Expanding-window stack": validation_stack,
    }
    selected_name, validation_metrics = evaluate_candidates(
        candidates, validation[FEATURE_COLS], validation[TARGET]
    )
    validation_metrics.to_csv(args.output / "evs03_validation_metrics.csv", index=False)
    bayes_history.to_csv(args.output / "bayesian_search_history.csv", index=False)
    random_results.to_csv(args.output / "randomized_search_results.csv", index=False)

    development = pd.concat([train, validation], ignore_index=True)
    x_development, y_development = development[FEATURE_COLS], development[TARGET]
    baseline_prediction = np.full(len(test), float(y_development.mean()), dtype=float)
    baseline_metrics = regression_metrics(test[TARGET], baseline_prediction)
    if selected_name == "Ridge (Bayesian search)":
        final_model: object = clone(ridge).fit(x_development, y_development)
        bundle: dict[str, object] = {"kind": "single", "model": final_model}
    elif selected_name == "Extra Trees (randomized search)":
        final_model = clone(extra_trees).fit(x_development, y_development)
        bundle = {"kind": "single", "model": final_model}
    else:
        final_base, final_meta = fit_expanding_window_stack(
            {"Ridge baseline": ridge, "Extra Trees": extra_trees},
            x_development,
            y_development,
            TimeSeriesSplit(n_splits=4),
        )
        bundle = {
            "kind": "stack",
            "models": final_base,
            "meta_model": final_meta,
            "base_names": list(final_base),
        }
        final_model = bundle

    raw_test_prediction = raw_predictions(final_model, test[FEATURE_COLS])
    test_prediction = np.maximum(raw_test_prediction, 0.0)
    test_metrics = regression_metrics(test[TARGET], test_prediction)
    first_test_timestamp = test["timestamp"].min()
    last_test_timestamp = test["timestamp"].max()
    test_metrics_row = {
        "Model": selected_name,
        **test_metrics,
        "Test samples": int(len(test)),
        "Test start": first_test_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "Test end": last_test_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "Negative raw predictions clipped": int((raw_test_prediction < 0).sum()),
    }
    pd.DataFrame([test_metrics_row]).to_csv(args.output / "evs03_test_metrics.csv", index=False)
    test_comparison = pd.DataFrame(
        [
            {"Model": "Development mean baseline", "Role": "Reference baseline", **baseline_metrics},
            {"Model": selected_name, "Role": "Selected model", **test_metrics},
        ]
    )
    test_comparison.to_csv(args.output / "evs03_test_comparison.csv", index=False)
    pd.DataFrame(
        {
            "timestamp": test["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            "actual_CO_mg_m3": test[TARGET].to_numpy(),
            "predicted_CO_mg_m3": test_prediction,
            "raw_prediction_CO_mg_m3": raw_test_prediction,
        }
    ).to_csv(args.output / "evs03_test_predictions.csv", index=False)

    input_ranges = {
        column: {
            "min": float(x_development[column].min()),
            "max": float(x_development[column].max()),
        }
        for column in SENSOR_FEATURES
        if x_development[column].notna().any()
    }
    bundle.update(
        {
            "feature_cols": FEATURE_COLS,
            "sensor_features": SENSOR_FEATURES,
            "selected_model": selected_name,
            "target": TARGET,
            "target_units": "mg/m³",
            "input_ranges": input_ranges,
            "training_end": train["timestamp"].max().strftime("%Y-%m-%d %H:%M:%S"),
            "validation_end": validation["timestamp"].max().strftime("%Y-%m-%d %H:%M:%S"),
            "training_rows_with_target": int(len(train)),
            "validation_rows_with_target": int(len(validation)),
            "test_rows_with_target": int(len(test)),
            "postprocess_floor_mg_m3": 0.0,
        }
    )
    joblib.dump(bundle, args.output / "evs03_model_bundle.joblib", compress=3)
    manifest = {
        "dataset": "UCI Air Quality dataset, AirQuality.csv",
        "source_url": "https://archive.ics.uci.edu/dataset/360/air+quality",
        "data_quality_summary": "data_quality_summary.json",
        "source_rows": int(data_quality["source_rows"]),
        "timestamped_rows_after_cleaning": int(data_quality["usable_timestamp_rows"]),
        "invalid_timestamp_rows_removed": int(data_quality["invalid_timestamp_rows_removed"]),
        "duplicate_timestamp_rows_removed": int(data_quality["duplicate_timestamp_rows_removed"]),
        "target_missing_rows_excluded_from_scoring": int(data_quality["target_missing_rows_excluded_from_scoring"]),
        "target_observed_rows": int(data_quality["target_observed_rows"]),
        "empty_source_columns_ignored": data_quality["empty_source_columns"],
        "target": TARGET,
        "target_units": "mg/m³",
        "features": FEATURE_COLS,
        "timestamped_rows": int(raw_count),
        "source_start": frame["timestamp"].min().strftime("%Y-%m-%d %H:%M:%S"),
        "source_end": frame["timestamp"].max().strftime("%Y-%m-%d %H:%M:%S"),
        "split_method": "chronological 70% train / 15% validation / 15% test by source row order",
        "cv_method": "expanding-window TimeSeriesSplit with four folds on training rows only",
        "bayesian_search": "Gaussian-process expected improvement over Ridge alpha (log10 range -3 to 3)",
        "bayesian_iterations": int(args.bayes_iterations),
        "randomized_search": "RandomizedSearchCV for ExtraTreesRegressor; time-series folds only",
        "randomized_iterations": int(args.random_iterations),
        "stack": "Secondary comparison: Ridge + Extra Trees; base hyperparameters selected on the full training partition, then the meta-model fit from expanding-window OOF predictions. Not a fully nested stack.",
        "selected_model": selected_name,
        "test_metrics": test_metrics_row,
        "test_mean_baseline": {"prediction": float(y_development.mean()), **baseline_metrics},
        "chronological_windows": {
            "training": {
                "rows_before_target_filter": int(train_end),
                "rows_with_measured_target": int(len(train)),
                "start": train["timestamp"].min().strftime("%Y-%m-%d %H:%M:%S"),
                "end": train["timestamp"].max().strftime("%Y-%m-%d %H:%M:%S"),
            },
            "validation": {
                "rows_before_target_filter": int(validation_end - train_end),
                "rows_with_measured_target": int(len(validation)),
                "start": validation["timestamp"].min().strftime("%Y-%m-%d %H:%M:%S"),
                "end": validation["timestamp"].max().strftime("%Y-%m-%d %H:%M:%S"),
            },
            "test": {
                "rows_before_target_filter": int(raw_count - validation_end),
                "rows_with_measured_target": int(len(test)),
                "start": test["timestamp"].min().strftime("%Y-%m-%d %H:%M:%S"),
                "end": test["timestamp"].max().strftime("%Y-%m-%d %H:%M:%S"),
            },
        },
    }
    (args.output / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"selected_model": selected_name, "validation": validation_metrics.to_dict("records"), "test": test_metrics_row}, indent=2))


if __name__ == "__main__":
    main()
