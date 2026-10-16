"""Rebuild the AirSense same-hour CO model from the UCI Air Quality dataset.

Provide the source as CSV, XLSX, or a ZIP containing the UCI Air Quality files.
This script keeps the final test period out of search, model choice, and fitting.
"""

from __future__ import annotations

import argparse
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
            na_values=[-200, "-200"],
            usecols=lambda column: column in {"Date", "Time", TARGET, *SENSOR_FEATURES},
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


def prepare_data(source: Path) -> pd.DataFrame:
    frame = read_source(source)
    keep = ["Date", "Time", TARGET, *SENSOR_FEATURES]
    frame = frame[[column for column in keep if column in frame.columns]].copy()
    required = {"Date", "Time", TARGET, *SENSOR_FEATURES}
    if not required.issubset(frame.columns):
        raise ValueError("The source file is missing one or more required Air Quality dataset columns.")
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
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    for column in [TARGET, *SENSOR_FEATURES]:
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
    return frame


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


def fit_time_safe_stack(
    base_estimators: dict[str, Pipeline], x: pd.DataFrame, y: pd.Series, cv: TimeSeriesSplit
) -> tuple[dict[str, Pipeline], Ridge]:
    """Train the meta-model only on expanding-window out-of-fold predictions."""
    oof = np.full((len(x), len(base_estimators)), np.nan, dtype=float)
    split_list = list(cv.split(x))
    for train_indices, valid_indices in split_list:
        for model_index, estimator in enumerate(base_estimators.values()):
            fitted = clone(estimator)
            fitted.fit(x.iloc[train_indices], y.iloc[train_indices])
            oof[valid_indices, model_index] = fitted.predict(x.iloc[valid_indices])
    valid_mask = np.isfinite(oof).all(axis=1)
    if valid_mask.sum() < 20:
        raise ValueError("Not enough time-safe out-of-fold predictions to fit the stack.")
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

    frame = prepare_data(args.data)
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

    stack_base, stack_meta = fit_time_safe_stack(
        {"Ridge baseline": ridge, "Extra Trees": extra_trees}, x_train, y_train, cv
    )
    validation_stack = {"models": stack_base, "meta_model": stack_meta, "base_names": list(stack_base)}
    candidates: dict[str, object] = {
        "Ridge (Bayesian search)": ridge,
        "Extra Trees (randomized search)": extra_trees,
        "Time-safe stack": validation_stack,
    }
    selected_name, validation_metrics = evaluate_candidates(
        candidates, validation[FEATURE_COLS], validation[TARGET]
    )
    validation_metrics.to_csv(args.output / "evs03_validation_metrics.csv", index=False)
    bayes_history.to_csv(args.output / "bayesian_search_history.csv", index=False)
    random_results.to_csv(args.output / "randomized_search_results.csv", index=False)

    development = pd.concat([train, validation], ignore_index=True)
    x_development, y_development = development[FEATURE_COLS], development[TARGET]
    if selected_name == "Ridge (Bayesian search)":
        final_model: object = clone(ridge).fit(x_development, y_development)
        bundle: dict[str, object] = {"kind": "single", "model": final_model}
    elif selected_name == "Extra Trees (randomized search)":
        final_model = clone(extra_trees).fit(x_development, y_development)
        bundle = {"kind": "single", "model": final_model}
    else:
        final_base, final_meta = fit_time_safe_stack(
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
        "target": TARGET,
        "target_units": "mg/m³",
        "features": FEATURE_COLS,
        "raw_rows": int(raw_count),
        "source_start": frame["timestamp"].min().strftime("%Y-%m-%d %H:%M:%S"),
        "source_end": frame["timestamp"].max().strftime("%Y-%m-%d %H:%M:%S"),
        "split_method": "chronological 70% train / 15% validation / 15% test by source row order",
        "cv_method": "expanding-window TimeSeriesSplit with four folds on training rows only",
        "bayesian_search": "Gaussian-process expected improvement over Ridge alpha (log10 range -3 to 3)",
        "bayesian_iterations": int(args.bayes_iterations),
        "randomized_search": "RandomizedSearchCV for ExtraTreesRegressor; time-series folds only",
        "randomized_iterations": int(args.random_iterations),
        "stack": "Ridge + Extra Trees; meta-model fit only on expanding-window OOF predictions",
        "selected_model": selected_name,
        "test_metrics": test_metrics_row,
    }
    (args.output / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"selected_model": selected_name, "validation": validation_metrics.to_dict("records"), "test": test_metrics_row}, indent=2))


if __name__ == "__main__":
    main()
