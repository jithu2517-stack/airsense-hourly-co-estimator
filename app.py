from __future__ import annotations

from datetime import date
from html import escape
import json
from pathlib import Path
import math

import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "evs03_model_bundle.joblib"
TEST_METRICS_PATH = APP_DIR / "evs03_test_metrics.csv"
VALIDATION_METRICS_PATH = APP_DIR / "evs03_validation_metrics.csv"
PREDICTIONS_PATH = APP_DIR / "evs03_test_predictions.csv"
MANIFEST_PATH = APP_DIR / "training_manifest.json"
QUALITY_PATH = APP_DIR / "data_quality_summary.json"
MISSINGNESS_PATH = APP_DIR / "cleaned_missingness.csv"
CO_DISTRIBUTION_PATH = APP_DIR / "cleaned_co_distribution.csv"
CO_MONTH_PATH = APP_DIR / "cleaned_co_by_month.csv"
CO_HOUR_PATH = APP_DIR / "cleaned_co_by_hour.csv"
CORRELATION_PATH = APP_DIR / "cleaned_correlation.csv"
TEST_COMPARISON_PATH = APP_DIR / "evs03_test_comparison.csv"
BAYESIAN_HISTORY_PATH = APP_DIR / "bayesian_search_history.csv"
RANDOMIZED_RESULTS_PATH = APP_DIR / "randomized_search_results.csv"
MODEL_SKLEARN_VERSION = "1.6.1"

SENSOR_FIELDS = [
    ("PT08.S1(CO)", "PT08.S1 · CO-targeted response", 1360.0, 5.0, "sensor response"),
    ("PT08.S2(NMHC)", "PT08.S2 · NMHC-targeted response", 1046.0, 5.0, "sensor response"),
    ("PT08.S3(NOx)", "PT08.S3 · NOx-targeted response", 1056.0, 5.0, "sensor response"),
    ("PT08.S4(NO2)", "PT08.S4 · NO₂-targeted response", 1692.0, 5.0, "sensor response"),
    ("PT08.S5(O3)", "PT08.S5 · O₃-targeted response", 1268.0, 5.0, "sensor response"),
    ("T", "Temperature", 13.6, 0.1, "°C"),
    ("RH", "Relative humidity", 48.9, 0.1, "%"),
    ("AH", "Absolute humidity", 0.7578, 0.01, "dataset units"),
]
SENSOR_FEATURES = [field[0] for field in SENSOR_FIELDS]
FALLBACK_FEATURES = SENSOR_FEATURES + ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos"]

st.set_page_config(page_title="AirSense · EVS-03", page_icon="A", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --ink: #1b302c;
        --muted: #536761;
        --line: #d8e1dd;
        --paper: #f5f7f5;
        --surface: #ffffff;
        --forest: #246052;
        --forest-dark: #1c453c;
        --mint: #e8f0ec;
        --amber: #8c5a11;
        --amber-bg: #fff4df;
    }
    html, body, [class*="css"] { font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    [data-testid="stAppViewContainer"] { background: var(--paper); color: var(--ink); }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 1360px; padding-top: 1.25rem; padding-bottom: 3rem; }
    [data-testid="stSidebar"] { background: #f0f3f1; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.15rem; }
    h1, h2, h3, h4 { color: var(--ink); letter-spacing: -0.025em; }
    h1 { font-size: 1.95rem !important; font-weight: 700 !important; }
    h2 { font-size: 1.3rem !important; font-weight: 680 !important; }
    h3 { font-size: 1.04rem !important; font-weight: 670 !important; }
    p, li { color: var(--muted); }
    [data-testid="stTabs"] [role="tablist"] { gap: .2rem; border-bottom: 1px solid var(--line); overflow-x: auto; }
    [data-testid="stTabs"] button[role="tab"] { padding: .72rem .9rem; color: var(--muted); font-weight: 620; white-space: nowrap; }
    [data-testid="stTabs"] button[aria-selected="true"] { color: var(--forest); border-bottom-color: var(--forest); }
    [data-testid="stTabs"] button[role="tab"]:focus-visible,
    button:focus-visible, input:focus-visible, [role="combobox"]:focus-visible { outline: 3px solid #4d8979 !important; outline-offset: 2px !important; }
    [data-testid="stMetric"] { background: transparent; border: 0; border-bottom: 1px solid var(--line); border-radius: 0; padding: .55rem .15rem .75rem; }
    [data-testid="stMetricLabel"] p { color: var(--muted); font-size: .82rem; font-weight: 580; letter-spacing: 0; text-transform: none; }
    [data-testid="stMetricValue"] { color: var(--ink); font-size: 1.55rem; font-weight: 690; }
    div[data-testid="stForm"] { border: 0; padding: 0; }
    div[data-testid="stVerticalBlockBorderWrapper"] { background: var(--surface); border-color: var(--line); border-radius: 9px; }
    div[data-testid="stButton"] button[kind="primary"], div[data-testid="stFormSubmitButton"] button[kind="primary"] { background: var(--forest); border-color: var(--forest); border-radius: 6px; font-weight: 650; min-height: 2.8rem; }
    div[data-testid="stButton"] button[kind="primary"]:hover, div[data-testid="stFormSubmitButton"] button[kind="primary"]:hover { background: var(--forest-dark); border-color: var(--forest-dark); }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 7px; overflow: hidden; }
    .brand-row { display:flex; align-items:center; gap:.65rem; margin: .15rem 0 .8rem; }
    .brand-name { color:var(--ink); font-weight:740; letter-spacing:-.025em; font-size:.98rem; }
    .brand-code { margin-left:auto; color:var(--muted); font-size:.7rem; font-weight:650; letter-spacing:.07em; }
    .hero { display:flex; justify-content:space-between; align-items:center; gap:2rem; background:var(--surface); color:var(--ink); padding:1.35rem 1.55rem; border:1px solid var(--line); border-left:4px solid var(--forest); border-radius:7px; margin:.3rem 0 1.1rem; }
    .hero h1 { color:var(--ink); font-size:1.85rem !important; margin:.25rem 0 .3rem; }
    .hero p { color:var(--muted); margin:0; max-width:720px; line-height:1.55; }
    .eyebrow { color:var(--forest) !important; font-size:.7rem; font-weight:720; letter-spacing:.095em; text-transform:uppercase; }
    .hero-side { min-width:205px; padding-left:1.25rem; border-left:1px solid var(--line); }
    .hero-side .small { color:var(--muted); font-size:.7rem; font-weight:650; letter-spacing:.045em; text-transform:uppercase; }
    .hero-side .model { color:var(--ink); font-size:1rem; font-weight:680; margin:.35rem 0 .2rem; }
    .hero-side .context { color:var(--muted); font-size:.77rem; line-height:1.45; }
    .section-kicker { color:var(--forest); font-size:.69rem; font-weight:720; letter-spacing:.075em; text-transform:uppercase; margin-bottom:.35rem; }
    .section-copy { color:var(--muted); font-size:.9rem; line-height:1.55; margin-top:-.15rem; margin-bottom:1rem; }
    .field-title { color:var(--ink); font-size:.82rem; font-weight:680; margin:.15rem 0 .15rem; }
    .field-unit { color:var(--muted); font-size:.7rem; margin:0 0 .3rem; }
    .estimate-panel { background:var(--mint); border:1px solid #cfdfd8; border-radius:7px; padding:1.1rem 1.2rem; margin:.2rem 0 .85rem; }
    .estimate-label { color:var(--forest); font-size:.7rem; font-weight:720; letter-spacing:.065em; text-transform:uppercase; }
    .estimate-value { color:var(--forest-dark); font-size:2.35rem; font-weight:730; letter-spacing:-.045em; line-height:1.1; margin:.45rem 0 .3rem; }
    .estimate-unit { color:var(--muted); font-size:.88rem; font-weight:560; letter-spacing:0; }
    .estimate-time { color:var(--muted); font-size:.78rem; }
    .quiet-note { background:#eef2ef; border-left:3px solid #73998b; border-radius:0 5px 5px 0; padding:.72rem .82rem; color:#40554f; font-size:.82rem; line-height:1.5; }
    .source-note { color:#73827e; font-size:.75rem; line-height:1.5; }
    @media (max-width: 1050px) {
        .block-container { padding-left:1.15rem; padding-right:1.15rem; }
        .hero { gap:1.2rem; }
    }
    @media (max-width: 800px) {
        .hero { align-items:flex-start; flex-direction:column; gap:1rem; padding:1.35rem; }
        .hero-side { border-left:0; border-top:1px solid var(--line); padding:1rem 0 0; width:100%; }
        .hero h1 { font-size:1.65rem !important; }
        [data-testid="stTabs"] [role="tablist"] { justify-content:flex-start; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_bundle(path: str) -> dict:
    return joblib.load(path)


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return load_csv(str(path))
    except Exception:
        return None


@st.cache_data
def load_optional_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def make_prediction(frame: pd.DataFrame, bundle: dict) -> float:
    if bundle.get("kind") == "stack":
        base_predictions = np.column_stack(
            [bundle["models"][name].predict(frame) for name in bundle["base_names"]]
        )
        return float(bundle["meta_model"].predict(base_predictions)[0])
    return float(bundle["model"].predict(frame)[0])


def feature_columns(bundle: dict) -> list[str]:
    return list(bundle.get("feature_cols", FALLBACK_FEATURES))


def format_hour(hour: int) -> str:
    return f"{hour:02d}:00"


def render_header(bundle: dict, test_metrics: pd.DataFrame | None) -> None:
    selected_model = str(bundle.get("selected_model", "Ridge baseline"))
    if test_metrics is not None and not test_metrics.empty:
        row = test_metrics.iloc[0]
        model_name = str(row.get("Model", selected_model))
        mae = row.get("MAE")
        right = (
            f'<div class="hero-side"><div class="small">Selected model</div>'
            f'<div class="model">{escape(model_name)}</div>'
            f'<div class="context">Held-out MAE: {float(mae):.3f} mg/m³<br>Historical, time-ordered test window</div></div>'
        ) if pd.notna(mae) else ""
    else:
        right = (
            f'<div class="hero-side"><div class="small">Selected model</div>'
            f'<div class="model">{escape(selected_model)}</div>'
            '<div class="context">Same-hour estimate · offline model</div></div>'
        )
    st.markdown(
        '<div class="brand-row"><span class="brand-name">AirSense</span>'
        '<span class="brand-code">EVS-03 · RESEARCH DEMO</span></div>'
        '<div class="hero"><div><div class="eyebrow">Sensor array · CO regression</div>'
        '<h1>Hourly CO estimation</h1>'
        '<p>Estimate the same-hour reference concentration from five sensor channels and weather readings. '
        'This interface explores a model trained on historical measurements.</p></div>'
        f'{right}</div>',
        unsafe_allow_html=True,
    )


def set_history() -> None:
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []


if not MODEL_PATH.exists():
    st.error("The trained model file is missing. Keep evs03_model_bundle.joblib beside app.py.")
    st.stop()

try:
    bundle = load_model_bundle(str(MODEL_PATH))
except Exception as exc:
    st.error(
        "The model could not be loaded. It was saved with scikit-learn "
        f"{MODEL_SKLEARN_VERSION}; reinstall requirements.txt and restart the app."
    )
    st.code(str(exc))
    st.stop()

test_metrics = load_optional_csv(TEST_METRICS_PATH)
validation_metrics = load_optional_csv(VALIDATION_METRICS_PATH)
test_predictions = load_optional_csv(PREDICTIONS_PATH)
test_comparison = load_optional_csv(TEST_COMPARISON_PATH)
quality_summary = load_optional_json(QUALITY_PATH)
missingness = load_optional_csv(MISSINGNESS_PATH)
co_distribution = load_optional_csv(CO_DISTRIBUTION_PATH)
co_by_month = load_optional_csv(CO_MONTH_PATH)
co_by_hour = load_optional_csv(CO_HOUR_PATH)
correlation = load_optional_csv(CORRELATION_PATH)
feature_cols = feature_columns(bundle)
input_ranges = bundle.get("input_ranges", {})
selected_model = str(bundle.get("selected_model", "Ridge baseline"))
set_history()

render_header(bundle, test_metrics)

if sklearn.__version__ != MODEL_SKLEARN_VERSION:
    st.warning(
        f"This model was saved with scikit-learn {MODEL_SKLEARN_VERSION}; "
        f"the current environment is using {sklearn.__version__}. Use the pinned requirements for consistent results."
    )

tab_overview, tab_predict, tab_quality, tab_cleaned, tab_evaluation, tab_method = st.tabs(
    ["Overview", "Estimate", "Data quality", "Cleaned data", "Model evaluation", "Method"]
)

with tab_overview:
    st.markdown('<div class="section-kicker">Project brief · EVS-03</div>', unsafe_allow_html=True)
    st.markdown("## Same-hour CO estimation from a sensor array")
    st.markdown(
        '<div class="section-copy">AirSense estimates the reference analyzer\'s CO(GT) concentration for the same hour '
        'from five gas-sensor channels, weather readings, and time features. The published score is from a later, '
        'chronological test window.</div>',
        unsafe_allow_html=True,
    )

    if test_metrics is not None and not test_metrics.empty:
        overview_result = test_metrics.iloc[0]
        score_columns = st.columns(3)
        score_columns[0].metric(
            "Held-out R²",
            f"{float(overview_result['R²']):.3f}",
            help="R² measures variance explained; it is not a percent of predictions that are exactly correct.",
        )
        score_columns[1].metric("Mean absolute error", f"{float(overview_result['MAE']):.3f} mg/m³")
        score_columns[2].metric("Held-out records", f"{int(overview_result.get('Test samples', 0)):,}")

        tolerance_columns = st.columns(2)
        within_half = overview_result.get("Within ±0.5 mg/m³ (%)")
        within_one = overview_result.get("Within ±1.0 mg/m³ (%)")
        if pd.notna(within_half) and pd.notna(within_one):
            tolerance_columns[0].metric("Within ±0.5 mg/m³", f"{float(within_half):.1f}%")
            tolerance_columns[1].metric("Within ±1.0 mg/m³", f"{float(within_one):.1f}%")
            st.caption(
                "These are shares of test predictions inside the stated absolute-error tolerance; they are not a generic accuracy percentage."
            )

        if test_comparison is not None and not test_comparison.empty:
            baseline_rows = test_comparison[test_comparison["Role"] == "Reference baseline"]
            if not baseline_rows.empty:
                baseline_mae = float(baseline_rows.iloc[0]["MAE"])
                model_mae = float(overview_result["MAE"])
                if baseline_mae > 0:
                    improvement = (baseline_mae - model_mae) / baseline_mae * 100.0
                    st.markdown(
                        f"On this test period, the selected model's MAE was **{improvement:.1f}% lower** than the development-mean baseline."
                    )

    if quality_summary:
        st.markdown("### Data behind the result")
        data_columns = st.columns(4)
        data_summary = [
            ("Source rows", quality_summary.get("source_rows", 0)),
            ("Usable timestamps", quality_summary.get("usable_timestamp_rows", 0)),
            ("Measured CO targets", quality_summary.get("target_observed_rows", 0)),
            ("Final test records", int(test_metrics.iloc[0].get("Test samples", 0)) if test_metrics is not None and not test_metrics.empty else 0),
        ]
        for column, (label, value) in zip(data_columns, data_summary):
            column.metric(label, f"{int(value):,}")

    st.markdown("### Explore the project")
    st.caption(
        "Review the data-quality ledger, cleaned-data graphs, and held-out evaluation before trying a sensor reading. "
        "This historical-data demo is not a live monitor, AQI conversion, or public-health alert."
    )

with tab_predict:
    st.markdown('<div class="section-kicker">Prediction workspace</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Enter one hourly record. The prefilled sensor values are from the first source record '
        '(2004-03-10 at 18:00); replace them with readings from your own sensor array.</div>',
        unsafe_allow_html=True,
    )

    form_col, result_col = st.columns([1.35, 1], gap="large")
    submitted = False
    entered_values: dict[str, float] = {}
    selected_date = date(2004, 3, 10)
    selected_hour = 18
    alert_limit = 2.0

    with form_col:
        with st.container(border=True):
            st.markdown("### 01 · Sensor record")
            st.caption("Use measurements for the same hour you want to estimate.")
            with st.form("same_hour_prediction"):
                date_col, hour_col = st.columns(2)
                selected_date = date_col.date_input(
                    "Record date",
                    value=date(2004, 3, 10),
                    help="The model uses the weekday as a time feature; the date is not a forecast horizon.",
                )
                selected_hour = hour_col.selectbox(
                    "Record hour",
                    options=list(range(24)),
                    index=18,
                    format_func=format_hour,
                    help="The model uses hour-of-day. Minutes are not part of its input features.",
                )

                st.markdown("#### Sensor and weather readings")
                input_columns = st.columns(2)
                for index, (feature, label, default, step, unit) in enumerate(SENSOR_FIELDS):
                    with input_columns[index % 2]:
                        st.markdown(f'<div class="field-title">{escape(label)}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="field-unit">{escape(unit)}</div>', unsafe_allow_html=True)
                        is_missing = st.checkbox(
                            "Mark this reading missing",
                            key=f"missing_{feature}",
                            help="The imputer fitted during training supplies the replacement value.",
                        )
                        reading = st.number_input(
                            label,
                            value=float(default),
                            step=float(step),
                            format="%.4f",
                            disabled=is_missing,
                            key=f"value_{feature}",
                            label_visibility="collapsed",
                        )
                        entered_values[feature] = float("nan") if is_missing else float(reading)

                with st.expander("Optional demo comparison limit", expanded=False):
                    alert_limit = st.number_input(
                        "Show when the estimate reaches (mg/m³)",
                        min_value=0.0,
                        value=2.0,
                        step=0.1,
                        format="%.2f",
                        help="A display setting only. It is not an official health, regulatory, or AQI threshold.",
                    )
                submitted = st.form_submit_button("Estimate same-hour CO", type="primary", width="stretch")

    if submitted:
        hour = int(selected_hour)
        weekday = selected_date.weekday()
        row = dict(entered_values)
        row.update(
            {
                "hour_sin": math.sin(2 * math.pi * hour / 24.0),
                "hour_cos": math.cos(2 * math.pi * hour / 24.0),
                "weekday_sin": math.sin(2 * math.pi * weekday / 7.0),
                "weekday_cos": math.cos(2 * math.pi * weekday / 7.0),
            }
        )
        model_input = pd.DataFrame([row], columns=feature_cols)
        raw_estimate = make_prediction(model_input, bundle)
        floor_value = float(bundle.get("postprocess_floor_mg_m3", 0.0))
        estimate = max(raw_estimate, floor_value)
        timestamp = pd.Timestamp(year=selected_date.year, month=selected_date.month, day=selected_date.day, hour=hour)
        missing_features = [name for name, value in entered_values.items() if pd.isna(value)]
        outside_features: list[str] = []
        for name, value in entered_values.items():
            bounds = input_ranges.get(name)
            if bounds and pd.notna(value) and (value < bounds.get("min", value) or value > bounds.get("max", value)):
                outside_features.append(name)
        st.session_state.latest_inputs = entered_values.copy()
        st.session_state.prediction_history.append(
            {
                "record_time": timestamp.strftime("%Y-%m-%d %H:%M"),
                "estimated_CO_mg_m3": estimate,
                "raw_model_output_mg_m3": raw_estimate,
                "demo_limit_mg_m3": float(alert_limit),
                "missing_readings": len(missing_features),
                "outside_training_range": len(outside_features),
            }
        )
        st.session_state.latest_quality = {
            "missing": missing_features,
            "outside": outside_features,
            "was_floored": bool(raw_estimate < floor_value),
        }

    with result_col:
        with st.container(border=True):
            st.markdown("### 02 · Estimate")
            if st.session_state.prediction_history:
                latest = st.session_state.prediction_history[-1]
                st.markdown(
                    '<div class="estimate-panel"><div class="estimate-label">Estimated CO · same hour</div>'
                    f'<div class="estimate-value">{latest["estimated_CO_mg_m3"]:.3f} '
                    '<span class="estimate-unit">mg/m³</span></div>'
                    f'<div class="estimate-time">Record time · {escape(latest["record_time"])}</div></div>',
                    unsafe_allow_html=True,
                )
                latest_limit = latest["demo_limit_mg_m3"]
                if latest["estimated_CO_mg_m3"] >= latest_limit:
                    st.warning(f"Estimate is at or above your demo limit of {latest_limit:.2f} mg/m³.")
                else:
                    st.info(f"Estimate is below your demo limit of {latest_limit:.2f} mg/m³.")
                st.caption("The demo limit is editable and is not a health warning or regulatory standard.")

                quality = st.session_state.get("latest_quality", {})
                if quality.get("was_floored"):
                    st.warning(
                        "The raw regression output was below 0 mg/m³. The displayed value is floored at zero; "
                        "this boundary does not make the model a health instrument."
                    )
                if quality.get("missing"):
                    st.info(
                        f"{len(quality['missing'])} missing reading(s) will be filled by the training-fitted imputer."
                    )
                if quality.get("outside"):
                    st.warning(
                        "Some readings are outside the training range: "
                        + ", ".join(quality["outside"])
                        + ". Treat this estimate as extrapolative."
                    )
                st.markdown(
                    '<div class="quiet-note">Historical-data research demo. No live sensor feed, '
                    'prediction interval, AQI conversion, or health guidance is provided.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="estimate-panel"><div class="estimate-label">Ready for an hourly record</div>'
                    '<div class="estimate-value">—</div><div class="estimate-time">Enter readings and run the model.</div></div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<div class="quiet-note">The result is a model estimate of the same hour, '
                    'not a next-hour forecast or a live air-quality reading.</div>',
                    unsafe_allow_html=True,
                )

    if st.session_state.prediction_history:
        st.markdown("### Session history")
        history = pd.DataFrame(st.session_state.prediction_history)
        history_chart = history.set_index("record_time")[["estimated_CO_mg_m3"]].rename(
            columns={"estimated_CO_mg_m3": "Estimated CO (mg/m³)"}
        )
        st.caption("Predictions made during this browser session only; they are not saved to a server.")
        st.line_chart(history_chart, height=250)
        history_col, download_col = st.columns([3, 1])
        with history_col:
            history_display = history.rename(
                columns={
                    "record_time": "Record time",
                    "estimated_CO_mg_m3": "Estimated CO (mg/m³)",
                    "raw_model_output_mg_m3": "Raw model output (mg/m³)",
                    "demo_limit_mg_m3": "Demo limit (mg/m³)",
                    "missing_readings": "Missing readings",
                    "outside_training_range": "Outside training range",
                }
            )
            st.dataframe(history_display, hide_index=True, width="stretch")
        with download_col:
            st.download_button(
                "Download session CSV",
                data=history.to_csv(index=False),
                file_name="airsense_session_predictions.csv",
                mime="text/csv",
                width="stretch",
            )

with tab_quality:
    st.markdown('<div class="section-kicker">Data cleaning · 40% of the rubric</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">A transparent cleaning ledger for the supplied UCI file. Every row count below comes from the training workflow.</div>',
        unsafe_allow_html=True,
    )
    if quality_summary:
        primary_quality_cols = st.columns(3)
        primary_quality_metrics = [
            ("Rows in source file", quality_summary.get("source_rows", 0)),
            ("Usable timestamps", quality_summary.get("usable_timestamp_rows", 0)),
            ("Measured CO targets", quality_summary.get("target_observed_rows", 0)),
        ]
        for column, (label, value) in zip(primary_quality_cols, primary_quality_metrics):
            column.metric(label, f"{int(value):,}")
        audit_quality_cols = st.columns(2)
        audit_quality_metrics = [
            ("Invalid timestamps removed", quality_summary.get("invalid_timestamp_rows_removed", 0)),
            ("Duplicate timestamps removed", quality_summary.get("duplicate_timestamp_rows_removed", 0)),
        ]
        for column, (label, value) in zip(audit_quality_cols, audit_quality_metrics):
            column.metric(label, f"{int(value):,}")
        st.caption(
            f"Source period: {quality_summary.get('timestamp_start', '—')} to "
            f"{quality_summary.get('timestamp_end', '—')} · {quality_summary.get('source_file', 'source file')}"
        )

        marker_counts = quality_summary.get("sentinel_counts_by_variable", {})
        co_markers = int(marker_counts.get("CO(GT)", 0))
        st.info(
            f"The source marker -200 was converted to missing. CO(GT) had {co_markers:,} missing markers; "
            f"{int(quality_summary.get('target_missing_rows_excluded_from_scoring', 0)):,} rows without measured CO "
            "were excluded from model fitting and scoring. Missing sensor inputs were retained and imputed "
            "with training-fold medians."
        )

        st.markdown("### Cleaning decisions")
        cleaning_steps = pd.DataFrame(
            [
                {"Stage": "Import", "Check": "Read semicolon-delimited values with decimal commas", "Decision": "Convert into numeric columns; ignore fully empty source columns."},
                {"Stage": "Timestamp", "Check": "Parse date + hour and sort chronologically", "Decision": f"Remove {int(quality_summary.get('invalid_timestamp_rows_removed', 0)):,} invalid timestamp rows and {int(quality_summary.get('duplicate_timestamp_rows_removed', 0)):,} duplicate timestamps."},
                {"Stage": "Missing markers", "Check": "Count -200 by source measurement", "Decision": "Replace the sentinel with missing values; preserve missing input rows for train-only imputation."},
                {"Stage": "Target", "Check": "Check CO(GT) availability", "Decision": f"Score only the {int(quality_summary.get('target_observed_rows', 0)):,} rows with measured CO."},
                {"Stage": "Outliers", "Check": "Review measurements without assumed physical bounds", "Decision": "No rows removed by automatic outlier rules; no validated clipping limits were supplied."},
                {"Stage": "Leakage control", "Check": "Separate reference pollutants from model inputs", "Decision": "Other pollutant ground-truth columns are retained for audit only and excluded from the model."},
            ]
        )
        st.dataframe(cleaning_steps, hide_index=True, width="stretch")

        if missingness is not None and {"variable", "missing_count", "missing_percent", "role"}.issubset(missingness.columns):
            st.markdown("### Missingness after timestamp cleaning")
            st.caption("Counts are shown after converting -200 to missing and removing rows without a usable timestamp.")
            missing_plot = missingness.copy()
            missing_plot["short_name"] = missing_plot["variable"].str.replace("(GT)", "", regex=False).str.replace("PT08.", "", regex=False)
            st.bar_chart(missing_plot, x="short_name", y="missing_count", height=310, y_label="Missing values", color="#c47a27")
            st.dataframe(
                missingness.rename(
                    columns={
                        "variable": "Source field",
                        "role": "Role",
                        "missing_count": "Missing values",
                        "missing_percent": "Missing (%)",
                        "observed_count": "Observed values",
                        "sentinel_minus_200_count": "-200 markers",
                    }
                ).round({"Missing (%)": 2}),
                hide_index=True,
                width="stretch",
            )
        if quality_summary.get("empty_source_columns"):
            st.caption("Fully empty source columns ignored: " + ", ".join(quality_summary["empty_source_columns"]))
        download_col1, download_col2 = st.columns(2)
        with download_col1:
            st.download_button(
                "Download cleaning summary",
                data=json.dumps(quality_summary, indent=2),
                file_name="airsense_data_quality_summary.json",
                mime="application/json",
            )
        with download_col2:
            if missingness is not None:
                st.download_button(
                    "Download missingness table",
                    data=missingness.to_csv(index=False),
                    file_name="airsense_cleaned_missingness.csv",
                    mime="text/csv",
                )
    else:
        st.warning("Cleaning summary files are not available. Rebuild the model from the supplied source archive.")

with tab_cleaned:
    st.markdown('<div class="section-kicker">Cleaned-data graphs · 10% of the rubric</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">These summaries are calculated after timestamp cleaning and -200 normalization. CO charts include measured target values only.</div>',
        unsafe_allow_html=True,
    )
    if co_distribution is not None and not co_distribution.empty:
        st.markdown("### Distribution of measured CO")
        st.caption("Hourly reference concentration in mg/m³; rows with missing CO are excluded from this chart.")
        st.bar_chart(
            co_distribution,
            x="bin_center_mg_m3",
            y="records",
            height=270,
            x_label="CO concentration (mg/m³)",
            y_label="Hourly records",
            color="#2f8273",
        )

    monthly_col, hourly_col = st.columns(2)
    with monthly_col:
        if co_by_month is not None and not co_by_month.empty:
            st.markdown("### Monthly mean CO")
            st.caption("Chronological pattern in the cleaned reference measurements.")
            st.line_chart(co_by_month.set_index("month")[["mean_CO_mg_m3"]], height=280)
    with hourly_col:
        if co_by_hour is not None and not co_by_hour.empty:
            st.markdown("### Mean CO by hour")
            st.caption("Average concentration by clock hour; this is descriptive, not a forecast.")
            st.bar_chart(co_by_hour, x="hour", y="mean_CO_mg_m3", height=280, x_label="Hour of day", y_label="Mean CO (mg/m³)", color="#c47a27")

    if correlation is not None and not correlation.empty and "variable" in correlation.columns:
        st.markdown("### Sensor and target correlations")
        st.caption("Pearson correlations use available cleaned values pairwise. Correlation shows association, not causation.")
        correlation_matrix = correlation.set_index("variable").astype(float)
        st.dataframe(correlation_matrix.round(2), width="stretch")

    with st.expander("Download the four-chart cleaning summary"):
        st.caption("A standalone vector image is also embedded in the project README.")
        if (APP_DIR / "cleaned_data_eda.svg").exists():
            st.download_button(
                "Download cleaned-data chart (SVG)",
                data=(APP_DIR / "cleaned_data_eda.svg").read_text(encoding="utf-8"),
                file_name="airsense_cleaned_data_eda.svg",
                mime="image/svg+xml",
            )

with tab_evaluation:
    st.markdown('<div class="section-kicker">Training, testing & evaluation · 40% of the rubric</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Model selection uses validation data. The final score comes from a later, '
        'time-ordered test window kept out of fitting and selection.</div>',
        unsafe_allow_html=True,
    )
    if test_metrics is not None and not test_metrics.empty:
        test_row = test_metrics.iloc[0]
        sample_count = int(test_row.get("Test samples", len(test_predictions) if test_predictions is not None else 0))
        metric_cols = st.columns(4)
        metric_cols[0].metric("Test R²", f"{float(test_row['R²']):.3f}")
        metric_cols[1].metric("Test MAE", f"{float(test_row['MAE']):.3f} mg/m³")
        metric_cols[2].metric("Test RMSE", f"{float(test_row['RMSE']):.3f} mg/m³")
        metric_cols[3].metric("Test records", f"{sample_count:,}")
        start = test_row.get("Test start", "")
        end = test_row.get("Test end", "")
        if pd.notna(start) and pd.notna(end) and str(start) and str(end):
            st.caption(f"Selected model: {test_row.get('Model', selected_model)} · Test period: {start} to {end}")
        else:
            st.caption(f"Selected model: {test_row.get('Model', selected_model)} · Units: CO in mg/m³")

    if test_comparison is not None and {"Model", "MAE", "RMSE"}.issubset(test_comparison.columns):
        st.markdown("### Test score against a simple baseline")
        st.caption("The baseline predicts the development-period mean CO for every test record. Lower MAE and RMSE are better.")
        st.bar_chart(
            test_comparison,
            x="Model",
            y="MAE",
            height=250,
            x_label="Test-set model",
            y_label="MAE (mg/m³)",
            color="#2f8273",
        )
        st.dataframe(
            test_comparison.rename(
                columns={"MAE": "MAE (mg/m³)", "MSE": "MSE ((mg/m³)²)", "RMSE": "RMSE (mg/m³)"}
            ).round(4),
            hide_index=True,
            width="stretch",
        )

    if test_predictions is not None and {"timestamp", "actual_CO_mg_m3", "predicted_CO_mg_m3"}.issubset(
        test_predictions.columns
    ):
        plot_data = test_predictions.copy()
        plot_data["timestamp"] = pd.to_datetime(plot_data["timestamp"], errors="coerce")
        chart = (
            plot_data.dropna(subset=["timestamp"])
            .set_index("timestamp")[["actual_CO_mg_m3", "predicted_CO_mg_m3"]]
            .rename(
                columns={
                    "actual_CO_mg_m3": "Measured CO (mg/m³)",
                    "predicted_CO_mg_m3": "Estimated CO (mg/m³)",
                }
            )
        )
        if not chart.empty:
            st.markdown("### Held-out actual vs estimated CO")
            st.caption("Hourly average concentration · UCI Air Quality dataset · test window")
            st.line_chart(chart, height=360)
            with st.expander("View the first 25 test records"):
                st.dataframe(plot_data.head(25), hide_index=True, width="stretch")
            st.download_button(
                "Download all test predictions",
                data=plot_data.to_csv(index=False),
                file_name="airsense_test_predictions.csv",
                mime="text/csv",
            )
            residuals = plot_data.dropna(subset=["actual_CO_mg_m3", "predicted_CO_mg_m3"]).copy()
            residuals["residual_mg_m3"] = residuals["actual_CO_mg_m3"] - residuals["predicted_CO_mg_m3"]
            if not residuals.empty:
                residual_counts, residual_edges = np.histogram(residuals["residual_mg_m3"], bins=20)
                residual_histogram = pd.DataFrame(
                    {
                        "Residual bin (mg/m³)": (residual_edges[:-1] + residual_edges[1:]) / 2.0,
                        "Test records": residual_counts,
                    }
                )
                st.markdown("### Test residual distribution")
                st.caption("Residual = measured CO − estimated CO; values centered near zero indicate smaller bias.")
                st.bar_chart(
                    residual_histogram,
                    x="Residual bin (mg/m³)",
                    y="Test records",
                    height=250,
                    color="#c47a27",
                )

    if validation_metrics is not None and {"Model", "MAE"}.issubset(validation_metrics.columns):
        st.markdown("### Validation comparison")
        st.caption("Lower validation MAE is better. These scores guided model selection; they are not test scores.")
        comparison = validation_metrics.copy().sort_values("MAE", ascending=True)
        chart = comparison.set_index("Model")[["MAE"]].rename(columns={"MAE": "Validation MAE (mg/m³)"})
        st.bar_chart(chart, height=270)
        comparison_display = comparison.rename(
            columns={
                "MAE": "MAE (mg/m³)",
                "MSE": "MSE ((mg/m³)²)",
                "RMSE": "RMSE (mg/m³)",
            }
        )
        st.dataframe(comparison_display, hide_index=True, width="stretch")
        st.caption(f"Selected for the saved model: {selected_model}.")
    else:
        st.info("Validation comparison file is not available in this project folder.")

    with st.expander("Search and cross-validation evidence"):
        bayesian_history = load_optional_csv(BAYESIAN_HISTORY_PATH)
        randomized_results = load_optional_csv(RANDOMIZED_RESULTS_PATH)
        if bayesian_history is not None and {"iteration", "cv_mae_mg_m3"}.issubset(bayesian_history.columns):
            st.markdown("#### Bayesian Ridge search")
            st.caption("Expanding-window time-series CV MAE for each tested regularization value; lower is better.")
            st.line_chart(bayesian_history.set_index("iteration")[["cv_mae_mg_m3"]], height=230)
            st.dataframe(
                bayesian_history.rename(columns={"alpha": "Ridge α", "cv_mae_mg_m3": "CV MAE (mg/m³)"}).round(5),
                hide_index=True,
                width="stretch",
            )
        if randomized_results is not None and {"iteration", "cv_mae_mg_m3", "parameters"}.issubset(randomized_results.columns):
            st.markdown("#### Extra Trees randomized search")
            ordered_search = randomized_results.sort_values("cv_mae_mg_m3", ascending=True).reset_index(drop=True)
            ordered_search["rank"] = np.arange(1, len(ordered_search) + 1)
            st.caption("Twelve Extra Trees configurations compared on training-only expanding-window folds.")
            st.bar_chart(
                ordered_search,
                x="rank",
                y="cv_mae_mg_m3",
                height=230,
                x_label="Ranked configuration",
                y_label="CV MAE (mg/m³)",
                color="#2f8273",
            )
            st.dataframe(
                ordered_search.rename(columns={"rank": "Rank", "cv_mae_mg_m3": "CV MAE (mg/m³)", "parameters": "Parameters"}),
                hide_index=True,
                width="stretch",
            )

with tab_method:
    st.markdown('<div class="section-kicker">Method & data</div>', unsafe_allow_html=True)
    st.markdown("## A time-aware regression workflow")
    st.markdown(
        "The target is the reference analyzer's hourly CO concentration, CO(GT), in mg/m³. "
        "The model estimates that same hour from the five sensor responses, temperature, relative humidity, "
        "absolute humidity, hour-of-day, and weekday."
    )
    process_cols = st.columns(4)
    stages = [
        ("01", "Chronological split", "70% training · 15% validation · 15% final test, by time."),
        ("02", "Search on training", "Gaussian-process Bayesian search for Ridge; randomized search for Extra Trees."),
        ("03", "Compare a stack", "Ridge and Extra Trees feed an expanding-window OOF meta-model; base hyperparameters are selected on the full training period first."),
        ("04", "Select and report", "Validation MAE selects the model; the later test period reports final performance."),
    ]
    for column, (number, heading, description) in zip(process_cols, stages):
        with column:
            st.markdown(f'<div class="section-kicker">Step {number}</div>', unsafe_allow_html=True)
            st.markdown(f"#### {heading}")
            st.caption(description)

    st.markdown("### Inputs and missing readings")
    st.write(
        "The fitted pipeline imputes missing sensor or weather readings using training-set medians. "
        "The source dataset's -200 missing-value marker is converted to missing before training. "
        "Other pollutant ground-truth columns are not used as model inputs."
    )
    st.markdown("### Dataset and scope")
    st.markdown(
        "The supplied records contain historical hourly readings from a sensor array deployed in an Italian city "
        "from 2004-03-10 through 2005-04-04. The data include cross-sensitivity, sensor drift, and missing values. "
        "This project is a research demonstration and should not be presented as a live monitor, AQI value, "
        "forecast, or public-health alert."
    )
    st.markdown(
        '<div class="source-note">Source: <a href="https://archive.ics.uci.edu/dataset/360/air+quality" target="_blank">'
        'UCI Machine Learning Repository · Air Quality</a> · DOI 10.24432/C59K5F. '
        'The UCI page lists CC BY 4.0 and also says commercial use is excluded; treat the data as research-only and check the current terms before reuse.</div>',
        unsafe_allow_html=True,
    )
    if MANIFEST_PATH.exists():
        with st.expander("View the training manifest"):
            try:
                st.json(MANIFEST_PATH.read_text(encoding="utf-8"))
            except OSError:
                st.caption("Training manifest could not be read.")
    st.markdown("### Rebuild the model")
    st.code(
        "python train_model.py --data path/to/AirQualityUCI.csv --output .",
        language="bash",
    )
    st.caption(
        "The script writes cleaning summaries, cleaned-data charts, the model bundle, validation/test metrics, a baseline comparison, search histories, and the training manifest."
    )

with st.sidebar:
    st.markdown(
        '<div class="brand-row"><span class="brand-name">AirSense</span></div>',
        unsafe_allow_html=True,
    )
    st.caption("EVS-03 · Model card")
    st.markdown("**Target**  \nSame-hour CO(GT), mg/m³")
    st.markdown(f"**Selected model**  \n{selected_model}")
    st.markdown(f"**Runtime**  \nscikit-learn {sklearn.__version__}")
    st.divider()
    st.markdown(
        "This interface uses a saved historical-data model. The estimate is not a live measurement, AQI conversion, "
        "or medical / regulatory advice."
    )
    st.markdown("[Dataset source](https://archive.ics.uci.edu/dataset/360/air+quality)")
