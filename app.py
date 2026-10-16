from __future__ import annotations

from datetime import date
from html import escape
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
        --ink: #18312e;
        --muted: #60736f;
        --line: #dce5e1;
        --paper: #f3f6f4;
        --surface: #ffffff;
        --forest: #145f53;
        --forest-dark: #113c36;
        --mint: #e4f0eb;
        --amber: #8c5a11;
        --amber-bg: #fff4df;
    }
    html, body, [class*="css"] { font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    [data-testid="stAppViewContainer"] { background: var(--paper); color: var(--ink); }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 1420px; padding-top: 1.35rem; padding-bottom: 3.5rem; }
    [data-testid="stSidebar"] { background: #edf2ef; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.3rem; }
    h1, h2, h3, h4 { color: var(--ink); letter-spacing: -0.025em; }
    h1 { font-size: 2.1rem !important; font-weight: 690 !important; }
    h2 { font-size: 1.35rem !important; font-weight: 650 !important; }
    h3 { font-size: 1.05rem !important; font-weight: 650 !important; }
    p, li { color: var(--muted); }
    [data-testid="stTabs"] [role="tablist"] { gap: .35rem; border-bottom: 1px solid var(--line); }
    [data-testid="stTabs"] button[role="tab"] { padding: .8rem 1rem; color: var(--muted); font-weight: 620; }
    [data-testid="stTabs"] button[aria-selected="true"] { color: var(--forest); border-bottom-color: var(--forest); }
    [data-testid="stMetric"] { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 1rem 1.05rem; }
    [data-testid="stMetricLabel"] p { color: var(--muted); font-size: .79rem; font-weight: 650; letter-spacing: .035em; text-transform: uppercase; }
    [data-testid="stMetricValue"] { color: var(--ink); font-size: 1.65rem; font-weight: 690; }
    div[data-testid="stForm"] { border: 0; padding: 0; }
    div[data-testid="stVerticalBlockBorderWrapper"] { background: var(--surface); border-color: var(--line); border-radius: 14px; }
    div[data-testid="stButton"] button[kind="primary"], div[data-testid="stFormSubmitButton"] button[kind="primary"] { background: var(--forest); border-color: var(--forest); border-radius: 8px; font-weight: 650; min-height: 2.8rem; }
    div[data-testid="stButton"] button[kind="primary"]:hover, div[data-testid="stFormSubmitButton"] button[kind="primary"]:hover { background: var(--forest-dark); border-color: var(--forest-dark); }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
    .brand-row { display:flex; align-items:center; gap:.65rem; margin: .2rem 0 1.1rem; }
    .brand-mark { width:2rem; height:2rem; border-radius:9px; background:var(--forest); color:#fff; display:inline-flex; align-items:center; justify-content:center; font-size:1rem; font-weight:800; letter-spacing:-.08em; }
    .brand-name { color:var(--ink); font-weight:720; letter-spacing:-.02em; font-size:1.03rem; }
    .brand-code { margin-left:auto; color:var(--muted); font-size:.72rem; font-weight:700; letter-spacing:.09em; }
    .hero { display:flex; justify-content:space-between; align-items:flex-end; gap:2rem; background:var(--forest-dark); color:#fff; padding:1.7rem 1.85rem; border-radius:16px; margin:.4rem 0 1.25rem; }
    .hero h1 { color:#fff; font-size:2.1rem !important; margin:.35rem 0 .25rem; }
    .hero p { color:#d4e4de; margin:0; max-width:720px; line-height:1.55; }
    .eyebrow { color:#a6cec0 !important; font-size:.72rem; font-weight:750; letter-spacing:.13em; text-transform:uppercase; }
    .hero-side { min-width:205px; padding-left:1.4rem; border-left:1px solid rgba(255,255,255,.24); }
    .hero-side .small { color:#c0d7cf; font-size:.72rem; font-weight:650; letter-spacing:.07em; text-transform:uppercase; }
    .hero-side .model { color:#fff; font-size:1.05rem; font-weight:680; margin:.45rem 0 .25rem; }
    .hero-side .context { color:#c0d7cf; font-size:.78rem; line-height:1.45; }
    .section-kicker { color:var(--forest); font-size:.7rem; font-weight:750; letter-spacing:.12em; text-transform:uppercase; margin-bottom:.35rem; }
    .section-copy { color:var(--muted); font-size:.9rem; line-height:1.55; margin-top:-.15rem; margin-bottom:1rem; }
    .field-title { color:var(--ink); font-size:.82rem; font-weight:680; margin:.15rem 0 .15rem; }
    .field-unit { color:var(--muted); font-size:.7rem; margin:0 0 .3rem; }
    .estimate-panel { background:#eaf2ee; border:1px solid #cfdfd8; border-radius:13px; padding:1.25rem 1.3rem; margin:.2rem 0 .85rem; }
    .estimate-label { color:var(--forest); font-size:.72rem; font-weight:750; letter-spacing:.09em; text-transform:uppercase; }
    .estimate-value { color:var(--forest-dark); font-size:2.55rem; font-weight:730; letter-spacing:-.055em; line-height:1.1; margin:.5rem 0 .3rem; }
    .estimate-unit { color:var(--muted); font-size:.88rem; font-weight:560; letter-spacing:0; }
    .estimate-time { color:var(--muted); font-size:.78rem; }
    .quiet-note { background:#edf2ef; border-left:3px solid #83a99b; border-radius:0 8px 8px 0; padding:.75rem .85rem; color:#435c56; font-size:.82rem; line-height:1.5; }
    .source-note { color:#73827e; font-size:.75rem; line-height:1.5; }
    @media (max-width: 800px) {
        .hero { align-items:flex-start; flex-direction:column; gap:1rem; padding:1.35rem; }
        .hero-side { border-left:0; border-top:1px solid rgba(255,255,255,.24); padding:1rem 0 0; width:100%; }
        .hero h1 { font-size:1.7rem !important; }
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
        '<div class="brand-row"><span class="brand-mark">A</span>'
        '<span class="brand-name">AirSense</span><span class="brand-code">EVS—03</span></div>'
        '<div class="hero"><div><div class="eyebrow">Sensor array · CO regression</div>'
        '<h1>Read the hour. Estimate CO.</h1>'
        '<p>A same-hour carbon monoxide estimate from five sensor channels and local weather readings. '
        'Built for exploring a trained model on historical measurements.</p></div>'
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

tab_predict, tab_performance, tab_method = st.tabs(["Estimate", "Performance", "Method & data"])

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

with tab_performance:
    st.markdown('<div class="section-kicker">Evaluation</div>', unsafe_allow_html=True)
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
        ("03", "Build a stack", "Ridge and Extra Trees feed a meta-model trained on expanding-window out-of-fold predictions."),
        ("04", "Select and report", "Validation MAE selects the model; the later test period reports final performance."),
    ]
    for column, (number, heading, description) in zip(process_cols, stages):
        with column:
            with st.container(border=True):
                st.markdown(f'<div class="section-kicker">{number}</div>', unsafe_allow_html=True)
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
        'The repository describes the dataset as for research use and excludes commercial use; consult its terms before reuse.</div>',
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
        "python train_model.py --data path/to/archive.zip --output .",
        language="bash",
    )
    st.caption(
        "The script writes the model bundle, validation/test metrics, test predictions, search histories, and training manifest."
    )

with st.sidebar:
    st.markdown(
        '<div class="brand-row"><span class="brand-mark">A</span>'
        '<span class="brand-name">AirSense</span></div>',
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
