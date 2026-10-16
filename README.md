# AirSense — EVS-03 hourly CO estimator

AirSense is a small, local research demo for estimating same-hour carbon monoxide (CO) concentration from a chemical sensor array. It includes a trained regression model, a user interface for entering sensor readings, a held-out evaluation view, and a reproducible training workflow.

## Why this project

Low-cost gas sensors respond to more than one gas and can drift over time. This project explores whether the array response and weather context can estimate the reference analyzer's hourly CO reading. The product makes the model's estimate and evaluation visible, while keeping the historical-data limits clear.

## What is included

- `app.py` — responsive Streamlit interface for same-hour estimates, optional missing readings, a demo-only comparison limit, session history, and evaluation charts.
- `evs03_model_bundle.joblib` — selected fitted model pipeline and feature metadata.
- `evs03_validation_metrics.csv` — candidate comparison on the validation period.
- `evs03_test_metrics.csv` and `evs03_test_predictions.csv` — final held-out evaluation and timestamped predictions.
- `train_model.py` — full data preparation, time-series model search, stacking, selection, and evaluation workflow.
- `training_manifest.json`, `bayesian_search_history.csv`, and `randomized_search_results.csv` — evidence of the training choices and search runs.
- `requirements.txt` and `run_demo.bat` — local setup files for Windows.

## Run the demo on Windows

1. Install Python 3.11.
2. Keep all project files together.
3. Double-click `run_demo.bat`.

On first launch, the script creates a local virtual environment, installs the pinned dependencies, and opens the app. The model bundle was saved with scikit-learn 1.6.1.

## Rebuild the model

Download the UCI dataset file or archive from the [UCI Air Quality dataset page](https://archive.ics.uci.edu/dataset/360/air+quality). The raw dataset is not included in this repository.

On Windows, run the training launcher with a CSV, XLSX, or ZIP path:

```powershell
.\train_model.bat "C:\path\to\AirQualityUCI.csv"
```

Or with the project dependencies installed, run:

```powershell
python train_model.py --data "C:\path\to\AirQualityUCI.csv" --output .
```

The script accepts a CSV, XLSX, or ZIP containing the UCI Air Quality file (including `AirQuality.csv` and `AirQualityUCI.csv`). It writes the fitted model bundle, search histories, validation metrics, held-out test metrics and predictions, and a JSON training manifest.

## Training and evaluation

- **Target:** `CO(GT)`, the reference analyzer's hourly CO concentration in mg/m³.
- **Inputs:** five PT08 sensor response channels; temperature, relative humidity, and absolute humidity; hour-of-day and weekday encoded as sine/cosine pairs.
- **Leakage control:** target-missing rows are excluded from scoring; the source's `-200` missing marker is converted to missing. Ground-truth columns for other pollutants are not used as inputs. Median imputation is fitted inside each model pipeline.
- **Time split:** chronological 70% training, 15% validation, and 15% final test by source row order. The last period is held out from search, model choice, and fitting.
- **Search:** a Gaussian-process Bayesian search selects Ridge regularization; `RandomizedSearchCV` tunes an Extra Trees model. Both use four expanding-window `TimeSeriesSplit` folds on training data only.
- **Stacking:** Ridge and Extra Trees base learners are combined with a Ridge meta-model trained only on expanding-window out-of-fold predictions.
- **Selection:** the candidate with the lowest validation MAE is selected. Final metrics are then measured on the later test period. Negative CO outputs are floored at 0 mg/m³ for display and metric calculation because concentration cannot be negative; the raw values are retained in the prediction CSV.

The exact selected model, sample counts, dates, and scores are in `training_manifest.json` and the evaluation CSV files. Do not compare validation scores with final test scores as if they came from the same period.

### Current saved model

The selected model is **Extra Trees**, chosen because it had the lowest validation MAE among the candidates. On the later held-out period, the saved model reports:

| Test period | Records | R² | MAE | RMSE |
| --- | ---: | ---: | ---: | ---: |
| 2005-02-05 03:00 to 2005-04-04 14:00 | 1,374 | 0.7831 | 0.3549 mg/m³ | 0.6127 mg/m³ |

Validation MAE was 0.6087 mg/m³ for Extra Trees, 0.6118 mg/m³ for the time-safe stack, and 0.6123 mg/m³ for Ridge. These validation scores are close; the candidate choice follows the stated lowest-validation-MAE rule. The untouched test MAE is reported separately.

## Dataset source and limits

Dataset citation: De Vito, S. et al. (2008), [UCI Machine Learning Repository: Air Quality](https://archive.ics.uci.edu/dataset/360/air+quality), DOI [10.24432/C59K5F](https://doi.org/10.24432/C59K5F). The supplied records span 2004-03-10 through 2005-04-04. The UCI page identifies a CC BY 4.0 license and also says commercial purposes are excluded; treat the data as research-only and check the current source terms before reuse.

This is a historical research demonstration, not a live monitor, next-hour forecast, AQI calculation, or health alert. The editable threshold in the app is only a display setting. The model has not been established for another city, sensor device, or present-day conditions. Its regression output is not a substitute for reference measurements or professional guidance.

## References

- [UCI Air Quality dataset](https://archive.ics.uci.edu/dataset/360/air+quality)
- [GOV.UK dashboard guidance](https://brand.design-system.service.gov.uk/data/dashboards/)
- [GOV.UK chart guidance](https://brand.design-system.service.gov.uk/data/charts/)
- [U.S. Web Design System data visualization guidance](https://designsystem.digital.gov/components/data-visualizations/)
- [U.S. EPA AirNow interactive map](https://gispub.epa.gov/airnow/)
