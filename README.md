# Time-series forecasting pipeline

[Open the interactive Kedro-Viz pipeline](https://weifengsiew.github.io/machine-learning-pipeline-time-series/?types=nodes&expandAllPipelines=false&pid=__default__)

## Problem

**Problem statement:** Walmart store managers need to estimate how many total
units the store will sell tomorrow. A reliable forecast helps them prepare the
right amount of stock and avoid shortages or excess inventory.

**Input:** Historical daily data for store `WI_2`, including:

- past daily sales totals
- recent 7 and 28-day averages
- calendar events
- the store's SNAP indicator

**Output:** Tomorrow's total sales for `WI_2`


**Main metric:** Root mean squared error (RMSE).

RMSE suits this forecasting problem because it is expressed in the same units
as sales and penalises large errors more heavily. A large under- or
over-forecast can create a meaningful inventory problem, so large misses should
affect the score more than several small misses.

## Data

We use the **M5 Forecasting Accuracy** dataset, which contains Walmart sales
and calendar data from 2011-01-29 to 2016-06-19. Download these files from the
[M5 Forecasting Accuracy competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data):

- `sales_train_evaluation.csv`
- `calendar.csv`

Place both files in `ts/data/m5-forecasting-accuracy/`. This directory is
gitignored because the source files are large and must not be committed.

## Data pipeline

The project now has one workflow: the Kedro post-June-2012 pipeline. It
aggregates the configured store, cleans and validates the daily series, removes
observations before `2012-06-01`, engineers leakage-safe forecasting features,
creates a chronological train/test split, tunes the regressors, evaluates naive
baselines, and writes the reporting outputs under `results/kedro/`.

## Modelling features

| Feature | Definition | Type |
|---|---|---|
| `snap` | Indicator showing whether SNAP benefits were issued in the store's state. | Non-engineered |
| `event_name` | Named calendar event on the forecast date; one-hot encoded. | Non-engineered |
| `event_type` | Category of the calendar event; one-hot encoded. | Non-engineered |
| `has_event` | Indicator equal to 1 when a calendar event occurs. | Engineered |
| `sales_lag_1_day` | Sales from the day immediately before the forecast date. | Engineered |
| `sales_lag_2_days` | Sales from two days before the forecast date. | Engineered |
| `sales_lag_1_week` | Sales from seven days before the forecast date. | Engineered |
| `sales_lag_2_weeks` | Sales from fourteen days before the forecast date. | Engineered |
| `sales_lag_4_weeks` | Sales from twenty-eight days before the forecast date. | Engineered |
| `sales_lag_1_month` | Sales from thirty days before the forecast date. | Engineered |
| `sales_lag_1_quarter` | Sales from ninety days before the forecast date. | Engineered |
| `sales_lag_1_year` | Sales from 365 days before the forecast date. | Engineered |
| `week_sin` | Sine encoding of the forecast date's week of year. | Engineered |
| `week_cos` | Cosine encoding of the forecast date's week of year. | Engineered |
| `month_sin` | Sine encoding of the forecast month. | Engineered |
| `month_cos` | Cosine encoding of the forecast month. | Engineered |
| `sales_rolling_mean_7_past` | Average sales over the seven most recent observed days. | Engineered |
| `sales_rolling_mean_28_past` | Average sales over the twenty-eight most recent observed days. | Engineered |

The future one-day sales value, `target_next_day`, is the prediction target and
is not used as an input feature. Event features are one-hot encoded, while the
configured numerical features are standardized using the training partition's
means and standard deviations.

The model search covers 40 configurations in total:

| Model | Grid focus | Combinations |
|---|---|---:|
| Linear regression | Intercept, positive coefficients, tolerance | 8 |
| Random forest | 5 `n_estimators` values × 2 `max_depth` values | 10 |
| Neural network | 6 one-, two-, and three-layer architectures × 2 `alpha` values | 12 |
| KNN | 5 `n_neighbors` values × 2 weighting strategies | 10 |

## Kedro pipeline

The post-June-2012 workflow is also exposed as a Kedro project. Its default
parameters retain observations from `2012-06-01`, split the test period at
`2014-10-01`, tune the four configured regressors with expanding-window
cross-validation, compare the learned models with both naive baselines, and
write reporting artifacts to `results/kedro/post_june_2012/`. Persisted Kedro
datasets follow the project structure under `data/raw_data/`,
`data/cleaned_data/`, and `data/feature_engineered_data/`. Model and reporting
artifacts are kept under `results/kedro/post_june_2012/`.

The modular stages are split into explicit nodes, following the reference
repository's stage-by-stage structure:

1. `data_ingestion` — `load_store_data` aggregates the M5 files into the configured store's daily series.
2. `data_cleaning` — `convert_datetime` → `sort_by_datetime` → `fill_event_nulls`.
3. `data_validation` — `build_expectation_suite`, then `validate_raw_data` and `validate_cleaned_data`.
4. `feature_engineering` — `filter_start_date` → `add_event_indicator` → `select_base_features` → `add_lag_features` → `add_cyclical_features` → `add_rolling_features` → `align_target_calendar_features` → `add_target`.
5. `train_test_split` — create the chronological split and fit preprocessing on training data only.
6. `model_training` — `build_candidate_models` → `build_candidate_grids` → `tune_candidate_models`, followed by complete-row selection, model comparison, baseline fitting, and model selection.
7. `model_evaluation` — create forecasts, metrics, and plots.

Each stage has its own `nodes.py` and `pipeline.py`. The
`src/time_series_pipeline/pipeline_registry.py` module discovers and composes
the stages into Kedro's `__default__` pipeline, following the reference
repository's modular structure.

## Repository architecture

| File | Responsibility |
|---|---|
| `src/time_series_pipeline/` | Kedro package and modular pipeline stages |
| `src/time_series_pipeline/pipelines/data_ingestion/nodes.py` | Store-level M5 ingestion |
| `src/time_series_pipeline/pipelines/data_cleaning/nodes.py` | Datetime and event-label cleaning |
| `src/time_series_pipeline/pipelines/feature_engineering/nodes.py` | Post-June-2012 feature engineering |
| `src/time_series_pipeline/pipelines/data_validation/expectations.py` | Code-defined Great Expectations suite |
| `src/time_series_pipeline/pipelines/data_validation/reporting.py` | Tabular validation reports for Kedro artifacts |
| `src/ml_model.py` | Model primitives and reporting plots used by Kedro nodes |
| `conf/base/catalog.yml` | Kedro dataset locations and output formats |
| `conf/base/parameters.yml` | Post-June-2012 Kedro run parameters |
| `.github/workflows/publish-kedro-viz.yml` | CI run and GitHub Pages publication |
| `tests/` | Unit, integration, and model-training tests |

## How to run

From the repository root, install the time-series project's locked
dependencies:

```bash
uv sync
```

Download `sales_train_evaluation.csv` and `calendar.csv` into
`data/m5-forecasting-accuracy/`, set `store_id` in
`conf/base/parameters.yml`, and run the required checks:

```bash
uv run ruff check
uv run pytest
```

Run the post-June-2012 Kedro pipeline with:

```bash
./run.sh
```

The same command is equivalent to:

```bash
uv run kedro run
```

Run an individual modular stage whose catalog inputs already exist with:

```bash
uv run kedro run --namespace feature_engineering
```

Launch the interactive visualisation locally with:

```bash
uv run kedro viz run
```

Kedro-Viz opens at `http://127.0.0.1:4141/`. Pushes to `main` run the workflow
and publish the same graph to GitHub Pages through
`.github/workflows/publish-kedro-viz.yml`.

The CI pipeline runs the same checks from the repository root:

```bash
uv sync --locked
uv run ruff check
uv run pytest
```

## Results

The verified Kedro run uses the post-June-2012 history and the shared test
window beginning on 2014-10-01. It selected linear regression by holdout test
RMSE; naive-baseline metrics are reported in original sales units.

### Kedro post-June-2012 variant

| Model | Train RMSE | Test RMSE |
|---|---:|---:|
| Linear regression | 370.11 | **456.80** |
| Neural network | 268.92 | 467.98 |
| Random forest | 162.41 | 599.32 |
| KNN | 0.00 | 636.18 |
| Naive, one day | 794.12 | 864.14 |
| Naive, one week | 1,024.30 | 1,140.12 |

The Kedro run saves comparison tables, tuning results, validation evidence,
forecast plots, a predicted-versus-actual scatter plot, forecast errors, fitted
models, and `metrics.json` under `results/kedro/post_june_2012/`.

Key artifacts include:

| Artifact | Description |
|---|---|
| `model_comparison.csv` | Train and test RMSE for tuned models and baselines |
| `tuning_results.csv` | Cross-validation RMSE and selected configurations |
| `validation_before_cleaning.json` | Complete Great Expectations result for raw data |
| `validation_after_cleaning.json` | Complete Great Expectations result for cleaned data; must pass |
| `validation_before_cleaning.csv` | Tabular Great Expectations results for raw data |
| `validation_after_cleaning.csv` | Tabular Great Expectations results for cleaned data |
| `predicted_vs_actual.png` | Selected-model predictions against actual sales with error guides |
| `forecast_errors.csv` | Actual values, forecasts, and forecast errors |
| `models/` | Serialized tuned models and naive baselines |

## Slides

[View the project slides](https://docs.google.com/presentation/d/1hl58SIdjG--_WyeirpIx7lrE6R3-HEeZWL_fmEr75XQ/edit?usp=sharing)
