# Time-series forecasting pipeline

[Open the interactive pipeline visualisation](https://weifengsiew.github.io/machine-learning-pipeline-time-series/?types=nodes&expandAllPipelines=false&pid=__default__)

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

## Exploratory data analysis (EDA)

For the full exploratory analysis, see [`eda.ipynb`](eda.ipynb).

<img src="docs/plots/eda_sales_across_years.png" alt="WI_2 daily sales across the historical years with rolling means" width="700">

*EDA Figure 1. WI_2 daily sales across the historical years, with 7-day and 28-day
rolling means.*

The figure shows lower and less variable sales before June 2012, followed by a
higher and more variable regime. It also shows recurring short-term movement
in the rolling means. This informed two actions: remove pre-June observations
from the modelling data, and use past sales as predictors.

<img src="docs/plots/eda_seasonal_components.png" alt="Weekly, monthly, quarterly, and yearly seasonal components of WI_2 sales" width="700">

*EDA Figure 2. Seasonal components estimated at weekly, monthly, quarterly, and
yearly periods.*

The figure shows recurring movement in sales at several time scales. The weekly
panel has the clearest repeated cycle, while the monthly, quarterly, and yearly
panels show slower patterns that recur over time. This informed the decision to
add sales lags at the corresponding horizons and cyclical calendar features to
the modelling data.

## Pipeline

### Pipeline stages

The pipeline uses the post-June-2012 history and runs through these
stages:

1. `data_ingestion` — aggregate the configured store's daily sales series.
2. `data_cleaning` — convert dates, sort records, fill event nulls, and remove observations before `2012-06-01`.
3. `data_validation` — validate raw data and cleaned data.
4. `feature_engineering` — create forecasting features from the cleaned data.
5. `chronological_train_test_split` — separate the future target and create the chronological train/test split.
6. `preprocessing` — fit transformations on training data only and apply them to both partitions.
7. `model_training` — tune candidate regressors, compare baselines, and select the best model.
8. `model_evaluation` — create forecasts, metrics, and plots.

<img src="assets/Screenshot%202026-10-09%20at%2010.51.51%E2%80%AFPM.png" alt="Kedro pipeline visualisation" width="400">

[Open the Kedro pipeline visualisation](https://weifengsiew.github.io/machine-learning-pipeline-time-series/?types=nodes&expandAllPipelines=false&pid=__default__)

### Data cleaning

| Pipeline node | Cleaning action | Evidence from EDA | Result |
|---|---|---|---|
| `convert_datetime` | Convert dates to datetime | Dates are parseable, unique, sorted, and complete. | Reliable datetime values for time-series operations. |
| `sort_by_datetime` | Sort by date and use a datetime index | The date field defines the complete chronological order. | Explicit ordering for lags, rolling features, and splits. |
| `fill_event_nulls` | Fill event-label nulls with `No event` | Event names and types are missing together on non-event days. | Expected non-events are represented explicitly. |
| `filter_start_date` | Remove observations before `2012-06-01` | Sales coverage is materially lower before June 2012, creating a separate low-sales regime. | The cleaned modelling table uses the post-June-2012 history. |

### Data validation

The shared validation suite checks the following expectations against both
tables:

| Expectation | Columns or rule | Raw data | Cleaned data |
|---|---|---|---|
| Expected columns and order | The store-level column list matches the configured ordered list. | Pass | Pass |
| No null values | All calendar, event, SNAP, and sales columns are non-null. | 4 expected event-label failures | Pass |
| Integer types | `wm_yr_wk`, `wday`, `month`, `year`, `snap`, and `sales` are `int64`. | Pass | Pass |
| String types | Event name and event type columns contain strings. | Pass | Pass |
| Calendar and sales ranges | `wday` is 1–7, `month` is 1–12, `year` is 2011–2016, and `sales` is non-negative. | Pass | Pass |
| SNAP values | `snap` is either 0 or 1. | Pass | Pass |

### Modelling features

| Feature group | Features | Type | Purpose |
|---|---|---|---|
| Calendar inputs | `snap`, `event_name`, `event_type` | Non-engineered | Provide the forecast date's observed SNAP and event information; event columns are one-hot encoded. |
| Event indicator | `has_event` | Engineered | Represents whether a calendar event is present. |
| Sales lags | Top 30 ACF-selected lags: 1, 2, 3, 6, 7, 27–31, 33–35, 61–64, 90–92, 119–120, 153–154, 181–183, 245, 273, and 365 days. | Engineered | Capture the strongest short-, monthly-, quarterly-, semiannual-, and annual-scale dependence identified in the EDA. |
| Cyclical calendar features | `week_sin`, `week_cos`, `month_sin`, `month_cos` | Engineered | Represent recurring calendar cycles continuously. |
| Past-only rolling features | `sales_rolling_mean_7_past`, `sales_rolling_mean_28_past` | Engineered | Summarise recent history without using the current target. |
| Forecast target | `target_next_day` | Target, not a feature | Stores next-day sales separately from the model inputs. |

The ACF indicates serial dependence in sales. The pipeline uses the 30
non-zero lags with the strongest absolute ACF values, including recent,
monthly, quarterly, semiannual, and annual horizons. These lag features provide
the model with the strongest recurring historical signals identified in the
EDA.

### Preprocessing and model training

The future target is kept separate from the input features. Event features are
one-hot encoded, and numerical features are standardized using statistics fit
on the training partition only. The chronological test period begins at
`2014-10-01`.

The hyperparameter search covers 40 configurations:

| Model | Search space | Combinations |
|---|---|---:|
| Linear regression | Intercept, positive coefficients, tolerance | 8 |
| Random forest | `n_estimators` × `max_depth` | 10 |
| Neural network | Layer architectures × `alpha` | 12 |
| KNN | `n_neighbors` × weighting strategy | 10 |

## Results

The results use the post-June-2012 history and the shared test window beginning
on 2014-10-01. Linear regression has the lowest holdout test RMSE among the
learned models; naive baselines are included for comparison.

<img src="docs/plots/forecast.png" alt="Test-period forecast compared with actual sales and naive baselines" width="600">

*Figure 1. Selected-model and naive-baseline forecasts compared with actual
sales over the final 90 days of the test period.*

Linear regression's predicted sales closely track actual sales. The one-day
and one-week naive baselines deviate more from actual sales.

<img src="docs/plots/predicted_vs_actual.png" alt="Predicted sales versus actual sales with an empirical 90 percent prediction interval" width="450">

*Figure 2. Selected-model predictions versus actual sales across the test
period. Dashed lines show perfect prediction and the empirical 90% prediction
interval formed from the 5th and 95th percentiles of forecast error.*

The predictions track actual sales reasonably well. On roughly 90% of test
days, actual sales fall within the empirical prediction interval; the remaining
days fall outside it. The interval uses observed test-period errors, so its
coverage is descriptive rather than a calibrated probability for future days.

<img src="docs/plots/model_comparison.png" alt="Test RMSE comparison across models and baselines" width="600">

*Figure 3. Test RMSE for tuned regressors and naive baselines; lower values are
better.*

All tuned regressors outperform both naive baselines. Linear regression has the
lowest holdout RMSE among the learned models and is the selected model for this
store.

The pipeline saves comparison tables, tuning results, validation evidence,
forecast plots, a predicted-versus-actual scatter plot, forecast errors, fitted
models, and `metrics.json` under `results/`.

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

## Conclusion

Linear regression has the lowest holdout RMSE among the learned models, and all
tuned models outperform the one-day and one-week naive baselines. The
leakage-safe lag and rolling features capture recurring demand patterns without
using future sales. The prediction interval shows that the model tracks sales
reasonably well on most days, while the days outside the interval identify
larger forecasting misses.

## Future work

- Calibrate the prediction interval to a coverage level that reflects the
  business cost of stockouts and excess stock.
- Extend the forecast from total store sales to item- and category-level
  predictions, especially for festive and perishable goods where overstocking
  is more specific to the item.
- Explore recurrent neural networks and LSTMs. These models process observations
  in chronological order and can learn longer temporal dependencies directly;
  LSTMs use gates to retain the most useful historical information.

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

Run the pipeline with:

```bash
./run.sh
```

The interactive visualisation is available locally through the project's
visualisation command. Pushes to `main` publish the same graph to GitHub Pages.

The CI pipeline runs the same checks from the repository root:

```bash
uv sync --locked
uv run ruff check
uv run pytest
```
