# M5 Forecasting - Accuracy: method, experiments and how to reproduce

A summary with all figures is in the [main README](../README.md). This file has the details: the data, the fixed roles,
the four variants, the numbers, the Kaggle scoring and the exact commands.

## Folder map

| Path | Contents |
| --- | --- |
| `src/m5lib.py` | Compact array file from the competition files, the exact 12-level WRMSSE (`WRMSSE`, `Scorer`), cumulative-sum tables for trailing-window statistics (`Hist`), the feature builder (`features`), training-cell sampling and the ratio alignment (`align`). |
| `src/cpu_variants.py` | V1 to V4 and the aggregate-level model: per-store LightGBM runs in a process pool, early stopping on the *choose* window, forecasts for the three origins, the alignment choice, atomic history file. |
| `scripts/build_prep.py` | Builds `prep/m5_prep.npz` (about 40 MB) from the competition files and caches the WRMSSE scale and weights. |
| `scripts/analyze.py` | Turns the saved forecasts into the aggregate tables in `results/`. |
| `scripts/make_figures.py` | Draws the six figures from those tables. |
| `scripts/make_submission.py` | Writes the Kaggle CSV of one variant. |
| `tests/test_m5.py` | Leak test with a mutant (corrupting every day from the origin on must not change any feature), a check that training cells end by the last fit day, the alignment identity, and a check that the scorer is zero for a perfect forecast and equals the reference WRMSSE class. |
| `results/` | `metrics.csv`, `per_level.csv`, `aggregate_model_per_level.csv`, `per_horizon.csv`, `bias_total.csv`, `bias_by_volume.csv`, `store_iterations.csv`, `cpu_history.json`, `cpu_run.log`, `data_profile.txt`, `kaggle_scores.csv`, `leaderboard.png`. The per-series forecasts are not published. |
| `figures/` | The six figures shown in the main README. |

## The task and the data

Forecast the daily unit sales of 30,490 item-store series (3,049 items, 10 stores in 3 states) for the next 28 days.
The data has 1,941 training days (d_1 to d_1941), the calendar with events and SNAP days, and weekly sell prices. The score
is the **weighted root mean squared scaled error** over 12 aggregation levels (42,840 series in total, from the grand total
to item by store):

RMSSE_i = sqrt( mean_h (y_ih - ŷ_ih)^2 / mean_t (y_it - y_i,t-1)^2 ),  WRMSSE = (1/12) Σ_levels Σ_i w_i RMSSE_i,

the denominator taken over the history after the first sale of the series, and w_i the share of the series in the dollar
sales of the last 28 known days within its level. Facts measured by `scripts/build_prep.py` and the profile in
[`results/data_profile.txt`](results/data_profile.txt):

- 68.0 % of the item-store-day cells are zero; 76 % of the series are intermittent by the Syntetos-Boylan classes (23,075 intermittent, 5,938 lumpy).
- 77.1 % of the series have their first sale after d_1, and 16.2 % after d_1000, so many series are young.
- The top 10 % of the series hold 48.8 % of the dollar weight at the item-store level.
- Total sales have a strong weekday profile (Saturday and Sunday about 1.4 times a Wednesday) and a SNAP effect on food (ratios 1.10, 1.16 and 1.30 in the three states).
- The mean daily total grew from 26,280 in 2011 to 41,835 in 2016 (flat in 2014), so the level of a series moves.

Trivial forecasts on the report window: the mean of the last 28 days gives WRMSSE 1.082, repeating the last 28 days 0.838,
and the **mean of the last 28 days times the weekday profile of the last 52 weeks gives 0.7161**. This last one is the
reference called *baseline* below.

## Fixed roles

All models are fitted on targets up to **d_1885** only. A forecast origin is the number of known days:

| Role | Origin | Forecast days | Used for |
| --- | --- | --- | --- |
| fit | up to 1857 (target by d_1885) | - | training cells |
| choose | 1885 | d_1886 to d_1913 | early stopping, alignment strength |
| report | 1913 | d_1914 to d_1941 | the reported number (also the Kaggle public window) |
| submit | 1941 | d_1942 to d_1969 | the Kaggle private window (no labels in the data) |

The fitted models are rolled to the later origins without refitting, so the report window is never used for fitting or
choosing. The weights and scales of the WRMSSE are those of the role (history before the origin, dollar sales of the 28
days before it).

## The four variants

**V1 and V2: bottom-up LightGBM per store.** One model per store (10), direct over the horizon: a training cell is
(series, origin, horizon h in 1..28) and the horizon is a feature. Cells come from the 52 origins 1129, 1143, ..., 1843 with
two random horizons per series and origin, and only cells whose target day is at most d_1885 are kept (about 317 k cells per
store). Features use only days before the origin: trailing means over 7, 14, 28, 56, 112 and 224 days, standard deviation
of the last 28 days, the share of non-zero days over 28 and 112 days, days since the last sale, the last value, the mean of
the last four same-weekday values, the ratio of the 28-day to the 224-day mean, the age since the first sale, the horizon,
and for the target day the weekday, month, day of month, event type, SNAP flag, price relative to the series' maximum and to
the price at the origin, and the item, department, category and state codes. LightGBM: learning rate 0.05, 63 leaves, at least
100 cells per leaf, feature and bagging fraction 0.8, L2 1, 127 bins, up to 400 rounds, early stopping after 50 rounds on the
RMSE of the *choose* window. V1 uses the squared error; V2 the Tweedie loss with variance power 1.1. The 30,490 forecasts
are the bottom level and every aggregate is their sum.

**V3: V2 reconciled to a separate aggregate-level model.** One more LightGBM (same settings) is trained on all non-total
series of the hierarchy, 42,839 series including the item-store level, with every lag feature and the target divided by
`1 + trailing 28-day mean` and the aggregation level as a feature, so that series of very different size share one scale.
Its forecasts for the 70 store-by-department groups are compared with the sums of V2; the bottom forecasts of a group are
multiplied by `clip(ref / sum, 0.7, 1.4)^alpha` per horizon day, with alpha chosen on the *choose* window from {0, 0.5, 1}.
**The choose window picked alpha = 0, so V3 is identical to V2.**

**V4: lag-free LightGBM per store.** Following the top-down alignment paper, the bottom model has no rolling or lagged
demand features. A cell is (series, day) over the last 728 fit days (about 1.1 M cells per store after keeping a random
half) with the day's weekday, month, day of month, event type, SNAP flag, a linear day index, the week of the year, price
relative to the series' maximum and to the price at the end of the fit period, and the item, department and category codes.
The forecast for a role is the model evaluated on the 28 calendar days of that role. The same ratio alignment as in V3 is
available toward the independent total forecast (the sum of the three state forecasts of the aggregate model), with alpha
chosen on the *choose* window. **The choose window picked alpha = 0 again, so V4 is the unaligned lag-free model and the
aggregate model did not influence any published forecast.** The Kaggle file is named `..._lagfree_aligned` after the design,
not the outcome.

## Results

| Model | What it is | Choose WRMSSE | Report WRMSSE |
| --- | --- | --- | --- |
| baseline | mean of last 28 days x weekday profile | 0.6430 | 0.7161 |
| V1 | per-store LightGBM, squared loss | 0.6689 | 0.7772 |
| V2 | same, Tweedie loss | 0.6617 | 0.7938 |
| V3 | V2 plus reconciliation (alpha chosen 0) | 0.6617 | 0.7938 |
| V4 | lag-free per-store LightGBM (alpha chosen 0) | **0.5856** | **0.6791** |

Per-level report numbers are in [`results/per_level.csv`](results/per_level.csv). Observations that the tables support:

- Only V4 beats the baseline, on both windows. V1 to V3 are worse than the baseline on both (0.662 to 0.669 against 0.643 on *choose*, 0.777 to 0.794 against 0.716 on *report*).
- V4 wins at the aggregate levels (0.50 at the total, 0.72 at store by department) and loses at the three item levels (0.97, 0.92 and 0.88 against 0.87, 0.85 and 0.84 for V1); there the baseline scores 0.85.
- The Tweedie loss was worse than the squared loss on the report window (0.794 against 0.777) and better on *choose* (0.662 against 0.669); one 28-day window per role cannot separate them.
- Every model forecasts too little in total over the report window (sum of forecasts over sum of actuals 0.93 for V1, 0.92 for V2, 0.96 for V4 and the baseline), and every model, the baseline included, has its largest daily error on forecast day 21 (V4 1.27, baseline 1.16).
- The Tweedie model forecasts 0.45 of the actual sales in the lowest-volume decile of the item-store series, V4 forecasts 1.23 to 1.51 times the actual in the three lowest deciles and 0.91 in the highest (`results/bias_by_volume.csv`).
- Early stopping on the *choose* window kept only 48 to 270 trees per store for V1 and 52 to 218 for V2, a sign that the choose window drifts from the fit cells fast.

## Kaggle scoring

The competition has closed. The V4 forecasts were submitted late as a CSV (validation rows from origin 1913, evaluation rows
from origin 1941):

| Column | Value |
| --- | --- |
| Public score (d_1914 to d_1941, the *report* window) | **0.67907** |
| Private score (d_1942 to d_1969) | **0.60711** |
| Local report WRMSSE | 0.6791 |

The public score equals the local report-window score to four digits, which confirms that `Scorer` reproduces the
competition's WRMSSE. The model scores lower (better) on the private window than on the public one. Sources:
[`results/kaggle_scores.csv`](results/kaggle_scores.csv) and [`results/leaderboard.png`](results/leaderboard.png).

## Experiment notes

- A first run used learning rate 0.1 and a patience of 30 and scored V1 0.6691 / 0.7812, V2 0.6644 / 0.7954 and V4 0.5730 /
  0.6664 (choose / report). The settings were then changed to learning rate 0.05 and patience 50 and the run was repeated; this
  second run is the one reported. That change was made after seeing the report numbers of the first run, and it did not help V4
  (report 0.6664 to 0.6791), so the choice is not a tuning gain; the first run was not submitted.
- In the first implementation of the aggregate model the intended selection (aggregate levels only) was written as "all series
  except the total", which includes the item-store level. The model therefore trains on 42,839 series. Because alpha = 0 was chosen
  in both uses, it did not change any reported number.
- **The aggregate-level model was itself defective, so V3 is not a fair test of reconciliation.** Evaluated alone, its forecasts score
  1.54 (choose) and 1.70 (report) on average over the nine aggregate levels, against 0.61 and 0.75 for V1 and 0.48 and 0.60 for V4
  ([`results/aggregate_model_per_level.csv`](results/aggregate_model_per_level.csv)). Early stopping kept only 26 trees: the
  validation rows are dominated by the 30,490 item-store series, whose scaled target (about 0.5) differs from the aggregates'
  (about 1), so the model stopped before it had moved off the pooled mean. That is why alpha = 0 was chosen, and it is not
  evidence about reconciliation to a good aggregate forecast.
- A mutant test (corrupting all days from the origin on) found no feature that reads the future.

## Compute

V1 to V4 and the aggregate model ran in **98 s** on a **4-core, 16 GiB cloud CPU sandbox** (two stores in parallel with two
threads each); no GPU was used. Building the array file takes about 80 s on a workstation.

## Reproducing

```bash
cd m5-forecasting-accuracy
pip install -r requirements.txt
export M5_DATA_DIR=/path/to/m5-forecasting-accuracy        # calendar.csv, sell_prices.csv, sales_train_evaluation.csv, sample_submission.csv
python scripts/build_prep.py                                # prep/m5_prep.npz
python -m pytest tests -q
python src/cpu_variants.py --cores 4 | tee results/cpu_run.log   # results/preds/*.npy, results/cpu_history.json
python scripts/analyze.py && python scripts/make_figures.py
python scripts/make_submission.py --variant V4_lagfree_aligned
```

Requirements: Python 3.11 or newer, LightGBM 4, 16 GiB of memory for the aggregate model. The competition data is not
redistributed here.

## Limitations

- One 28-day window per role and no confidence intervals: the order of V1, V2 and V3 on one window is not a reliable ranking.
- The models are not refitted for later origins; the private window is forecast with weights fitted by d_1885.
- The alignment and reconciliation steps both came out as no-ops because the aggregate-level model behind them was defective (see the experiment notes); neither idea has been tested with a working aggregate model.
- V4 has no recent-demand information at all besides the item code, so it cannot follow a series whose level has just changed.
- Two neural variants trained on a GPU are not part of this folder.
