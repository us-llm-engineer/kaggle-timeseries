# Optiver realized volatility: method, experiments and how to reproduce

A summary with all figures is in the [main README](../README.md). This file has the details: the data, the fixed roles,
the features, the four model families, the experiment table, the compute, the Kaggle scoring and the exact commands.

## Folder map

| Path | Contents |
| --- | --- |
| `src/opt14/features.py` | Per-stock feature and sequence builder (vectorised): realised volatility over cumulative windows, bipower, quarticity, 1 s forward-filled grid and sub-sampled variants, pre-averaging estimators, book spread/imbalance/depth statistics, trade statistics, 120 x 13 sequence tensor, and the same-`time_id` cross-stock aggregates (`add_cross`). Uses only the window's own rows: no target, no other window. |
| `src/opt14/common.py` | Data access (a local directory in the Kaggle layout, or a Hugging Face dataset repo), the deterministic role split by `time_id`, RMSPE and the optimal scalar. |
| `src/opt14/build.py` | Runs `features.py` over all stocks with a process pool and writes `features.parquet` (and the sequence tensor for V3). |
| `src/opt14/v1_ridge.py` | V1: ridge regression on log target, plain volatility features versus plus pre-averaging estimators. |
| `src/opt14/v2_gbdt.py` | V2: LightGBM on log target, with and without the cross-stock features; chunked training, early stopping on the *choose* role, atomic history file. |
| `src/opt14/nn_models.py` | V3 (1-D CNN over the second-level tensor plus aggregated features) and V4 (permutation-equivariant transformer over the stocks of one `time_id`), and the percentage-error loss. |
| `src/run_cpu.py`, `src/run_nn.py`, `src/run_gpu.py` | Runners: feature build + V1 + V2 on CPU; one neural worker; the GPU pipeline that builds features and then runs V3 and V4 as parallel processes beside a GPU sampler. `src/gpu_sampler.py`, `src/cpu_status.py`, `src/gpu_status.py` record and read utilisation and progress. |
| `scripts/investigate_phase1.py`, `scripts/analyze_phase1.py` | Data profile: sizes, target structure, baselines, the shared market factor. Output: `results/data/data_profile.txt`. |
| `scripts/aggregate_predictions.py` | Turns per-row predictions into the aggregate CSVs in `results/post/` and `results/metrics.csv` (per-row predictions are not published). |
| `scripts/make_figures.py` | Draws the six figures from the shipped result files. |
| `scripts/package_submission.py`, `scripts/build_colab_notebook.py` | Author-side helpers: assemble the weights dataset for the Kaggle kernel; generate the GPU-host notebook. |
| `notebooks/inference.py` | The Kaggle inference script (blend of V2, V3 and V4 with a self-check on the training files). |
| `tests/` | Feature, estimator, split and neural-network tests, including an input-influence probe and a stock-permutation check. |
| `results/` | `metrics.csv`, `post/` aggregate error tables, `cpu/` and `gpu/` per-epoch histories, logs, GPU samples, feature importance, `data/` data profile, `kaggle_scores.csv`. |
| `figures/` | The six figures shown in the main README. |

## The task and the data

For each of 112 stocks and about 3,830 ten-minute windows (`time_id`), the data has the top two levels of the limit-order
book at one-second resolution and the executed trades. The label is the realised volatility of the **next** ten minutes of
the same stock. 428,932 windows are labelled. The score is the root mean square **percentage** error,

RMSPE = sqrt( mean( ((y - ŷ) / y)^2 ) ),

so an under-prediction of a small label costs as much as the same relative error of a large one. Facts measured by
[`scripts/investigate_phase1.py`](scripts/investigate_phase1.py) (full output in
[`results/data/data_profile.txt`](results/data/data_profile.txt)):

- The label is right-skewed (skew 2.82; 0.22 after the logarithm), median 0.00305.
- The book is sparse: the median window has 388 of 600 seconds, 97.6 % of windows have gaps.
- The realised volatility of the weighted-average-price returns inside the window correlates 0.911 with the log label;
  with one fitted multiplicative constant (0.837) it already scores RMSPE 0.2858 on all labelled rows.
- A `time_id`'s cross-stock mean explains 55.6 % of the log-label variance: windows share a market-wide factor.
- `time_id`s carry no temporal order (autocorrelation of the window-mean label between adjacent ids: -0.014), so
  lag features across windows do not exist and the validation cannot be chronological.

## Fixed roles

A deterministic hash of `time_id` assigns every window to one role, so no `time_id` appears in two roles:

| Role | Share | Used for |
| --- | --- | --- |
| fit | 70 % | training |
| choose | 15 % | early stopping, calibration scalar, blend weights |
| report | 15 % | read for the reported number; monitor-only during training |

Every model predicts `log y` (V1, V2) or `log y` through a skip connection from the window's own log realised
volatility (V3, V4). The final prediction is multiplied by one scalar fitted on the *choose* role:
`c = sum(p/y) / sum((p/y)^2)` is the minimiser of RMSPE for a prediction `c·p`, and it is below 1 because the percentage
loss punishes over-prediction of small labels more than under-prediction.

## Features

117 columns per window (`src/opt14/features.py`):

- **Realised measures:** volatility of the WAP returns over the first 100, 200, ..., 600 seconds, the last 300 and 150
  seconds, the second book level, bipower variation, realised quarticity, trade-price volatility, the 1 s forward-filled
  grid and its 5 s and 30 s sub-sampled versions.
- **Pre-averaging estimators** (Li, Laeven and Vellekoop): applied to the forward-filled grid and to the observation
  index, in raw, i.i.d.-noise and bias-corrected forms.
- **Book and trade statistics:** spread, imbalance, depth (mean and standard deviation), update frequency, trade count,
  size, order count and their cumulative sub-window versions.
- **Same-`time_id` cross-stock aggregates** (V2 with cross-stock features and V4): for ten base columns the
  cross-stock mean, standard deviation, rank and deviation within the window, plus the number of stocks present and a
  per-stock level (mean log volatility over the stock's training windows).
- **Sequence tensor** (V3 only): 120 bins of five seconds x 13 channels (squared return, return, spread, imbalance, depth,
  update count, trade count, size, order count, trade return, two presence masks).

Missing seconds are treated as in the literature the features follow: a quote stays until new information, an absent
trade second is no trade.

## The four methodologies

| Variant | Model | Input | Why |
| --- | --- | --- | --- |
| V1 | Ridge on log target with stock effects | plain volatility family, or plus the pre-averaging estimators | tests whether noise-robust estimators add anything over plain realised volatility |
| V2 | LightGBM on log target | 75 per-stock features, or all 117 with the cross-stock aggregates | tabular reference; isolates the value of the shared market factor |
| V3 | 1-D CNN (3 conv layers, 265 k parameters) + MLP on the aggregated features, three seeds | sequence tensor + 117 features | tests whether second-level structure adds signal beyond aggregates |
| V4 | Transformer encoder over the set of stocks of one `time_id` (3 layers, 4 heads, 417 k parameters), three seeds | 117 features per stock, no positional encoding | learns cross-stock structure directly; permutation-equivariant by construction |

V3 and V4 are trained with the squared percentage error on `exp(output)`, AdamW, a cosine schedule over a fixed wall-clock
budget, bf16 autocast on the L4, and early stopping on an exponential moving average of the *choose* RMSPE.

## Results

Report role (62,491 windows), never used to choose anything:

| Model | Choose RMSPE | **Report RMSPE** |
| --- | --- | --- |
| V1 plain volatility features | 0.2461 | 0.2432 |
| V1 plus pre-averaging estimators | 0.2459 | 0.2431 |
| V2 LightGBM, per-stock features | 0.2391 | 0.2384 |
| V2 LightGBM, with cross-stock features | 0.2187 | 0.2219 |
| V3 CNN, 3-seed ensemble | 0.2128 | 0.2073 |
| V4 set transformer, 3-seed ensemble | 0.2119 | 0.2074 |
| **Blend** (V2 0.1, V3 0.4, V4 0.5; weights chosen on *choose*) | 0.2106 | **0.2057** |

Single seeds: V3 0.2091 to 0.2101, V4 0.2109 to 0.2111 (report). Source: [`results/metrics.csv`](results/metrics.csv).

What the numbers say:

- **Pre-averaging estimators add nothing** on these windows (0.2431 against 0.2432). The windows have a median of 388
  observations, far from the 23,400 to 468,000 observations in the estimator paper's own study.
- **The shared market factor is worth about 7 % of RMSPE** in LightGBM (0.2384 to 0.2219).
- **The neural models are about 6 % better again** (0.2219 to 0.2073). Seeds agree to within 0.001 to 0.002, so the
  ordering V3 versus V4 is not a result.
- **They overfit quickly.** The best *choose* epoch is 13 to 17 for V3 and 15 to 29 for V4 while the fit error keeps
  falling ([fig. 1](figures/fig1_gap_per_epoch.png)).
- **Predictions sit below the target on average** (mean log ratio -0.06 to -0.08, [fig. 4](figures/fig4_residuals_calibration.png)). This is
  the RMSPE-optimal shading, not a bug: the scalar above is fitted for it.

## Experiment notes

| # | What | Outcome |
| --- | --- | --- |
| 1 | Plain volatility, one constant, all labelled rows | 0.2858 (reference) |
| 2 | V1 ridge, plain or plus estimators | 0.2432 / 0.2431: estimators add nothing |
| 3 | V2 without / with cross-stock features | 0.2384 / 0.2219 |
| 4 | V3, V4, three seeds each | 0.2073 / 0.2074 |
| 5 | Blend on *choose* | 0.2057 |

Defects found and fixed while building it, each guarded by a test:

- The first pre-averaging noise correction used the wrong constant (`2M/(k+1)` instead of `2Mk/(k+1)^2`). A simulation
  with known noise showed the estimate far above the truth; `tests/test_features.py` now checks that heavy i.i.d. noise
  is removed to within 25 %.
- The dependent-noise second step (`iv3`) is unstable on windows this short: a median ratio near 0 and about 40 %
  non-positive values in the simulation. The stable i.i.d. second step (`iv2`) is the one used as a feature.
- The cross-stock features are computed from inputs only; a test checks that the feature table is identical whatever the
  target column holds.

## Compute

| Stage | Host | Wall time | Notes |
| --- | --- | --- | --- |
| Data profile | 4-core CPU sandbox | 47 s | 112 stocks, one process each |
| Features + V1 + V2 | 8-core CPU sandbox | 205 s | 7 build workers, then two LightGBM processes with 4 + 3 threads |
| Features + V3 + V4 | one NVIDIA L4, 12 vCPUs | 439 s | 127 s feature build with 10 workers, then six training processes beside a sampler; GPU memory fraction 0.85 split 0.6 / 0.4 between V3 and V4 |

The L4 stage logged 62 utilisation samples every 5 s: mean 96.8 %, median 100 %, 97 % of samples at or above 80 %, peak
memory 9.5 GB of 23 GB ([`results/gpu/logs/gpu_utilisation.json`](results/gpu/logs/gpu_utilisation.json)). CPU utilisation of the CPU
stage was not sampled. Total training stayed well inside a 45-minute cap.

## Kaggle scoring

The competition has closed. The blend was submitted late, as a code-competition notebook that loads the frozen weights and
predicts the hidden test windows ([`notebooks/inference.py`](notebooks/inference.py)). Before submitting, the notebook
re-ran the whole pipeline on the training files mounted in the Kaggle kernel and reproduced the numbers above (report 0.2057,
choose 0.2106), so the weights, features and blend arrived intact.

| Column | Value | Source |
| --- | --- | --- |
| Local report RMSPE (this repository) | 0.2057 | `results/metrics.csv` |
| Kaggle public score shown by the scorer | **0.00000** | `results/kaggle_scores.csv` |
| Kaggle private score shown by the scorer | 0.24597 | `results/kaggle_scores.csv` |
| Best benchmark quoted by the repository owner | 0.19548 | not measured here |

**Read this honestly.** The scorer displays a public score of exactly 0.0, a "perfect score", which would beat the best
benchmark of 0.19548. It is not a credible result: an RMSPE of exactly zero requires every prediction to equal its target,
which a forecast of a future volatility cannot do, and the same listing shows a private score of 0.24597 for the same file.
We treat the 0.0 as a scorer artefact whose cause we have not established (the public test file is a three-row stub), and we
make no performance claim from it. The private score of 0.24597 is the number that measures the forecasting period; it is
worse than the local 0.2057 and worse than the 0.19548 benchmark.

Candidate reasons for the gap between 0.2057 and 0.24597, none of them tested: the hidden windows come from a later market
period than the training windows, while the local roles are a random split of shuffled `time_id`s; the hidden test windows
contain only some of the stocks, so the cross-stock features see different inputs than in training; the per-stock level
feature was computed over the training period. The training pipeline was not re-run after the submission.

## Reproducing

```bash
cd optiver-realized-volatility
pip install -r requirements.txt
python -m pytest tests -q                                   # features, estimators, split, network probes
export OPT_DATA_DIR=/path/to/optiver-realized-volatility-prediction   # Kaggle layout: train.csv, book_train.parquet/stock_id=*/...
python scripts/investigate_phase1.py                        # builds the per-window profile table; scripts/analyze_phase1.py prints the data profile
python src/run_cpu.py --out runs/cpu --workers 7            # features + V1 + V2   (about 4 minutes on 8 cores)
python src/run_gpu.py --out runs/gpu                        # features + V3 + V4   (about 7 minutes on one L4)
python scripts/aggregate_predictions.py --results runs --out results
python scripts/make_figures.py
```

Requirements: Python 3.12, NumPy, pandas, PyArrow, LightGBM 4, PyTorch 2 with CUDA for V3 and V4 (an L4 or any 16 GB card;
the six processes use 9.5 GB). Alternatively set `OPT_HF_REPO` and `HF_TOKEN` to read the same layout from a Hugging Face
dataset repository. The competition data is not redistributed here.

## Limitations

- One split, one origin, no confidence intervals: differences below about 0.002 between V3 and V4 are not a ranking.
- Random split of shuffled windows: the local numbers cannot measure the shift to the forecasting period, and the private
  score shows that shift is large.
- The cross-stock models assume most stocks of a window are present at prediction time.
- V3 and V4 stop early because they overfit; more regularisation, augmentation or longer-context inputs were not tried.
- The tick-index pre-averaging features use windows with at least 17 observations; shorter windows get missing values.
- The training-time per-stock level feature uses the inputs of all training windows, including the report role (no target
  is involved).
