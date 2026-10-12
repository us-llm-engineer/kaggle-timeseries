# Enefit prosumers: day-ahead solar and consumption forecasting, method, experiments and how to reproduce

A summary with all figures is in the [main README](../README.md). This file has the details: the data, the leak-free
feature rules, each of the four models, the experiment table, the compute, and the exact commands.

## Folder map

| Path | Contents |
| --- | --- |
| `src/enefit13/data.py` | Data layer: reads the competition tables, builds county-level day-ahead weather, capacity scales, calendar features, target-history lags (all at least 48 h old). Fixed roles: fit, choose, report. |
| `src/enefit13/gbdt.py` | V1 and V2: LightGBM models, early stopping on the *choose* role only, atomic history files. Also the same-season tuning variant (`v1m`, `v2m`, implemented, not run). |
| `src/enefit13/nn.py`, `nn_ens.py` | V3 and V4: TiDE-style day-ahead network. `nn_ens.py` trains 8 independent members at once with batched matrix products. |
| `src/enefit13/serve.py` | Final fit on all labelled data and incremental serving through the competition's time-series API. |
| `src/run_variant.py`, `src/run_dl_ens.py` | Back-test runners (V1 to V4). `src/prep_dl.py` builds the dense tensors once. `src/gpu_sampler.py` records GPU utilisation. |
| `src/aggregate_predictions.py` | Turns per-row predictions into the small CSVs in `results/post/` (no competition rows are published). |
| `src/mock_api_check.py` | End-to-end check of the serving path against a mock API. |
| `scripts/data_profile*.py` | Data profile: sizes, scale shift, baselines. `scripts/make_figures.py` draws every figure. `scripts/build_kaggle_script.py` builds the single-file notebook. |
| `notebooks/enefit_blend.py` | The single-file Kaggle script (fit on all data, then serve the API). |
| `tests/test_leakage.py` | Leakage mutants and an input-influence probe. |
| `results/data/` | Data profile. `results/backtest/` LightGBM metrics and learning curves. `results/dl/` DL metrics, per-epoch histories, GPU samples. `results/post/` aggregate error tables. |
| `figures/` | The 9 figures shown in the main README. |

## The task and the data

Predict the hourly electricity **production** and **consumption** of 69 prosumer segments (county x business flag x
contract type), one forecast per morning for the next day. Score: plain mean absolute error over all rows
(`results/data/data_profile.txt`).

- 2,018,352 training rows, 2021-09-01 to 2023-05-31, one row per segment, target type and hour.
- Side tables: installed solar capacity and number of metering points per segment and day, forecast weather at 112
  grid points (3.4 M rows), day-ahead electricity and gas prices.
- Consumption is 84 % of the absolute target mass, production 16 %. Ten segments carry 64 % of the mass.
- Production is zero at night (34.8 % exact zeros) and follows forecast radiation (correlation 0.82 after dividing by
  capacity).
- **Scale shift:** installed capacity grew 110 % over the window (70.7 k to 148.2 k kW). May 2023 production is about
  1.85 times the peak month of 2022. A tree model on raw targets cannot predict above the largest value it saw.

## Methods

The leak-free feature rules, the evaluation protocol and the four models are described in [`docs/methodologies.md`](docs/methodologies.md).

## Experiments

Report window, MAE in kWh (lower is better). Sources: `results/backtest/*/metrics.json`, `results/dl/*/metrics.json`.

| # | Experiment | Report MAE | Production | Consumption | Outcome |
| --- | --- | --- | --- | --- | --- |
| E1 | Zero / unit x hour median | 347.8 / 207.2 | | | reference |
| E2 | Same hour 48 h ago | 123.8 | | | reference (newest legal value) |
| E3 | Same hour 168 h ago | 105.3 | | | strongest simple baseline |
| E4 | V1 LightGBM L1, capacity-relative, lags | 63.6 | 70.3 | 56.9 | 40 % below E3 |
| E5 | V2 LightGBM Tweedie, no target history | 99.1 | 66.9 | 131.4 | production fine, consumption needs history |
| E6 | First V3/V4 on a local GPU | 92.8 (V3) | 110.2 | 75.4 | **invalid: a LayerNorm over one output feature is a constant, so the network ignored the weather**; found when two dropout values gave identical results; fixed and covered by a test |
| E7 | V3 fixed, T4, fp32, 8 members | 68.8 | 66.3 | 71.4 | the time-budget estimator cut it to 11 epochs |
| E8 | V4, T4, fp32, 8 members | 61.0 | 67.0 | 55.0 | 41 epochs, plateau stops |
| E9 | V3, T4, fp16, 46 epochs | 61.3 | 62.1 | 60.5 | best production, stops at the epoch cap |
| E10 | V4, T4, fp16, 43 of 47 epochs | 60.8 | 67.1 | 54.5 | best consumption; all members plateau-stop |
| E11 | Mean of V1, V3, V4 | **56.6** | 60.3 | 52.9 | best; weights chosen on the choose role gave the same 56.6 |
| E12 | Same-season tuned LightGBM (`v1m`, `v2m`) | not run | | | implemented, not evaluated |

Findings from the saved statistics:

- **Bias, not variance, dominates production.** At 13:00, V1's mean error is -133 kWh, V3's -5 and V4's -20
  ([`figures/post_error_by_hour.png`](figures/post_error_by_hour.png)). Capacity-relative targets fix the scale
  shift for the network but V1's trees still under-predict the rising summer level.
- **The models make similar errors.** The error correlation between V1, V3 and V4 is 0.68 to 0.85
  (`results/post/error_correlation.csv`), so averaging helps by 4 kWh (60.8 to 56.6), not more.
- **The large segments set the metric.** Error grows almost linearly with segment size
  ([`figures/post_per_unit.png`](figures/post_per_unit.png)).
- **GPU use.** On one T4 the fp32 run held 83 % mean utilisation (83 % of samples at or above 80 %), the fp16 run 88 %
  (87 %), with peak memory 2.4 GB of 15.4 GB (`results/dl/*/gpu_utilisation.json`). The cloud host has a single CPU core,
  so separate worker processes were not used; one process with member-batched matrix products was.

## Submission attempt and what is not claimed

A notebook that fits all models on the full training data and serves the competition's time-series API day by day was
built (`notebooks/enefit_blend.py`). Its serving path passed a mock-API check in the cloud (4 days, 12,480 rows, no NaN,
no feature mismatch against the training pipeline, `results/dl/serve_check.json`).

**It could not be scored on Kaggle.** The competition's API package is a compiled module for CPython 3.10
(`competition.cpython-310-x86_64-linux-gnu.so`), and the notebook runner now uses Python 3.13; the environment menu
offers only the original or the latest image, and the competition ended on 2024-04-30. So the figures above are
**back-test numbers on the last 60 days of the public training data, not leaderboard scores.** The first month of the
report window is also the season for which only the previous year was seen at a smaller capacity.

## Reproducing

```bash
pip install -r requirements.txt                   # Python 3.12, LightGBM 4, PyTorch 2 with CUDA for V3/V4
# competition CSVs: kaggle competitions download -c predict-energy-behavior-of-prosumers (rules must be accepted)
python scripts/data_profile.py                    # sizes, scale shift, baselines -> results/data/data_profile.txt
python -m pytest tests -q                         # leakage mutants and input-influence probe
python src/run_variant.py --variant v1 --root data --out results/backtest      # V1; v2 likewise
python src/prep_dl.py --root data --out prep                                    # dense tensors once
python src/run_dl_ens.py --prep prep --out results/dl --variants v3,v4 --drops 0.5,0.2 --seeds 4 --epochs 100 --budget-min 22
python src/aggregate_predictions.py --v1 ... --v3 ... --v4 ... --out results/post
python scripts/make_figures.py                    # figures from results/
```

`--root` expects the competition tables as parquet (see `src/prepare_hf_dataset.py`, which converts the CSVs and shards
the weather tables by month) or, for the serving code, the original CSV folder. The competition data is not
redistributed here.

## Limitations

- Only a 60-day back-test, one origin, no confidence intervals; differences below about 1 kWh between V3 and V4 should
  not be read as a ranking.
- The report window's first month overlaps the capacity step-up; a different origin could change the ordering of V1
  against the networks.
- Production-model rounds (61) were fixed by a low-sun choose month; the same-season tuning was not evaluated.
- V3 stopped at the epoch cap while still improving; 100 epochs were not reached for it within the time budget.
- No leaderboard score exists for the reasons above.
