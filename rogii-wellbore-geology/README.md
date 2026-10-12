# ROGII wellbore geology: method, experiments and how to reproduce

A summary with all figures and the key equations is in the [main README](../README.md). This file covers the details:
the data split, every model that was tried, the per-fold tables, the design decisions and the exact commands.

## Folder map

| Path | Contents |
| --- | --- |
| `src/prepare_data.py` | Converts the competition CSVs into well-per-row Parquet shards (`data/stage`). |
| `src/wells.py` | Loads wells and type wells, decides which wells are eligible, defines the fixed 5-fold split, computes suffix metrics. |
| `src/tracker.py` | Baseline: an exact grid tracker over (offset, slope) with forward-backward and the AR(1) slope prior. Variants V1 and V2. |
| `src/coarsened_tracker.py` | Baseline: the same tracker with a coarsened (power-tempered) likelihood. |
| `src/grid_unet.py` | Main model: the 2-D candidate-offset × block U-Net. Covers training, per-epoch held-out snapshots and full-data checkpoints. |
| `src/stack.py` | Row-level stack over the U-Net snapshots: blend A, margin B, gate and scale direction, all out-of-fold. |
| `src/final_margin.py` | Trains the submitted margin model on all wells, using out-of-fold A. |
| `notebooks/inference.ipynb` | Kaggle inference notebook (`MODE = "A"` or `"B"`). |
| `tests/test_leakage.py` | Leakage mutants: changing the true TVT after the cut must not change a prediction; changing GR after the cut must. |
| `scripts/` | `fetch_data.sh`, `export_horizon.py`, `make_figures.py`. |
| `results/training/` | One JSON line per epoch for all 15 fold runs and the 3 full-data runs. |
| `results/stack/` | Stack log per fold and pooled, per-well errors, RMSE by distance from the cut. |
| `results/baselines/` | Summary logs of the baselines and of the failed variants. |
| `results/history/` | Logs from earlier rounds and `experiments.csv`, the table of all 21 experiments (E1-E21). |
| `results/leaderboard.csv`, `results/leaderboard.png` | Kaggle scores of the two submissions. |
| `figures/` | The 12 figures shown in the main README (5 during training, 6 after training, 1 for the experiment history). |
| `REFERENCES.md` | Every paper, its role (used, tried or read), and BibTeX entries. |

## Data and evaluation protocol

- The training split has 773 horizontal wells, each with its own type well. Three training ids also appear in the
  public test set, so they are excluded. Every well with a valid known prefix and a complete target is eligible:
  **770 wells and 3,769,838 scored rows**.
- **Folds:** `KFold(5, shuffle=True, random_state=20261011)` over the sorted eligible ids. The roles were fixed before
  any model was trained:
  - fold 1 = *choose* (design decisions are made here);
  - folds 2-5 = *report* (unbiased numbers);
  - all five folds out-of-fold = training data for the stack.
- **Metric:** pooled row RMSE in feet over all rows after the cut. Carry-forward (repeat the last known TVT) is the
  reference: 15.92 ft on all folds and 15.71 ft on folds 2-5.
- **Leakage control:** at prediction time a model reads only the known prefix of the well, the trajectory, GR and the
  type well. `tests/test_leakage.py` adds noise of 50 ft to the true TVT after the cut and checks that the prediction
  is bit-identical.

## Results

### All methods (pooled out-of-fold RMSE, ft)

| Method | Code | Role measured | RMSE | Carry on the same rows | Source |
| --- | --- | --- | --- | --- | --- |
| Carry-forward | (in `stack.py`) | folds 2-5 | 15.71 | 15.71 | `results/stack/stack_log.jsonl` |
| Grid tracker V1 (heavy-tailed emission) | `tracker.py --variant v1` | folds 2-5 | 16.92 | 15.71 | `results/baselines/tracker_v1_report.jsonl` |
| Grid tracker V2 (spliced reference, multi-scale) | `tracker.py --variant v2` | folds 2-5 | 17.38 | 15.71 | `results/baselines/tracker_v2_report.jsonl` |
| Coarsened tracker | `coarsened_tracker.py` | folds 2-5 | 14.46 | 15.71 | `results/baselines/coarsened_tracker_report.jsonl` |
| OT (W2) window misfit | not shipped (failed) | fold 1 | 46.33 | 16.75 | `results/baselines/ot_misfit_choose.jsonl` |
| Multi-hypothesis net, beam-chained | not shipped (failed) | fold 1 | 21.11 | 16.75 | `results/baselines/mtp_net_choose.jsonl` |
| 1-D residual U-Net, 3 widths | not shipped (superseded) | fold 1 | 15.71 | 16.75 | `results/baselines/unet1d_choose.jsonl` |
| **A: grid U-Net snapshot blend** | `grid_unet.py` + `stack.py` | folds 2-5 | **13.27** | 15.71 | `results/stack/stack_log.jsonl` |
| **B: A + LightGBM margin** (submitted) | `stack.py`, `final_margin.py` | folds 2-5 | **13.07** | 15.71 | `results/stack/stack_log.jsonl` |
| Gated A/B | `stack.py` | folds 2-5 | 13.18 | 15.71 | `results/stack/stack_log.jsonl` |
| Gated + scale direction | `stack.py` | folds 2-5 | 13.17 | 15.71 | `results/stack/stack_log.jsonl` |

The "not shipped" variants are reported so that the negative results are on record. The complete history, including
round 1-2 attempts, is in the main README and `results/history/experiments.csv`. They are not needed to reproduce
the submission.

### Stack stages per fold (`results/stack/stack_log.jsonl`)

| Fold | Role | Carry | A | B | Gated | Gated + scale | Share of wells the gate sent to B | Scale coefficient $c$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | choose | 16.75 | 14.16 | 14.16 | 14.35 | 14.32 | 0.47 | 0.148 |
| 2 | report | 14.65 | 12.89 | 12.59 | 12.88 | 12.81 | 0.51 | 0.125 |
| 3 | report | 14.71 | 13.02 | 12.47 | 12.58 | 12.54 | 0.59 | 0.148 |
| 4 | report | 14.52 | 12.78 | 12.43 | 12.41 | 12.43 | 0.48 | 0.216 |
| 5 | report | 18.56 | 14.32 | 14.65 | 14.70 | 14.77 | 0.45 | 0.203 |
| **2-5 pooled** | | 15.71 | 13.27 | **13.07** | 13.18 | 13.17 | | |
| **1-5 pooled** | | 15.92 | 13.45 | **13.30** | 13.42 | 13.41 | | |

The gate and the scale direction did not beat B on the report folds, so B was submitted. On the leaderboard, B scored
11.622 public and 11.980 private, and A scored 12.008 public and 12.180 private (`results/leaderboard.csv`).

### Other diagnostics

- **Per well:** the stack beats carry-forward on 506 of 770 wells. Median per-well RMSE is 8.77 ft against 10.67 ft
  for carry-forward (`results/stack/final_per_well.csv`).
- **By distance from the cut:** the error grows from 2.4 ft in the first 250 rows to about 17 ft beyond 4,000 rows. The
  gain over carry-forward peaks at about 4.4 ft between 500 and 1,500 rows (`results/stack/horizon_rmse.csv`).

## Main model in detail (`src/grid_unet.py`)

**Image.** Each well and cut becomes a $21 \times 256 \times 656$ tensor.

- Rows are candidate offsets from the last known TVT: $\pm 96$ ft in steps of 0.75 ft.
- Columns are blocks of 16 rows: 16 history blocks before the cut and up to 640 after it.
- Channels:
  1. GR, normalised by the type-well reference;
  2. calibrated type-well GR at each candidate depth;
  3. their difference;
  4. the heavy-tailed score $-\log(1+|r|)$;
  5. a 5-block smoothed difference;
  6. a Gaussian line on the known history (σ = 1 ft);
  7. a Gaussian line on the constant-dip prior (σ = 4 ft);
  8. history mask;
  9. position mask;
  10. GR mask;
  11. type-well range mask;
  12. 8 geometry rows: $\Delta Z$, inclination, azimuth (sin, cos), $\Delta$MD, prior offset, $\Delta X$, $\Delta Y$;
  13. 2 coordinate maps.

**Network.** A 4-level U-Net with residual blocks: two 3×3 convolutions, GroupNorm, GELU and Dropout2d 0.1. The
channel widths are $w, 2w, 4w, 8w$ with $w \in \{16, 24, 32\}$, giving 0.52 M, 1.16 M and 2.06 M parameters. The
softmax temperature $\tau = e^{\theta}$ is learned and clamped to $[0.25, 4]$.

**Decoding and loss.** See points 4-5 of the main README. Decoding takes the expected offset under the softmax. The
loss is soft-target cross-entropy plus 0.1 L1 on the offset plus 0.1 L1 on the slope.

**Training recipe.**

| Setting | Value |
| --- | --- |
| Optimiser | AdamW, weight decay 1e-4 |
| Learning rate | one-cycle, peak 1e-3, 10 % warm-up |
| Gradient clipping | norm 1.0 |
| Batch | 8 wells |
| Steps per epoch | 2 × (number of fit wells) / 8 |
| Epochs | 12 |
| Precision | bf16 autocast |
| Cut | the real cut with probability 0.5, otherwise uniform between half the prefix and 64 rows before the end |
| Synthetic suffix | probability 0.5 |
| Fit wells per CV fold | a random 50 % of the four training folds, to fit the time budget |
| Wall-clock guard | 1,500 s per run |

**Snapshots.** After every epoch, each fold run predicts its held-out fold from the real cut and saves the block
offsets (`*_snapshots.npz`). The stack averages the epochs in the second half of training (7-12). The full-data runs
(`--fold 0`) save checkpoints for epochs 7-12, and the inference notebook averages those 18 checkpoints.

**What the in-training logs show.**

- Held-out RMSE flattens after about epoch 7 in every fold, which led to the snapshot averaging.
- The predicted posterior spread falls from 20-44 ft after epoch 1 to 11-12 ft by epoch 12.
- Wider nets have smaller gradient norms and slightly better late-epoch held-out RMSE.

The plots are in the main README.

## Stack in detail (`src/stack.py`)

1. **A** is the snapshot blend over widths 16, 24 and 32, epochs 7-12, interpolated to rows.
2. **Margin.** A LightGBM model with Huber loss ($\alpha = 5$) uses 63 leaves, 300 minimum rows per leaf, learning
   rate 0.03, 500 rounds, and every 4th row. Its target is $y - A$. Its 20 features are target-free:
   - X, Y, Z and their changes since the cut;
   - $\Delta$MD;
   - the anchor TVT;
   - the dip of the last 300 rows, and the prior residual;
   - $A$'s offset and its gradient;
   - the GR residual against the type well at $A$, smoothed, and at $A \pm 2$ ft;
   - the missing-GR flag;
   - the calibration scale and gain;
   - the row index after the cut.

   The margin for the held-out fold is fitted on the other four folds.
3. **Gate.** A small LightGBM classifier on 13 target-free per-well aggregates learns whether B beats A. Its labels
   come from an inner 4-fold cross-fit, so they reflect B's held-out behaviour.
4. **Scale direction.** $(A_{32} - A_{16})$ is added with one least-squares coefficient, fitted on the training folds.
5. **Submitted margin** (`final_margin.py`). The same LightGBM recipe, trained on all 770 wells with their
   **out-of-fold** A. That way, the margin learns how the blend errs on wells it has not seen, which matches the
   setting at test time.

## Reproduce

```bash
pip install -r requirements.txt
bash scripts/fetch_data.sh                                   # -> data/extracted/{train,test,sample_submission.csv}
python src/prepare_data.py --source data/extracted --out data/stage
python -m pytest tests -q                                    # needs data/stage

# baselines (CPU, multi-process)
python src/tracker.py --variant v1 --mode choose             # grid over B and mult on fold 1
python src/tracker.py --variant v1 --mode report             # folds 2-5 with the chosen setting
python src/coarsened_tracker.py --mode report

# main model: 5 CV folds x 3 widths (GPU), then 3 full-data runs that save the snapshot checkpoints
for f in 1 2 3 4 5; do python src/grid_unet.py --fold $f --runs 16:0,24:0,32:0 --out runs/grid; done
python src/grid_unet.py --fold 0 --runs 16:0,24:0,32:0 --out runs/full

# stack (out-of-fold metrics) and the submitted margin
python src/stack.py --grid runs/grid --out runs/stack
python src/final_margin.py --grid runs/grid --out runs/final
python scripts/export_horizon.py runs/stack/final_rows.npz results/stack/horizon_rmse.csv

# figures from results/ (no data or GPU needed)
python scripts/make_figures.py
```

For Kaggle, upload `src/*.py`, `runs/full/full_w*_s0_ep*.pt` and `runs/final/margin_lgbm.txt` as one dataset. Then
attach it, together with the competition, to `notebooks/inference.ipynb` (GPU on, internet off).

**Hardware used:** one NVIDIA L4 (24 GB), a T4-class card, for all U-Net training. Total training time was about
1.7 GPU-hours (sum of `epoch_s` in `results/training/`). Stacking and the trackers ran on a multi-core CPU. One fold run of three widths spends about 14 minutes in training epochs on the L4, plus the held-out predictions after each epoch.

## Caveats

- **Low compute.** Each width has one seed and a 12-epoch schedule, and the CV runs use half of the wells. Stronger
  public solutions use many seeds, synthetic-well pre-training and much longer training. That is the main reason for
  the remaining gap of about 6 ft.
- **Fold 1 was used for choices.** Fold 1 numbers are therefore slightly optimistic. Quote folds 2-5.
- **Fold 5 is hard.** Its carry-forward error is 18.56 ft, and it is the only report fold where the margin hurts.
- **Scored after the deadline.** Both leaderboard scores were obtained after the competition closed.
