# Methodologies

Every method of the ROGII project, described the same way: input, model, output, and why it succeeded or failed. Results and the experiment table are in the [main README](../../README.md).

Each method below is described the same way: input, model, output, and why it succeeded or failed. The notation follows
the key mathematical points in the [main README](../../README.md#key-mathematical-points).

**Carry-forward (E1).** Predict $\hat y_t = \mathrm{TVT}_p$, the last known value, for every row after the cut. It is
the reference for every other method.

**Recent-trend extrapolation (E2) and geometric extrapolation (E6).** Fit a line to TVT, or to $\mathrm{TVT} + Z$,
against MD or against X, Y, Z over the last 100, 500 or 1,500 known rows, using ridge-damped least squares, and extend
it to the end of the well. A further variant assumes horizontal layers ($\Delta\mathrm{TVT} = -\Delta Z$). There
is no GR evidence. One-sided slope errors compound over 4,000-plus rows, giving 57-1,066 ft depending on the variant.

**Row-level LightGBM (E3).** For each row after the cut, predict $\mathrm{TVT}_t - \mathrm{TVT}_p$ from row features:
distance from the cut, changes in Z and position, GR and prefix statistics. It uses 800 trees and 63 leaves, with
5-fold CV on a multi-core CPU. The model has no notion of the type well's depth axis, so it cannot locate the layer.

**GRU-D seq2seq and companion SSM (E4, E5).**
- Setup: on one T4, 256 known rows serve as context and the model forecasts the next 256-row offsets, rolled out over
  the full suffix.
- GRU-D: a GRU encoder over the context, and a GRU-cell decoder trained with decaying teacher forcing. In the style of
  GRU-D, the inputs carry a GR-missing mask and the distance since the last observed GR, alongside MD, X, Y, Z and GR.
- SSM: a linear state-space model in companion form whose spectral radius is monitored.
- Both train for 8 epochs.
- Why they fall short: they forecast from history without looking up the type well, so on long suffixes they approach
  carry-forward (E4) or drift (E5).

**Grid tracker V1/V2 (E7, E8).**
- State: the offset $d$ (0.25 ft grid, ±160 ft) and the slope $v$ per block.
- Prior: the AR(1) slope model.
- Emission: the heavy-tailed GR match against the calibrated type well. V2 adds a multi-scale window average and a
  spliced reference.
- Inference: exact forward-backward; the prediction is the posterior mean.
- Why it fails: exact inference treats correlated samples as independent, so it is overconfident. Its 1σ coverage was
  only 0.40.

**Learned emission (E9).** A LightGBM classifier answers "is cell $d$ the true depth for block $k$?" Its negatives
include hard decoys: wrong cells whose GR matches best. Its logit would replace the hand-made emission. It reached
AUC 0.896 on the choose fold in the smoke test, but was dropped in favour of the coarsening and deep-learning lines.

**Coarsened tracker (E10).** This is V1 with the emission raised to the power $\zeta = \alpha/(\alpha+n)$ (Miller and
Dunson):
- $n$ is the number of informative blocks;
- $\alpha = 2\sigma^2/\delta^2$ comes from the prefix residual scale $\sigma$ and the typical calibration drift
  $\delta$, both measured on the training wells.

Tempering stops a single wrong layer from dominating; it scored 14.46 ft.

**Optimal-transport tracker (E11).** The emission is the $W_2$ distance between an observed 17-block GR window and the
window predicted by $(d, v)$, after a softplus mapping (Engquist and Yang). $W_2$ is smooth in shift, but it is too
permissive between similar-looking layers and lets the track wander (46 ft).

**Multi-hypothesis network (E12).** On 64-ft windows, a network sees a heat map of type-well-minus-GR mismatches and
outputs $M$ candidate paths with probabilities. It is trained with a winner-takes-all L1 loss plus cross-entropy
(Alyaev and Elsheikh). Windows are chained along the suffix with a beam of 8. Errors compound across hundreds of chained
windows, and the beam drifts away from the true path.

**1-D U-Nets with a cost volume (E14).** Residual 1-D U-Nets of widths 16, 32 and 64 run on 10-ft blocks. At each of 6
iterations, a local cost volume is rebuilt around the current offset estimate and the estimate is refined. This
follows the residual 1-D U-Net with augmentation and the iterative local cost volume of the two papers in the
references. It scored 15.71 ft on fold 1, a small gain.

**Block-level LightGBM base (E15).** Per 10-ft block, the target is the offset. The features are absolute-frame
geometry and GR residuals, plus the coarsened tracker's posterior mean and s.d. It was built as a level-0 model for a
stack but was superseded by the 2-D U-Net. It is reported only at block level, so it is not comparable.

**Submitted methodology: grid U-Net snapshot blend (A, E18) + LightGBM margin (B, E19).**

1. **Input image per well.** Rows are 256 candidate offsets (±96 ft, 0.75 ft apart). Columns are 16-row blocks: 16
   before the cut and up to 640 after. The 21 channels include:
   - GR, the type-well GR at each candidate depth, and their difference;
   - the heavy-tailed match score and a smoothed difference;
   - a line marking the known history and a line marking the constant-dip prior;
   - masks, geometry and coordinates.
2. **Network.** A 4-level residual 2-D U-Net at widths 16, 24 and 32 (0.5 M to 2.1 M parameters). It outputs one logit
   per candidate × block.
3. **Decoding.** A softmax over candidates with a learned temperature; the prediction is the expected offset. Block
   values are linearly interpolated to rows.
4. **Training.**
   - Loss: soft-target cross-entropy plus L1 on the offset and on the slope.
   - Cuts: random, so one well yields many training cases.
   - Synthetic suffixes: half the batches use paths drawn from the AR(1) prior with re-rendered GR.
   - Schedule: 12 epochs, AdamW with one-cycle, bf16, on one L4.
5. **Snapshot blend A.** Average the predictions from epochs 7-12 of all three widths: 18 checkpoints for the final
   model.
6. **Margin B.** LightGBM with Huber loss predicts $y - A$ from 20 target-free features: geometry, the dip prior, $A$
   and its slope, and GR residuals at $A$ and $A \pm 2$ ft. It is trained on out-of-fold $A$, so that it corrects the
   blend's held-out errors, and the final version uses all 770 wells. Submission: $B = A + f(x)$.
7. **Evaluated but not submitted.** A per-well A/B gate (E20) and a width-difference scale direction (E21). Neither
   beat B on folds 2-5.
8. **Inference.** A Kaggle notebook with GPU on and internet off loads the 18 checkpoints and the margin model and
   writes `submission.csv`.
