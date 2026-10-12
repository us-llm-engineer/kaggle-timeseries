# Methodologies

Details for the Enefit prosumer project. Results and experiments are in the [project README](../README.md).

## What the forecast may know (leak-free alignment)

The forecast for day D is made at 11:00 on day D-1. The data feed delivers, for that day: the weather forecast issued
on D-1 (hours 22 to 45 ahead), the actual targets up to D-2 (a two-day delay), the client table up to D-2 and the
electricity price of D-1. Every feature follows exactly this:

| Feature group | Rule |
| --- | --- |
| Target history | lags 48, 72, 96, 120, 168, 336 h; same-hour mean/std/max over days D-2..D-8; daily level of D-2 and of the last 7 days |
| Scale | installed capacity (production) or metering-point count (consumption) from the client row of D-2 |
| Weather | forecast origin D-1 02:00, mean over the grid points of the county (national mean for counties without a station) |
| Price | day-ahead price of the previous day, same hour |
| Calendar | hour, weekday, month, day of year (sin/cos) |

`tests/test_leakage.py` mutates everything newer than the reveal lag and checks that no feature or look-back changes,
and that a mutant that does leak is detected. It also checks that the network output changes when one weather channel
changes.

## Evaluation protocol

Fixed before any model was trained, on a rolling-origin split of the last months:

- **fit:** before 2023-03-05; **choose:** 2023-03-05 to 2023-04-01 (early stopping, epoch choice only);
  **report:** 2023-04-02 to 2023-05-31 (60 days, 190,224 rows, never used to select anything).
- Baselines on the report window: predict zero 347.8, unit x hour median 207.2, same hour 48 h ago (the newest value
  the feed allows) **123.8**, same hour 168 h ago **105.3** (it also matches the weekday, which matters for
  consumption).

## Models

**V1, LightGBM with an L1 objective on capacity-relative targets.** Target `y / scale`, sample weight `scale`, so the
weighted L1 equals the raw MAE. One global model per target type, with the target-history features above, weather,
calendar, price and the segment ids. Production stops after 61 rounds, consumption after 538, early-stopped on the
choose role (`results/backtest/v1/`). The production model stops early because the choose month (March) has little
sun; the same-season tuning variant `v1m` is implemented but not run.

**V2, LightGBM with a Tweedie objective and no target history.** Weather, calendar, capacity and a seasonal
per-segment hour profile fitted on the fit role only; one model per target type and business flag. Tweedie handles the
exact zeros of production. It has no autoregressive signal, so consumption is poor (131 kWh).

**V3 and V4, TiDE-style day-ahead network** ([`figures/arch_tide.png`](../figures/arch_tide.png)). One sample is a
segment-day: input is the 168 h look-back ending at D-2 23:00, the 24 hourly weather/calendar vectors of day D, and
static ids; output is 24 hourly values. A residual block projects the weather and calendar to 4 dimensions per hour; a
dense encoder (two residual blocks, width 1024) and decoder produce 8 values per hour; a per-hour temporal decoder
adds the covariates again; a global linear skip maps the look-back to the 24 outputs. Loss: L1 on raw kWh, so it is
the competition MAE. AdamW, cosine decay, batch 512, gradient clipping per member, fp16 autocast.

- **V3** normalises by capacity: the look-back is `y / scale` and the output is multiplied by the target hour's scale.
  This addresses the capacity growth directly.
- **V4** normalises by the window: look-back mean and standard deviation (plus 1 kWh) are removed and added back, and
  `log mean`, `log std` are extra inputs. It does not use capacity as a divisor.

Each variant is trained as a **batched ensemble** of 8 independent members (dropout 0.5 and 0.2, 4 seeds each) in one
process. Members have their own weights, data order and dropout mask, and are early-stopped separately on a 3-epoch
smoothed choose MAE: a **plateau** (no 0.2 % gain for 12 epochs, after 20) or **overfitting** (train loss still
falling while the smoothed choose MAE sits more than 5 % above its best for 5 epochs). The best snapshot of each
member is kept. The epoch budget is 100 with a wall-clock cap (46 to 47 epochs fit).

**Blend.** Plain mean of V1, V3 and V4 predictions.
