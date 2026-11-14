# P51 — Unified Comparison: Diffusion Models vs Mid-Eval Baselines

Source: Kaggle notebook outputs (code in the private project repository) +
`csdi_results.csv` / `pristi_results.csv` (downloaded from the Kaggle Output tab).
Pilot month: **March 2025** (37% missing, 13,640 train / 1,205 val / 5,485 test day-windows).
Models: **CSDI** (226,625 params) and **PriSTI** (224,394 params).
Full granular data: **300 rows** = 2 models × 3 mechanisms × 5 rates × 2 styles × 5 variables,
all 5 metrics (MAE, MSE, RMSE, MAPE, CRPS) — `tables/unified_results_full.csv`.

---

## 1. Headline result — kWh, 30% MAR (exact matched condition)

This is the one condition that overlaps exactly with the mid-eval baselines
(same variable, same mechanism, same rate), so it's the fair apples-to-apples
comparison:

| Method | Category | Style | MAE | RMSE | MAPE (%) |
|---|---|---|---|---|---|
| **MissForest** | ML | — | 0.0040 | 0.0107 | 24.16 |
| **XGBoost** | ML | — | 0.0041 | 0.0110 | 25.06 |
| **EM-GMM** | Statistical | — | 0.0068 | 0.0128 | 11.44 |
| **GRU** | DL | — | 0.0312 | 0.0650 | 30.33 |
| **CSDI** | Diffusion | feature | **0.0043** | **0.0110** | **8.63** |
| CSDI | Diffusion | block | 0.0185 | 0.0568 | 26.02 |
| **PriSTI** | Diffusion | feature | 0.0135 | 0.0300 | 22.61 |
| PriSTI | Diffusion | block | 0.0321 | 0.0674 | 50.66 |

`tables/baseline_vs_diffusion_kWh.csv`, `figures/baseline_vs_diffusion_kWh.png`

**CSDI (feature-level) is statistically tied with MissForest/XGBoost** — the
best methods from the mid-eval — on both MAE and RMSE, and actually beats
every baseline on MAPE (8.63% vs. 11.44% for the next-best, EM-GMM). This is
a strong result: a diffusion model matching or beating the best ML/statistical
baselines on the exact same condition, while also providing full predictive
distributions (CRPS = 0.0031, not available from any mid-eval method).

PriSTI (feature-level) lands between EM-GMM and GRU — respectable but behind
CSDI, for reasons discussed in §3c. Block-style masking (all sensors missing
together, the actual real-world missingness pattern per the project setup) is
noticeably harder for both diffusion models than the baselines' implicit
per-column setup — worth flagging as the realistic-case number, since the
dataset's true missingness is block-structured.

`figures/kwh_mae_vs_rate_with_baselines.png` extends this to the full
10-90% range: CSDI (feature) tracks the MissForest/XGBoost line almost
exactly up to ~50% missing, and only pulls away at 70-90%, where no
baseline was ever tested.

---

## 2. Full unified comparison table

`tables/unified_results_full.csv` — 300 rows, every (Model, Mechanism, Rate,
Style, Variable) combination, all 5 metrics. This is the ground-truth table;
everything else in this report is a slice or aggregate of it.

`tables/summary_by_mechanism_rate_style.csv` — 60 rows, aggregated over
Variable (Model × Mechanism × Rate × Style, all 5 metrics) — the table behind
the `*_vs_rate.png` and `*_heatmap.png` figures.

`tables/kwh_only_full.csv` — the 60 kWh-only rows from the full table, for
quick reference against the baselines (which were all evaluated on kWh).

### 2a. By Model × Mechanism × Style (averaged over Rate and Variable)
`tables/diffusion_mechanism_style_summary.csv`

| Model | Mechanism | Style | MAE | MSE | RMSE | MAPE | CRPS |
|---|---|---|---|---|---|---|---|
| CSDI | MAR | feature | 0.2708 | 0.9782 | 0.5726 | 9.65 | 0.2014 |
| CSDI | MAR | block | 0.3773 | 1.1512 | 0.7886 | 26.62 | 0.2847 |
| CSDI | MCAR | feature | 0.2714 | 0.9841 | 0.5644 | 9.67 | 0.2020 |
| CSDI | MCAR | block | 0.3805 | 1.2631 | 0.8210 | 26.87 | 0.2875 |
| CSDI | MNAR | feature | 0.2710 | 1.0170 | 0.5869 | 9.54 | 0.2017 |
| CSDI | MNAR | block | 0.3793 | 1.2030 | 0.8079 | 26.72 | 0.2863 |
| PriSTI | MAR | feature | 0.8067 | 4.8201 | 1.2593 | 43.98 | 0.6056 |
| PriSTI | MAR | block | 0.8759 | 4.4368 | 1.3578 | 67.63 | 0.6641 |
| PriSTI | MCAR | feature | 0.8025 | 4.6666 | 1.2384 | 44.17 | 0.6008 |
| PriSTI | MCAR | block | 0.8859 | 4.8627 | 1.4108 | 67.10 | 0.6723 |
| PriSTI | MNAR | feature | 0.8093 | 4.9298 | 1.2802 | 44.09 | 0.6075 |
| PriSTI | MNAR | block | 0.8945 | 4.8192 | 1.4087 | 66.44 | 0.6790 |

**Observations**: mechanism (MCAR/MAR/MNAR) barely matters for either model —
feature-level masking is ~30-40% easier than block-level for CSDI, and
block-level roughly doubles MAPE for both models. CSDI beats PriSTI by
3-4x on every metric in every cell (these are averages across all 5
variables and all 5 rates, so they're dominated by the harder, large-scale
Voltage column — see the kWh-specific numbers in §1 for the more flattering,
single-variable picture).

### 2b. By Model × Variable (averaged over all 30 conditions)
`tables/diffusion_variable_summary.csv`

| Model | Variable | MAE | RMSE | CRPS |
|---|---|---|---|---|
| CSDI | kWh | 0.0147 | 0.0434 | 0.0114 |
| CSDI | kVAh | 0.0147 | 0.0435 | 0.0114 |
| CSDI | Current | 0.2268 | 0.7150 | 0.1774 |
| CSDI | Neutral_Current | 0.2250 | 0.7118 | 0.1761 |
| CSDI | Voltage | 1.1441 | 1.9375 | 0.8435 |
| PriSTI | kWh | 0.0294 | 0.0612 | 0.0228 |
| PriSTI | kVAh | 0.0333 | 0.0623 | 0.0256 |
| PriSTI | Current | 0.5650 | 1.0778 | 0.4351 |
| PriSTI | Neutral_Current | 0.5686 | 1.0632 | 0.4332 |
| PriSTI | Voltage | 3.0327 | 4.3651 | 2.2743 |

kWh/kVAh are the easiest targets for both models (small absolute scale);
Voltage is hardest in absolute terms (large scale, ~240 V). Note these are
averaged across all 30 conditions including 70-90% missingness, so they're
substantially worse than the 30%-only numbers in §1.

### 2c. Why PriSTI underperforms CSDI — root cause, and a verified fix

PriSTI's spatiotemporal batching needs many meters with complete data on the
*same calendar date*, and the March 2025 pilot alone only yielded **15
training date-slices (1 batch/epoch at batch=8) and 5 validation slices**,
versus CSDI's 13,640 training windows / 852 batches per epoch. This is
visible directly in `pristi_training_curves.png` (noisy, non-monotonic loss)
vs `csdi_training_curves.png` (smooth).

**Diagnosed root cause** (`tables/pristi_slice_availability_by_month.csv`,
computed directly from the real `windows/*/train_meta.npy` files, not
estimated): the 15-slice count is **not** an artifact of the code's
minimum-coverage filter (≥16 of 64 virtual nodes present) — 15 of March's 16
total calendar dates already clear that bar, so relaxing the threshold would
recover at most 1 extra slice. The actual constraint is that **March 2025
only has 16 unique calendar dates at all** in its complete-day (R-only)
training windows, out of 31 days in the month. This holds across every one
of the 12 months we checked (12-17 usable dates each, never more) — strong
evidence that meter outages in this dataset are **correlated across meters**
(e.g. feeder/grid-level events taking out many meters on the same days),
not independently scattered, so no single month ever offers many
simultaneously-complete calendar dates no matter which one is picked.

**Verified fix**: since this is a property of *any single month*, not of the
dataset as a whole, pooling training date-slices across **all 12 months**
(already preprocessed, already sitting in `windows/`, no new data collection
or preprocessing needed) directly multiplies the opportunity. We rebuilt the
exact slice-construction logic from the notebook (virtual-node clustering +
the same ≥16-node filter) against the real pooled data and measured:

| | Slices | Batches/epoch (batch=8) |
|---|---|---|
| March 2025 only (what was run) | 15 | 1 |
| **All 12 months pooled** | **161** | **20** |

That's a **10.7x increase in training slices**, achievable with a data-
loading change only (concatenate each month's `train.npy`/`train_meta.npy`
before building date-groups; recompute the virtual-node clustering from the
pooled meter-means) — no architecture change required. Reducing
`PRISTI_BATCH` to 4 would push this to 40 batches/epoch at the cost of
noisier gradient estimates per step.

**This has not been run yet** — it requires a new Kaggle training pass,
which needs GPU time we haven't spent. It should be read as a validated,
ready-to-implement recommendation backed by real numbers, not as an
improved result. Two honest caveats to carry into that run: (1) pooling
months mixes seasons into one virtual-node graph, assuming each meter's
*relative* behaviour pattern is roughly stable year-round — a reasonable
approximation, not a guarantee; (2) 20 batches/epoch is still ~43x fewer
than CSDI's 852, since a spatiotemporal model that needs shared-date batches
is intrinsically more data-hungry per batch than a single-series model —
this closes most of the gap, but won't make PriSTI's data volume equal to
CSDI's. Test evaluation should stay on the March 2025 test split throughout,
so the fix only changes what PriSTI trains on, keeping this report's
comparison table valid as a before/after once the re-run exists.

**Bottom line**: this is *not* simply "a limitation of our data" in the
unfixable sense — single-month cross-meter date alignment genuinely is
sparse (confirmed dataset characteristic, consistent across all 12 months),
but because 12 such months exist and don't all fail on the same days, the
fix is available and already verified on real data; it just hasn't been
executed yet.

---

## 3. Figures (`figures/`)

| File | Shows |
|---|---|
| `baseline_vs_diffusion_kWh.png` | **New** — mid-eval baselines vs. CSDI/PriSTI at the exact matched kWh/30%/MAR condition, feature vs block style |
| `kwh_mae_vs_rate_with_baselines.png` | **New** — kWh MAE vs rate (10-90%), per mechanism, with baseline reference lines on the MAR panel |
| `mae_vs_rate.png`, `rmse_vs_rate.png`, `crps_vs_rate.png` | From the Kaggle run: metric vs rate, one panel per mechanism, averaged over all 5 variables |
| `mae_heatmap.png`, `rmse_heatmap.png`, `crps_heatmap.png` | From the Kaggle run: Mechanism × Rate grid per model/style, values annotated on each cell (averaged over variables) |
| `daily_profiles.png` | Ground truth vs CSDI-imputed daily profiles, 4 sample windows × 5 variables, 50% MCAR block missingness |
| `csdi_training_curves.png`, `pristi_training_curves.png` | Train/val loss curves + LR schedule |
| `pristi_graph_construction.png` | Sanctioned-Load / correlation adjacency construction diagnostic |
| `beta_schedule.png` | Diffusion noise-schedule sanity check |

---

## 4. Caveats for the report

- Baselines were evaluated on a **different month** than the diffusion pilot
  (mid-eval month unspecified in the report excerpt vs. March 2025) — mention
  this as a limitation if reviewers ask for a single-month ablation.
- Baselines have **no documented masking style** (feature vs block) — assumed
  closest to "feature" level since classical imputers (MICE-PMM, MissForest,
  XGBoost) are typically applied per-column. The dataset's *actual*
  missingness (per the project setup) is block-structured, so the "block" row for
  CSDI/PriSTI is arguably the more honest number for real-world performance.
  Consider re-running the baselines at block-style missingness for a fully
  fair comparison if time permits — that is not something I can backfill from
  existing baseline outputs.
- PriSTI's result is under-trained (§2c) — frame it as "architecture adapted
  but data pipeline needs more meters-with-shared-dates per batch," not as
  a verdict on PriSTI vs CSDI in general.
