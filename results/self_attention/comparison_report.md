# P51 — Self-Attention Imputation (SAITS, Transformer) vs Diffusion Models and Baselines

**Source**
- Kaggle notebook outputs, run on a Kaggle Tesla T4 on 17 Nov 2026 (code in the private project repository).
- Merged with the diffusion results in `../comparison_report.md` by an analysis script.

**Setup**
- **Pilot month:** March 2025, with 13,640 train / 1,205 val windows. The test set is the **same first 1,000 test windows** that CSDI and PriSTI were scored on.
- **Conditions:** the same 30 (3 mechanisms × 5 rates × 2 styles), with **byte-identical masks**. The notebook's mask-verification printout matches the CSDI Kaggle run exactly.

Full granular data: **900 rows** (6 models × 30 conditions × 5 variables) in `tables/unified_results_all_models.csv`.

| Model | Type | Parameters | Epochs (early-stopped) | Training time (T4) | Time to evaluate all 30 conditions |
|---|---|---|---|---|---|
| **SAITS** | Self-attention (Du et al., ESWA 2023) | 1,325,094 | 176 | 23.2 min | **7.3 s** |
| Transformer | Self-attention baseline | 1,320,453 | 52 | 6.6 min | 6.8 s |
| SAITS-paper-mask | Ablation: the paper's 20 % random masking | 1,325,094 | 120 | 15.9 min | 7.3 s |
| LinearInterp | Reference | — | — | — | 4.9 s |
| CSDI | Diffusion (from the previous phase) | 226,625 | 78 | ≈ 40 min | ≈ 5 h 02 min (20 samples × 50 steps) |
| PriSTI | Diffusion (from the previous phase) | 224,394 | 93 | < 1 min (15 date-slices) | ≈ 44 min |

CSDI and PriSTI timings come from the cell timestamps in `diffusion_imputation_kaggle_executed.ipynb`.

---

## 1. Headline: kWh, 30 % MAR (the condition matched with the mid-eval baselines)

| Method | Category | Style | MAE | RMSE | MAPE (%) |
|---|---|---|---|---|---|
| SAITS-paper-mask | Self-attention (ablation) | feature | **0.0038** | 0.0101 | **8.33** |
| MissForest | ML | unspecified | 0.0040 | 0.0107 | 24.16 |
| **SAITS** | Self-attention | feature | 0.0040 | **0.0092** | 8.81 |
| XGBoost | ML | unspecified | 0.0041 | 0.0110 | 25.06 |
| CSDI | Diffusion | feature | 0.0043 | 0.0110 | 8.63 |
| Transformer | Self-attention | feature | 0.0046 | 0.0102 | 9.94 |
| EM-GMM | Statistical | unspecified | 0.0068 | 0.0128 | 11.44 |
| PriSTI | Diffusion | feature | 0.0135 | 0.0300 | 22.61 |
| LinearInterp | Interpolation | feature | 0.0138 | 0.0424 | 20.67 |
| **SAITS** | Self-attention | block | **0.0171** | **0.0483** | 27.22 |
| Transformer | Self-attention | block | 0.0181 | 0.0526 | 28.53 |
| CSDI | Diffusion | block | 0.0185 | 0.0568 | **26.02** |
| LinearInterp | Interpolation | block | 0.0192 | 0.0490 | 35.69 |
| SAITS-paper-mask | Self-attention (ablation) | block | 0.0214 | 0.0619 | 28.38 |
| GRU | DL | unspecified | 0.0312 | 0.0650 | 30.33 |
| PriSTI | Diffusion | block | 0.0321 | 0.0674 | 50.66 |

Sources: `tables/kwh_30mar_headline.csv` and `figures/headline_kwh_30mar.png`. Exact values: SAITS feature MAE 0.004001, SAITS-paper-mask 0.003844, CSDI 0.004313.

- **Feature-level:** SAITS matches MissForest on MAE (0.0040 vs 0.0040). It has the lowest RMSE of every method (0.0092), and its MAPE is under 9 %, against 24 % for MissForest.
  - The ablation model scores even lower MAE (0.0038). That fits this condition: 30 % random cells is close to the 20 % random cells it was trained on.
- **Block-level** (the dataset's real failure pattern): SAITS is the most accurate model on MAE and RMSE. It cuts CSDI's MAE by 8 % (0.0171 vs 0.0185) and RMSE by 15 %.

---

## 2. Overall picture (averaged over 3 mechanisms × 5 rates × 5 variables)

**Normalised MAE (NMAE) = MAE ÷ that variable's training standard deviation.** It puts Voltage (around 250 V) and kWh (around 0.07) on one scale.

The raw-MAE averages in `../comparison_report.md` §2a are dominated by Voltage. NMAE averages are not, so this report uses NMAE for anything averaged across variables. Per-variable tables in raw units are in §5.

| Model | NMAE feature | NMAE block | MAPE feature (%) | MAPE block (%) |
|---|---|---|---|---|
| **SAITS** | **0.0904** | **0.2157** | 9.91 | **25.82** |
| CSDI | 0.0944 | 0.2337 | **9.62** | 26.74 |
| Transformer | 0.1012 | 0.2275 | 11.48 | 27.89 |
| SAITS-paper-mask | 0.1259 | 0.2777 | 12.11 | 27.35 |
| LinearInterp | 0.2375 | 0.2515 | 31.18 | 34.68 |
| PriSTI | 0.3205 | 0.4562 | 44.08 | 67.06 |

Source: `tables/summary_model_style.csv`.

- SAITS has the lowest average error in both styles: 4 % below CSDI for feature-level and 8 % below for block-level.
- CSDI keeps a marginally lower feature-level MAPE (9.62 % vs 9.91 %).
- **Mechanism barely matters**, the same as for CSDI. MCAR, MAR and MNAR give NMAE within 2 % of each other for every model and style. For SAITS the spread is under 0.7 % (`tables/summary_model_mechanism_style.csv`).

---

## 3. By missingness rate: where SAITS wins and where CSDI still wins

**Feature-level masking, NMAE**

| Rate | SAITS | CSDI | Transformer | LinearInterp | SAITS-paper-mask | PriSTI |
|---|---|---|---|---|---|---|
| 10% | 0.0377 | 0.0339 | 0.0523 | 0.1780 | **0.0324** | 0.2130 |
| 30% | 0.0463 | 0.0438 | 0.0580 | 0.1957 | **0.0432** | 0.2362 |
| 50% | **0.0657** | 0.0662 | 0.0761 | 0.2181 | 0.0696 | 0.2711 |
| 70% | **0.1053** | 0.1111 | 0.1152 | 0.2535 | 0.1370 | 0.3335 |
| 90% | **0.1969** | 0.2168 | 0.2046 | 0.3424 | 0.3473 | 0.5489 |

**Block-level masking, NMAE**

| Rate | SAITS | CSDI | Transformer | LinearInterp | SAITS-paper-mask | PriSTI |
|---|---|---|---|---|---|---|
| 10% | **0.1783** | 0.1892 | 0.1923 | 0.1922 | 0.2198 | 0.3797 |
| 30% | **0.1896** | 0.2030 | 0.2027 | 0.2085 | 0.2343 | 0.3828 |
| 50% | **0.2028** | 0.2199 | 0.2138 | 0.2302 | 0.2526 | 0.4002 |
| 70% | **0.2264** | 0.2452 | 0.2366 | 0.2680 | 0.2878 | 0.4699 |
| 90% | **0.2812** | 0.3114 | 0.2918 | 0.3585 | 0.3938 | 0.6485 |

**Lowest MAE per cell.** For each style and rate there are 15 cells (3 mechanisms × 5 variables). The table counts how many each model wins (`tables/wins_lowest_mae_by_style_rate.csv`).

| Style | Rate | SAITS | CSDI | SAITS-paper-mask | others |
|---|---|---|---|---|---|
| block | 10% | 14 | 0 | 1 | 0 |
| block | 30–90% | 15 each | 0 | 0 | 0 |
| feature | 10% | 0 | 9 | 6 | 0 |
| feature | 30% | 0 | 9 | 6 | 0 |
| feature | 50% | 3 | 9 | 3 | 0 |
| feature | 70% | 14 | 1 | 0 | 0 |
| feature | 90% | 15 | 0 | 0 | 0 |

- **Block-level:** SAITS wins **74 of 75** cells, at every rate.
- **Feature-level up to 50 %:** CSDI is still slightly better (9 of 15 cells at each rate). At 10 % CSDI's NMAE is 0.0339 against 0.0377 for SAITS.
- **Feature-level at 70–90 %:** SAITS takes over, with 29 of 30 cells.

So the split is clear. With most sensors still visible, a 20-sample diffusion median is a very strong estimator. Once whole timesteps or most of the window are missing, SAITS is more accurate.

Source: `figures/all_models_nmae_vs_rate.png`. `figures/kwh_mae_vs_rate_all_models.png` shows the same comparison for kWh only under MAR, with MissForest marked.

---

## 4. Ablation: does our training-mask change matter? Yes, a lot.

The SAITS paper trains by hiding **20 % of cells at random**. Our runs of SAITS and the Transformer hide **10–90 % of cells**, half the time as whole-timestep blocks, which is the same distribution CSDI was trained with. Everything else is identical between the two SAITS runs.

| | NMAE feature | NMAE block | feature 90 % | block 90 % |
|---|---|---|---|---|
| SAITS (our masking) | **0.0904** | **0.2157** | **0.1969** | **0.2812** |
| SAITS (paper masking) | 0.1259 | 0.2777 | 0.3473 | 0.3938 |

- Our masking lowers average error by **28 % (feature-level) and 22 % (block-level)**. At 90 % feature-level, the paper-masked model's error is 76 % higher.
- The paper-masked model only wins where test conditions resemble its training: 10–30 % random cells. That includes the kWh 30 % MAR headline cell.
- **Takeaway:** the training masks must match the missingness you expect at test time. Copied unchanged from the paper, SAITS would have looked worse than CSDI here.

---

## 5. Per variable (raw units, averaged over all 30 conditions)

| Variable | SAITS | CSDI | Transformer | LinearInterp | PriSTI |
|---|---|---|---|---|---|
| Voltage (V) | **1.1270** | 1.1441 | 1.2118 | 1.3236 | 3.0327 |
| Current (A) | **0.2093** | 0.2268 | 0.2255 | 0.3696 | 0.5650 |
| kWh | **0.0134** | 0.0147 | 0.0142 | 0.0199 | 0.0294 |
| kVAh | **0.0137** | 0.0147 | 0.0146 | 0.0236 | 0.0333 |
| Neutral Current (A) | **0.2087** | 0.2250 | 0.2271 | 0.3674 | 0.5686 |

Source: `tables/summary_model_variable.csv`.

SAITS has the lowest average MAE on every variable.

---

## 6. Why linear interpolation loses here, unlike in the TSI-Bench benchmark

TSI-Bench (arXiv 2406.12747) found linear interpolation *beat* SAITS on its Electricity dataset. On our data it is far behind on feature-level masking (NMAE 0.2375 vs 0.0904, 2.6× worse) and only close on block-level at 10 % (0.1922 vs 0.1783).

The reason is the sensors themselves. In the March 2025 training data, **Current, kWh, kVAh and Neutral Current are correlated at r ≥ 0.98** with each other, while Voltage is nearly independent (|r| < 0.1).

- **Feature-level gaps:** the other sensors at the same 15-minute slot almost determine the missing value. The learned models exploit that; interpolation can't.
- **Block-level gaps:** the whole row is gone, so every method has to infer from time alone. Interpolation, a purely temporal method, becomes competitive at low rates. It falls behind again as gaps get longer (0.3585 vs 0.2812 at 90 %).

TSI-Bench's Electricity data has one consumption series per client (370 different clients), so it has no near-duplicate sensors from the same meter. That is a plausible reason the two benchmarks disagree; we have not tested it directly.

---

## 7. What SAITS learned (`figures/saits_attention_and_weights.png`)

The figure is for window 510 at 50 % MCAR block masking.

- **Attention, second block.** The diagonal is exactly zero: each timestep is blocked from attending to itself (diagonal masking). Most weight goes to the immediately neighbouring timesteps, plus a few "anchor" slots around 08:00, 14:00 and 19:00 that many timesteps attend to.
- **Combining weight η.** Its average is 0.37 on missing timesteps and 0.17 on observed ones. So the model leans more on the first block's estimate X̃1 exactly where data is missing, and otherwise on the second block's estimate X̃2.

`figures/attention_daily_profiles.png` shows SAITS, Transformer and LinearInterp on the same 4 windows and masks as `../figures/daily_profiles.png`. Across long blank stretches (for example, window 585 from 16:00 to 18:00), interpolation draws straight lines while SAITS follows the day's shape.

---

## 8. Trade-offs and caveats for the report

- **No uncertainty from SAITS.** It gives one point estimate, so its CRPS is NaN in the CSVs. For a point forecast CRPS would simply equal MAE. CSDI's predictive distribution (CRPS 0.0031 for kWh at 30 % MAR) remains a capability SAITS doesn't have.
- **Cost.** SAITS imputes all 30 conditions × 1,000 windows in **7.3 s**, against about 5 hours for CSDI's 20-sample evaluation, roughly **2,500× faster** at inference. It trains in 23 minutes, against about 40 for CSDI. It is about 6× larger (1.33 M vs 0.23 M parameters), because we used the paper's SAITS-base size rather than matching CSDI's.
- **One training seed per model.** Differences of a few percent (for example, SAITS vs CSDI feature-level at 50 %, 0.0657 vs 0.0662) should be read as ties. The clear-cut findings are the block-level advantage (74/75 cells), the high-rate feature-level advantage, and the ablation gap.
- **The Transformer early-stopped at epoch 52.** Its validation curve was noisier (`figures/attention_training_curves.png`). A longer patience might narrow its gap to SAITS, but under the same protocol it ranks behind SAITS everywhere.
- **Baselines are from a different month and an unspecified masking style.** This carries over from `../comparison_report.md` §4.
- **The port is exact.** A verification script (in the private project repository) loads our weights into the official SAITS and Transformer classes. Imputations and losses match with zero difference.

---

## 9. Files

| Path | Contents |
|---|---|
| `tables/` | Merged tables with CSDI and PriSTI. `unified_results_all_models.csv` is the 900-row ground truth. |
| `figures/all_models_nmae_vs_rate.png`, `kwh_mae_vs_rate_all_models.png`, `headline_kwh_30mar.png` | Cross-phase comparison (from the analysis script) |
| `figures/attention_*.png`, `saits_attention_and_weights.png` | Figures from the Kaggle notebook |
| `../../docs/P51_self_attention_report.pdf` | Explanation report (PDF) for this phase |

Raw per-run outputs, trained weights and code are kept in the private project repository.
