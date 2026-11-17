# SAITS & Transformer Imputation, Explained — and What We Did With Them

This is a plain-language companion to `results/self_attention/comparison_report.md`.
That report has the numbers; this document covers:
- how we chose the method from the literature;
- what SAITS actually is and how it fills gaps;
- what we changed to fit our data, and how we checked the code;
- how to talk about the results in the report or viva.

It follows the same structure as `docs/DIFFUSION_MODELS_GUIDE.md`.

---

## Part 1 — The big idea: imputation by "looking at the rest of the day"

A day-window is a 96 × 5 grid: 96 fifteen-minute slots × 5 sensors. When some cells are missing, there are two places to look for clues:

1. **Other sensors at the same moment.** If Current is missing at 18:15 but kWh and kVAh are visible, you can almost read the Current off them. In our data Current, kWh, kVAh and Neutral Current are correlated at **r ≥ 0.98** (measured on the March 2025 training windows).
2. **The same sensor at other moments.** What did this meter do at 18:00, 18:30, or at the same time on another part of the day?

**Self-attention** is a neural-network layer built for exactly the second kind of lookup. For every timestep it computes a set of weights, "how relevant is each other timestep to me?", and then forms a weighted mix of those timesteps' information.

Because the weights are learned and depend on the data, the model can learn rules a fixed method can't. For example: "the evening peak looks like the morning peak", or "ignore the flat overnight stretch when filling a spike".

Each timestep's input also carries all 5 sensor values plus a 5-bit "which of these are real" mask, so the first kind of clue (cross-sensor) is used too.

Unlike the RNN baselines (LSTM, GRU), attention doesn't read the day left to right. Every timestep can look directly at every other one, so a gap at 18:00 can borrow from 06:00 as easily as from 17:45.

---

## Part 2 — How we chose the method (literature review)

The brief was: *find a self-attention-based imputation method, preferably from IEEE or other peer-reviewed venues, check whether usable code exists, and run it on our data like CSDI and PriSTI.*

The criteria, in order:
1. peer-reviewed;
2. self-attention is the core of the model, not a side component;
3. works on a single multivariate window (our 96 × 5 days) rather than needing a graph of many sensors;
4. official, licensed code exists;
5. trains on a Kaggle T4.

Every entry below was checked against its DOI, publisher, IEEE Xplore, arXiv or GitHub page on 17 Nov 2026.

### 2a. General attention-based imputation methods

| Paper | Venue | Attention core? | Fits one 96×5 window? | Official code | Decision |
|---|---|---|---|---|---|
| **SAITS**: Du, Côté & Liu, *"SAITS: Self-Attention-based Imputation for Time Series"*, doi:10.1016/j.eswa.2023.119619 | *Expert Systems with Applications* 219:119619 (2023) | Yes, entirely | Yes, designed for it | github.com/WenjieDu/SAITS (MIT) | **Implemented, main model** |
| **MTSIT**: Yıldız, Koç & Koç, *"Multivariate Time Series Imputation With Transformers"*, doi:10.1109/LSP.2022.3224880 | *IEEE Signal Processing Letters* 29:2517–2521 (2022) | Yes, an encoder-only Transformer | Yes | None released | Represented by our **Transformer baseline**, the same model family (see Part 4) |
| **NRTSI**: Shan, Li & Oliva, doi:10.1109/ICASSP49357.2023.10095054 | IEEE ICASSP 2023 | Yes | Yes | lupalab/nrtsi, **no licence** | Not used: code has no licence, and its irregular-sampling design isn't needed here |
| **STING**: Oh, Kim, Ji & Khyalia, doi:10.1109/ICDM51629.2021.00155 | IEEE ICDM 2021 | Partly (GAN + bidirectional RNN + attention) | Yes | None found | Not used: no code; attention isn't the core |
| **ImputeFormer**: Nie et al., doi:10.1145/3637528.3671751 | ACM KDD 2024 | Yes (low-rank attention) | Partly; designed for many sensors | tongnie/ImputeFormer (MIT) | Not used: its strength (low rank across many nodes) has little to work with on 5 sensors |
| **SPIN**: Marisca, Cini & Alippi, arXiv:2205.13479 | NeurIPS 2022 | Yes (sparse graph attention) | No, needs a sensor graph | MIT | Not used: needs a graph of many sensors (the same problem that held back PriSTI) |
| **DeepMVI**: Bansal, Deshpande & Sarawagi | PVLDB 14(11), 2021 | Partly | Weak, needs many related series | **No licence** | Not used |
| **CDSA**: Ma et al., arXiv:1905.09904 | Venue unconfirmed | Yes | Weak, designed for location-tagged data | No official code | Not used |

### 2b. Attention-based imputation for power and smart-meter data

These show the approach is established in the power-systems literature, mostly IEEE:

| Paper | Venue | What it does | Code |
|---|---|---|---|
| **Load-PIN**: Li et al., doi:10.1109/TSG.2023.3293188 | *IEEE Trans. Smart Grid* 15(2):2251–2260 (2024) | A GAN whose refinement network uses self-attention; fills gaps in load profiles | None found |
| **BERT-PIN**: Hu, Ye, Kim & Lu, doi:10.1109/TII.2024.3417272 | *IEEE Trans. Industrial Informatics* 20(10):12241–12251 (2024) | A BERT-style Transformer that recovers multiple missing load segments, using load and temperature profiles | Public but **no licence**; built for load profiles, not multi-sensor windows |
| **Nayak et al.**, *"Data Imputation Using Self Attention Based Model for Enhancing Distribution Grid Monitoring and Protection Systems"*, doi:10.1109/TIM.2024.3372213 | *IEEE Trans. Instrumentation & Measurement* 73 (2024) | **Adapts SAITS** to impute missing data from distribution-level phasor measurement units (D-PMUs) in digital substations, including during fault events | None found |
| **Bahmani & Afrasiabi**, doi:10.1109/ISGTEUROPE62998.2024.10863227 | IEEE ISGT Europe 2024 | Self-attention recovery of PMU data during disturbances | None found |
| **Lotfipoor, Patidar & Jenkins**, doi:10.1016/j.enbuild.2023.113675 | *Energy and Buildings* 300:113675 (2023) | Transformer imputation of household demand under MCAR, MAR and MNAR, the closest match to our test design | None found |

### 2c. Benchmarks and surveys
- **TSI-Bench** (Du et al., arXiv:2406.12747) tested 28 imputers on 8 datasets. SAITS ranked first or second on several, including electricity-transformer data, but **linear interpolation won on its Electricity dataset**. That is why we added linear interpolation as a reference; on our data it loses clearly (see Part 7).
- **Wang, Du et al., "Deep Learning for Multivariate Time Series Imputation: A Survey"** (IJCAI 2025, arXiv:2402.04059) reports that attention models generally outperform RNN- and CNN-based imputers.

### 2d. Why SAITS
- It is the only candidate that meets every criterion.
- It is built entirely on self-attention and designed for exactly our single-window setting.
- Its official code is MIT-licensed, small (about 200 lines of PyTorch) and actively maintained.
- It has a direct IEEE power-grid precedent (Nayak et al., IEEE TIM 2024).
- Its paper title is literally the phrase in our brief: *Self-Attention-based Imputation for Time Series*.

The one gap is that SAITS appeared in an Elsevier journal rather than an IEEE one. The IEEE papers above (MTSIT, Load-PIN, BERT-PIN, Nayak et al.) give the IEEE context. MTSIT is represented in our experiments by the Transformer baseline.

---

## Part 3 — SAITS: how it works

### What goes in, what comes out
- **In:** the day-window with missing cells set to 0, plus the mask (1 = real reading, 0 = missing), concatenated into a 96 × 10 input.
- **Out:** a complete 96 × 5 window. Observed cells are copied through unchanged; only the gaps are filled.

There is no sampling and no noise schedule: **one forward pass, one answer**. That is the biggest practical difference from CSDI.

### Trick 1 — Diagonally-Masked Self-Attention (DMSA)

In ordinary self-attention, each timestep may also attend to *itself*. For imputation that's a shortcut the model shouldn't have: it could learn to copy its own input instead of reasoning from context.

SAITS **blocks the diagonal** of the attention matrix (each timestep's weight on itself is forced to zero). Every timestep must be reconstructed from the *other* 95, which is exactly the skill imputation needs. Our trained model shows this directly: the diagonal in `results/self_attention/figures/saits_attention_and_weights.png` is exactly zero.

### Trick 2 — Two passes, then a learned blend

```
Input [X ‖ m] → DMSA block 1 (2 attention layers) → X̃1         first draft
X' = observed values, with gaps filled from X̃1
[X' ‖ m]      → DMSA block 2 (2 attention layers) → X̃2         second draft (sees the first draft)
η = sigmoid(Linear([m ‖ block-2 attention]))                    per-cell blend weight
X̃3 = (1 − η)·X̃2 + η·X̃1                                         final estimate
```

- The **first block** makes a rough fill.
- The **second block** refines it, seeing a complete-looking window instead of one full of zeros.
- The model then **learns, cell by cell, which draft to trust** (η). The blend weight is computed from the mask and from block 2's attention pattern, so it can trust X̃1 more where block 2 had little real information to attend to.

In our trained model, the average η is 0.37 on missing timesteps and 0.17 on observed ones (`results/self_attention/figures/saits_attention_and_weights.png`).

### Trick 3 — Two training tasks at once (joint optimisation)
- **ORT (Observed Reconstruction Task):** reproduce the cells the model *can* see, averaged over X̃1, X̃2 and X̃3. This keeps the representations honest.
- **MIT (Masked Imputation Task):** we *hide* some cells whose true values we know, and score X̃3 on them. This is the actual imputation skill.

Both are mean absolute errors and are added with equal weight (1 + 1), as in the paper. Validation and early stopping use the MIT error on held-out validation windows.

---

## Part 4 — The Transformer baseline

The same official repository includes an encoder-only Transformer imputer, which the SAITS paper uses as its main baseline:
- one stack of ordinary self-attention layers on `[X ‖ m]`;
- **no diagonal masking, no second block, no blend**;
- the same ORT + MIT loss.

This is the same model family as MTSIT (IEEE SPL 2022), an encoder-only Transformer trained to reconstruct masked inputs. Since MTSIT's authors released no code, this is the closest faithful stand-in; we call it a re-implementation of the model family, not MTSIT's own code.

We gave it SAITS-base's width and **4 layers, matching SAITS's total depth** (2 blocks × 2 layers), so the two have almost the same parameter count (1.320 M vs 1.325 M).

Any gap between them is therefore what SAITS's two extra ideas add: diagonal masking, and two-pass refinement with a learned blend.

---

## Part 5 — Reusing the official code, and proving we didn't break it

- **Source:** github.com/WenjieDu/SAITS, commit `660b87f` (MIT licence).
- **Ported files:** `modeling/layers.py`, `modeling/saits.py`, `modeling/transformer.py`, plus the one-line masked-MAE helper.
- **Where the port lives:** inside the Kaggle notebook in the private project repository, so the notebook stays self-contained, like the CSDI and PriSTI notebook.

What we changed is plumbing only:
- explicit constructor arguments instead of `**kwargs`;
- the diagonal mask stored as a buffer, so it follows the model to the GPU;
- the same three-method interface as our CSDI class: `compute_loss`, `imputation_mae` and `impute`.

**Verification.** A verification script (in the private project repository) builds the official classes and loads our weights into them with strict key matching. Imputations and losses match with **zero numerical difference**, for both SAITS and the Transformer:

```
SAITS        params 1,325,094   max|imputation diff| 0.00e+00   loss ours 1.634987 vs official 1.634987
Transformer  params 1,320,453   max|imputation diff| 0.00e+00   loss ours 2.038383 vs official 2.038383
PORT MATCHES OFFICIAL IMPLEMENTATION
```

**Why not the PyPOTS library?** PyPOTS, by the SAITS author, also packages SAITS. We didn't use it for two reasons:
- Its 2.0 release was only 8 days old and can downgrade packages on Kaggle.
- Its training always hides a fixed 20 % of cells at random. That can't express our 10–90 % block masks without patching library internals, and Part 6 shows that difference matters a lot.

---

## Part 6 — Our one real modification: training masks that match the test

The SAITS paper's MIT hides **20 % of cells at random**, which suits benchmark datasets tested at 10–20 % random missingness. Our evaluation goes up to **90 %** and includes **block** masking (all 5 sensors missing in the same slot), which is the dataset's real failure mode.

So for SAITS and the Transformer we used the **same training-mask distribution CSDI was trained with** (`ImputationDataset`, copied verbatim): for each window, a missing rate drawn uniformly from 10–90 %, applied half the time as random cells and half the time as whole timesteps.

That keeps the comparison with CSDI like-for-like. It is also the right thing to do in general: a model should practise on the kind of gaps it will face.

**We measured whether it mattered** by also training an identical SAITS with the paper's 20 % masking ("SAITS-paper-mask"):
- Our masking lowered average error by **28 % (feature-level) and 22 % (block-level)**.
- At 90 % feature-level missingness, the paper-masked model's error was 76 % higher.
- The paper-masked model only wins at 10–30 % random cells, the conditions closest to what it trained on.

Lesson for the report: *the training-mask distribution matters as much as the architecture*. Copied unchanged from the paper, SAITS would have looked worse than CSDI.

---

## Part 7 — Hyperparameters and how we ran it

| Setting | Value | Why |
|---|---|---|
| SAITS layers | 2 blocks × 2 layers | Paper's "SAITS-base" config (`configs/*_SAITS_base.ini`) |
| d_model / d_ffn / heads / d_k = d_v | 256 / 128 / 4 / 64 | SAITS-base |
| Dropout / attention dropout | 0.1 / 0.0 | SAITS-base (the official code passes 0 attention dropout for SAITS) |
| Transformer | 4 layers, same widths, attention dropout 0.1 | Same depth and size as SAITS; dropout as in the official Transformer |
| Loss weights ORT / MIT | 1 / 1 | Paper |
| Optimiser | Adam, lr 1e-3, batch 128 | SAITS-base |
| Schedule | ReduceLROnPlateau (×0.5 after 5 stagnant epochs), early stopping with patience 20, max 200 epochs, gradient clipping 1.0 | Same loop as our CSDI training, for comparability |
| Training masks | Rate ~ U(10 %, 90 %), half feature-style, half block-style | Same as CSDI (Part 6) |
| Validation | Fixed 50 % MCAR mask per validation window; MIT MAE | SAITS paper's early-stopping metric |

We used the paper's base configuration rather than running a hyperparameter search. The paper uses it unchanged across all its datasets, and our single-seed budget is better spent on the ablation.

**Pipeline** (identical to the diffusion phase):
1. Pilot month March 2025: 13,640 train / 1,205 val windows.
2. Evaluate on the **same first 1,000 test windows**, with **masks generated by the same code and seed**. The notebook's 30-condition mask printout is identical to the CSDI Kaggle run's.
3. Impute, denormalise, apply the physics clamp, then compute MAE, MSE, RMSE and MAPE on the hidden cells, per variable.

Run time on one Kaggle T4:
- Training: 23 min (SAITS), 7 min (Transformer), 16 min (ablation).
- Evaluation: all 30 conditions × 1,000 windows in **7 seconds per model**. CSDI took about **5 hours** for the same evaluation, because it draws 20 samples × 50 denoising steps.

---

## Part 8 — What we found, and how to talk about it

**One-line summary:** *SAITS is the most accurate model we have tested on this dataset's realistic (block) missingness and at high missing rates. It matches the best mid-eval baseline on the headline condition, and it imputes about 2,500× faster than CSDI. CSDI remains slightly better when only a few random cells are missing, and it is the only model that gives uncertainty estimates.*

- **Headline (kWh, 30 % MAR, feature-level):**

  | Method | MAE | RMSE | MAPE |
  |---|---|---|---|
  | SAITS | 0.0040 | **0.0092** (lowest of all methods) | 8.8 % |
  | MissForest | 0.0040 | 0.0107 | 24 % |
  | CSDI | 0.0043 | 0.0110 | 8.6 % |

- **Block-level, the real-world pattern:** SAITS has the lowest MAE in **74 of 75** condition × variable cells. At kWh 30 % MAR it beats CSDI by 8 % on MAE (0.0171 vs 0.0185) and 15 % on RMSE.
- **Where CSDI still wins:** feature-level masking at 10–50 % (9 of 15 cells at each of those rates). With most sensors visible, a 20-sample diffusion median is hard to beat. From 70 % up, SAITS wins 29 of 30 feature-level cells.
- **What each SAITS idea adds:** compared with the plain Transformer of the same size, SAITS is better everywhere (NMAE 0.0904 vs 0.1012 feature-level, 0.2157 vs 0.2275 block-level). That gap is the value of diagonal masking plus two-pass refinement.
- **Why interpolation loses here:** the near-duplicate sensors (r ≥ 0.98) give learned models same-time clues that interpolation can't use. Under block masking those clues disappear, and interpolation becomes competitive only at low rates.
- **Mechanism (MCAR/MAR/MNAR) barely matters**, the same finding as for CSDI.

**Honest caveats to state:**
- SAITS gives no uncertainty; CRPS is reported as NaN. If calibrated intervals matter, CSDI is still the answer.
- There is one training seed per model, so treat differences of a few percent as ties.
- SAITS-base is about 6× larger than our CSDI (1.33 M vs 0.23 M parameters).
- The Transformer early-stopped at epoch 52.
- The mid-eval baselines are from a different month with unspecified masking style, the same caveat as the diffusion report.

**Natural future work:**
- multiple seeds with error bars;
- a capacity-matched SAITS (d_model 64) against CSDI;
- MC-dropout or ensembles to give SAITS uncertainty estimates;
- pooling all 12 months for training (as already proposed for PriSTI);
- testing on October 2024, the 74 %-missing stress-test month.
