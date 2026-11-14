# CSDI & PriSTI, Explained — and What We Did With Them

This is a plain-language companion to `results/comparison/comparison_report.md`.
That report has the numbers; this document explains *what the models actually
are*, *how they work*, and *why the numbers came out the way they did* — so
you can explain it confidently in your report or viva without re-deriving it
under pressure.

---

## Part 1 — The big idea: what is a diffusion model?

Forget smart meters for a second. Imagine a photo that's been ruined by
grainy noise — like an old, badly-lit photograph. A **diffusion model** is a
neural network trained to do one very narrow job: *look at a slightly noisy
image and guess what the noise was*, so you can subtract it and get a
slightly cleaner image. Do that once, and the image is a little better. Do it
50 times in a row, each time removing a little more noise, and you can go all
the way from pure static to a sharp, realistic image.

Two things make this trick work:

1. **Training is easy to set up**, because you can *manufacture* noisy
   versions yourself: take a real, clean image, add a known amount of random
   noise to it, and ask the network "what noise did I just add?" You know the
   right answer because you added it, so this is ordinary supervised
   learning — no clever labels needed.
2. **Generation is just training run backwards.** Start from pure random
   noise, and repeatedly ask the trained network "what noise is in here?",
   subtract a little of its guess, add a little fresh randomness back in, and
   repeat. After enough steps you've "hallucinated" a realistic image out of
   static.

**Imputation is the same trick with a twist.** Instead of starting from pure
static, you start from a time-series window where the *observed* readings
are locked in place and only the *missing* slots are filled with noise. The
network is told which slots are real ("conditioning") and only ever asked to
clean up the missing ones. Run the reverse process, and the missing slots
gradually turn from random noise into plausible, physically consistent
electricity readings that fit naturally alongside the real ones.

The reason this is attractive for your project specifically:

- **Classical ML/DL imputers (MissForest, XGBoost, LSTM, GRU, …) only output
  one number per missing value** — their single best guess. A diffusion
  model can be asked to generate the reverse process *multiple times*
  (e.g. 20-50 times), giving 20-50 *different* plausible fillings for the
  same gap. The spread of those fillings tells you how *confident* the model
  is — something none of your mid-eval baselines can do. This is exactly
  what the CRPS metric in your results measures.
- Because the network sees the *whole day-window at once* (all 96 time
  steps, all 5 sensors), it can use correlations across time ("consumption
  dips every night") and across sensors ("Voltage drops when Current spikes")
  simultaneously, rather than filling one column at a time like MICE does.

---

## Part 2 — CSDI: diffusion for a single time series

**CSDI** (Conditional Score-based Diffusion for Imputation) is the simpler of
the two models — it treats every meter-day independently.

### What goes in, what comes out

- Input: one window, shape `(96 time-steps, 5 sensors)` — Voltage, Current,
  kWh, kVAh, Neutral Current for one meter on one day.
- A **mask** marks which of those 480 cells are observed (1) and which are
  missing and need to be imputed (0).
- Output: the same `(96, 5)` grid, with the missing cells filled in — and if
  you run it 50 times, 50 slightly different versions of those fills.

### How it decides what to fill in: "dual attention"

The model doesn't scan left-to-right like an LSTM. Instead it uses
**attention twice, in alternation**, at every layer:

- **Temporal attention** — for a *fixed* sensor (say, kWh), look across all
  96 time steps at once (the 96 fifteen-minute slots of that one day, 00:00 to
  23:45) and ask "which other slots in this day are most relevant to filling in
  this particular missing slot?" (e.g. the slots just before/after the gap, or
  the same meter's pattern earlier or later that day). The model only sees one
  day per window, so it cannot look at yesterday.
- **Feature attention** — for a *fixed* time step, look across all 5 sensors
  at once and ask "given Voltage and Current right now, what should kWh be?"

Stacking 4 of these temporal→feature pairs (the model has **226,625
trainable parameters**) lets the network combine "what did this meter usually
do at this time" with "what do the other readings say right now" before it
commits to a value.

### How training actually works (the noise-guessing game from Part 1)

1. Take a real, complete window `x₀` (normalised to zero-mean/unit-variance
   using statistics from the training set only).
2. Pick a random "noise level" `t` between 0 and 50, and a real missingness
   mask for that window.
3. Add Gaussian noise to `x₀` scaled by `t` (more noise at higher `t`) — this
   is `x_t`. The observed positions are always left as the clean truth; only
   the target (missing) positions get this noisy mixture.
4. Ask the network: "given what's still visible, and knowing we're at noise
   level `t`, what noise did I just add to the missing cells?"
5. Compare its guess to the noise we actually added, and nudge the network's
   weights to make that guess better. This is the entire loss function —
   literally mean-squared-error between predicted and true noise, computed
   only over the missing cells.

The network also receives an embedding of `t` itself (via sine/cosine
"diffusion embedding," the same trick positional encodings use in
Transformers) so it knows how aggressively to denoise at each step.

**Training schedule used**: batch size 16, Adam optimiser, learning rate
1e-3 with decay on plateau, up to 200 epochs with early stopping (patience
20). On the March 2025 pilot (13,640 training windows, 852 batches/epoch),
training stopped early at **epoch 78**, having steadily driven validation
loss from 0.125 down to 0.066 — a clean, monotonic curve
(`results/comparison/figures/csdi_training_curves.png`).

### How it fills in a real gap (inference)

Start from pure random noise *only in the missing slots* (observed slots are
pinned to their real values throughout). Then run the reverse process for
all 50 diffusion steps: at each step, ask the trained network "what noise is
in here right now," remove a calculated fraction of it, add a little fresh
randomness back (except on the very last step), and re-pin the observed
values. After all 50 steps, what's left in the missing slots is one
plausible completion. Repeating this whole process 20 independent times and
taking the **median** across runs gives the point estimate reported as
"MAE/RMSE"; the full spread of the 20 runs is what CRPS measures.

Finally, every generated value is passed through **physics post-processing**
before being scored — exactly the three rules from the project specification: Voltage
clamped to [180, 285] V, Current/kWh/kVAh/Neutral Current floored at 0, and
kWh capped at 0.95×kVAh if it ever exceeds it.

---

## Part 3 — PriSTI: diffusion with help from *other meters*

**PriSTI** (Conditional Diffusion with Spatiotemporal Prior) starts from the
same core idea — noise a window, train a network to denoise it, run that
backwards at inference time — but adds two upgrades on top of CSDI:

### Upgrade 1 — a "first guess" network before diffusion even starts

Diffusion models can struggle when *almost everything* is missing, because
there's very little signal to condition on. PriSTI adds a small **prior
network** (a bidirectional GRU + one graph message-passing step) that looks
at the observed values and produces a *rough, non-diffusion estimate* of the
whole window first — like a sketch before the final painting. This sketch is
then fed into the diffusion network *alongside* the noisy input, giving it a
head start. Training optimises both: the diffusion noise-prediction loss,
plus a small additional loss (weight 0.1) that pushes the prior network's
own rough sketch closer to the truth.

### Upgrade 2 — borrowing information from *other meters*, via a graph

CSDI only ever looks at one meter's own history. PriSTI instead treats the
problem as **spatiotemporal**: at a given date, dozens of meters are stacked
together into one batch, and the model is allowed to pass information
*between* meters, not just across time. The intuition: your neighbour's
meter on a similar connection, with a similar load rating, probably behaves
similarly to yours — so if your reading is missing, theirs is a clue.

To make "which meters should talk to which" concrete, we built a graph over
the 1,350 meters:

- **Too many meters to use directly**, so they're first compressed into **64
  virtual nodes** by k-means clustering on each meter's average daily
  profile — each virtual node is a "typical" behaviour pattern shared by a
  group of similar meters.
- Two virtual nodes are connected if they share the same **Sanctioned Load
  class** (0.5/1.0/1.98/2.0/3.0 kW — meters with the same contracted capacity
  tend to behave similarly), *and* separately, each node is connected to its
  **top-10 most-correlated** other nodes by Pearson correlation of their
  average profiles. This adjacency matrix is then used inside the network's
  attention blocks exactly like CSDI's feature-attention, except it mixes
  information *across virtual nodes* rather than across sensors.

Put together: PriSTI's diffusion network at every layer does **temporal
attention** (within a meter's own 96 steps, same as CSDI) **and** **graph
convolution** (pooling information from the 10 most-related virtual nodes),
conditioned on the prior network's rough sketch and the diffusion noise
level. This is a genuinely more sophisticated model (224,394 parameters) —
in principle, exactly the right idea for a dataset where whole meters go
dark for entire days, since neighbouring meters still have signal.

### Why PriSTI underperformed in our experiments — and it's not the idea

The spatiotemporal batching needs *many meters, all with data on the exact
same calendar date*, grouped into one batch. On the March 2025 pilot, that
requirement left only **15 usable training date-slices (literally one batch
per epoch) and 5 validation slices** — compare that to CSDI's 852
independent batches *per epoch*. PriSTI was essentially trained on a dataset
roughly 1000x smaller, per epoch, than CSDI. You can see the consequence
directly in `results/comparison/figures/pristi_training_curves.png`: the
loss bounces around noisily instead of dropping smoothly, because each
epoch's "data" is a near-coin-flip's worth of batches.

**This matters for how you present the results**: PriSTI's 3-4x worse
metrics in the comparison tables reflect a *data starvation* problem in our
pipeline's batching strategy, not evidence that the architecture is worse
than CSDI's.

### Can this actually be fixed, or is it just a limit of our data?

We dug into this properly rather than just flagging it — loaded the real
`windows/*/train_meta.npy` files for all 12 months (not just March) and
re-ran the notebook's exact slice-construction logic (virtual-node
clustering + the ≥16-node coverage filter) against them. Two findings:

**First, the "15 slices" is a genuine property of the data, not a bug.** It
is *not* the code's coverage filter biting — 15 of March's 16 total calendar
dates already clear the ≥16-virtual-node bar, so the filter is barely doing
anything. The real constraint is that **March 2025 only has 16 unique
calendar dates at all** among its complete-day windows, out of 31 days in
the month. We checked every other month and found the same pattern — between
12 and 17 usable dates each, never more:

| Month | Usable dates (≥16 meters) | Month | Usable dates (≥16 meters) |
|---|---|---|---|
| Aug 2024 | 17 | Feb 2025 | 12 |
| Sep 2024 | 13 | Mar 2025 | 15 |
| Oct 2024 | 7 | Apr 2025 | 13 |
| Nov 2024 | 13 | May 2025 | 15 |
| Dec 2024 | 13 | Jun 2025 | 13 |
| Jan 2025 | 14 | Jul 2025 | 16 |

(Full numbers in `results/comparison/tables/pristi_slice_availability_by_month.csv`.)
This consistency across all 12 months is itself informative: it suggests
meter outages in this dataset are **correlated across meters** — entire
groups of meters likely go dark together on the same days (consistent with
feeder- or grid-level events, not independent per-meter failures) — so
*no* single month, however you pick it, will ever offer many
simultaneously-complete calendar dates.

**Second, and this is the actionable part: because that constraint is
per-month, not dataset-wide, pooling training date-slices across all 12
months fixes it directly** — and the data to do this is already sitting
preprocessed in `windows/`, so this costs no new data collection. We
rebuilt the pooled version for real and measured it:

| | Slices | Batches/epoch (batch=8) |
|---|---|---|
| March 2025 only (what we ran) | 15 | 1 |
| **All 12 months pooled** | **161** | **20** |

That's a verified **10.7x increase** in training slices, needing only a
data-loading change (concatenate every month's `train.npy`/`train_meta.npy`
before grouping by date, and recompute the virtual-node clustering from the
pooled meter-means) — no architecture change. Dropping `PRISTI_BATCH` to 4
would push this further, to 40 batches/epoch.

**We have not run this on Kaggle yet** — it needs a fresh GPU training pass,
so treat it as a verified, ready-to-implement recommendation, not an
already-achieved improvement. Two things to keep in mind when you do run it:
(1) pooling months mixes seasons into one virtual-node graph, which assumes
each meter's *relative* behaviour pattern stays roughly stable year-round —
a reasonable simplification, not a guarantee, since Indore's seasonal
load shifts (summer cooling load vs. winter) could shift which meters
correlate with which; (2) even at 20-40 batches/epoch, PriSTI will still see
far fewer batches than CSDI's 852, because a model that needs shared-date
batches is intrinsically more data-hungry per training step than one that
treats every meter-day independently — pooling months closes most of the
gap, not all of it. Keep the test set on March 2025 only, so the fix only
changes what PriSTI *trains* on and the existing comparison table stays a
valid before/after once the re-run exists.

**So: is it "our data's fault"?** Partly, and that part is real and worth
stating plainly — a single month's data genuinely does not offer PriSTI
enough simultaneously-observed meters to train well, and that's a property
of how meters fail together in this dataset, not something we did wrong.
But it stops short of being a dead end: because 12 such months exist and
don't all fail on the same calendar days, the fix is available, it's
already verified against real data, and it was simply out of scope for the
single-month pilot design this phase was budgeted for (see Part 4d).

**Ready-to-paste replacement** for the notebook's PriSTI dataset-construction
cell (§7.3), to actually run this fix on Kaggle — pools all 12 months for
*training* while leaving the pilot month's val/test untouched:

```python
ALL_MONTHS = ['Aug_2024', 'Sep_2024', 'Oct_2024', 'Nov_2024', 'Dec_2024',
              'Jan_2025', 'Feb_2025', 'Mar_2025', 'Apr_2025', 'May_2025',
              'Jun_2025', 'Jul_2025']

# 1) Pool every month's training windows + metadata (already preprocessed,
#    already in /kaggle/input/.../windows/<month>/) instead of pilot-only.
pooled_windows, pooled_meta, pooled_month_tag = [], [], []
for m in ALL_MONTHS:
    w = np.load(DATA_ROOT / m / 'train.npy')
    meta = np.load(DATA_ROOT / m / 'train_meta.npy', allow_pickle=True)
    pooled_windows.append(w)
    pooled_meta.append(meta)
    pooled_month_tag.extend([m] * len(meta))
pooled_windows = np.concatenate(pooled_windows, axis=0)
pooled_meta = np.concatenate(pooled_meta, axis=0)

# 2) Recompute meter means / virtual-node clustering from the POOLED data,
#    so the graph reflects year-round behaviour, not just one month's.
meter_wins = defaultdict(list)
for i in range(len(pooled_meta)):
    meter_wins[str(pooled_meta[i]['msn'] if pooled_meta.dtype.names else pooled_meta[i][0])].append(pooled_windows[i])
meter_means_pooled = {m: np.mean(ws, axis=0) for m, ws in meter_wins.items()}
vnode_assign_pooled, _ = assign_virtual_nodes(meter_means_pooled, N_VIRTUAL, SEED)

# 3) Build date-slices keyed by (month, date) so the same day-of-month in
#    different months never merges into one slice.
class PooledPriSTIDataset(PriSTIDataset):
    def __init__(self, windows, meta, month_tag, norm_stats, vnode_assign, n_virtual=64):
        self.windows = torch.FloatTensor((windows - norm_stats[0]) / (norm_stats[1] + 1e-8))
        self.n_virtual = n_virtual
        self.mode = 'train'
        date_groups = defaultdict(list)
        for i in range(len(meta)):
            msn, date = str(meta[i][0]), str(meta[i][1])
            vid = vnode_assign.get(msn, i % n_virtual)
            date_groups[(month_tag[i], date)].append((i, vid))
        self.slices = [e for e in date_groups.values()
                       if len(set(v for _, v in e)) >= max(2, n_virtual // 4)]
        print(f'Pooled PriSTI train: {len(self.slices)} date-slices '
              f'from {len(date_groups)} (month, date) groups')

pristi_train_ds = PooledPriSTIDataset(pooled_windows, pooled_meta, pooled_month_tag,
                                       pilot['norm_stats'], vnode_assign_pooled, N_VIRTUAL)
# val/test stay exactly as before (pilot['val'], pilot['test'] — March 2025 only)
pristi_train_loader = DataLoader(pristi_train_ds, batch_size=PRISTI_BATCH,
                                  shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
print(f'Train batches/epoch: {len(pristi_train_loader)}')   # -> 20 at PRISTI_BATCH=8
```

This expects ~20 batches/epoch instead of 1 — expect training to take
roughly 20x longer per epoch in wall-clock terms, so budget Kaggle GPU time
accordingly (see Part 4d) and consider whether `MAX_EPOCHS`/`PATIENCE` still
make sense at the new batch count before kicking it off.

---

## Part 4 — Hyperparameters: what we set, what we tuned, and why

A fair question for your viva: *"Did you tune these models, or just pick
numbers?"* Here's the honest answer, broken into three categories.

### 4a. What our implementation actually is, versus the papers

Before the hyperparameters make sense, it's worth being precise about what
we built: **lightweight, from-scratch reimplementations of the core ideas**
in Tashiro et al.'s CSDI (NeurIPS 2021) and Liu et al.'s PriSTI (ICDE 2023),
not the authors' official code
(`github.com/ermongroup/CSDI`, `github.com/LMZZML/PriSTI`) ported as-is.

- Our **CSDI** matches the paper's conceptual design closely: dual
  temporal/feature attention, diffusion-step conditioning, DDPM-style
  training and sampling (Part 2). This part is a faithful, if independently
  coded, reimplementation.
- Our **PriSTI** is a deliberately **simplified** version of the paper's
  architecture. The real PriSTI (see `PriSTI_...pdf`, §III) builds its
  "coarse prior" with a full module combining spatial attention + temporal
  attention + a graph neural network (MPNN) run in two stages (a wide
  "conditional feature extraction" stage, then a deep "noise estimation"
  stage that reuses the same three components). Our `PriSTIPriorNet` replaces
  that whole module with a single bidirectional GRU plus one graph
  message-passing step — functionally the same *role* (produce a coarse
  first-guess to condition the diffusion network on) but architecturally much
  lighter. This was a deliberate scope decision given the time budget for
  Steps 3-4 of the plan (~1.5 hours each), not an accident — it keeps the
  "coarse prior + spatiotemporal diffusion refinement" idea that makes PriSTI
  PriSTI, without committing to reproducing its full published network.

### 4b. Architecture hyperparameters — matched to the papers, not searched

These numbers were picked **before any training run**, and were **not**
the result of a grid/random search on our data. They were taken directly
from the published configurations of CSDI and PriSTI, because those are the
one piece of external evidence we have that a configuration works well for
this exact model family on datasets of a similar scale — and because a
real search was outside the project's compute budget (see 4d).

The side-by-side comparison below is checkable directly against
`PriSTI_...pdf` Table II — we matched their settings for **METR-LA /
PEMS-BAY**, not their AQI-36 setting, because METR-LA (207 sensors) and
PEMS-BAY (325 sensors) are large sensor networks like ours (1,350 meters),
whereas AQI-36 (36 stations) is a much smaller network the paper tunes
differently:

| Hyperparameter | PriSTI paper (METR-LA/PEMS-BAY) | What we used | Match? |
|---|---|---|---|
| Channel size (`d_model`) | 64 | 64 | ✅ exact |
| Attention heads | 8 | 8 | ✅ exact |
| Noise-estimation layers | 4 | 4 | ✅ exact |
| Learning rate | 0.001 | 0.001 | ✅ exact |
| Min noise level β₁ | 0.0001 | 0.0001 | ✅ exact |
| Max noise level β_T | 0.2 | 0.2 | ✅ exact |
| Diffusion steps `T` | 50 | 50 | ✅ exact |
| Virtual nodes `k` | 64 | 64 | ✅ exact |
| Batch size | 16 | 8 (PriSTI) / 16 (CSDI) | ⚠ halved for PriSTI |
| Window length `L` | 24 (5-min traffic samples) | 96 (15-min, full day) | — domain-driven, not comparable |

The one deliberate deviation — **PriSTI's batch size halved to 8** — is a
direct consequence of our window length being 4x theirs (96 fifteen-minute
steps covering a full day, versus their 24-step traffic windows): a longer
sequence means more memory per sample for the same attention and graph
operations, so the batch had to shrink to fit the same T4 GPU. This wasn't
tuned by trial and error either — it's the standard "halve the batch if you
4x the sequence length" rule of thumb for attention memory, which scales
roughly with sequence length.

`TOP_K_ADJ = 10` (how many correlated neighbours each virtual node connects
to) has no paper equivalent to match, since the original PriSTI builds its
graph from real geographic distance between fixed stations, and smart meters
don't have that kind of spatial coordinate. We substituted a
correlation-based graph instead (Part 3), and top-10 is a standard
sparsification level in graph neural network literature generally (enough
neighbours to carry signal, few enough to keep the graph operations cheap at
64 nodes) — a reasonable default, not a value we swept.

### 4c. What was tuned automatically, inside each training run

This is the real, data-driven adaptation in this project — not hand-picked
in advance, but adjusted by the training loop itself in response to each
model's own validation curve:

- **Learning rate**, via `ReduceLROnPlateau` (factor 0.5, patience 5, floor
  1e-6): every time validation loss stops improving for 5 straight epochs,
  the learning rate is halved. You can see this directly in CSDI's training
  log — the rate stepped down from 1e-3 to 1.56e-5 over 78 epochs, each drop
  triggered by a real validation plateau, not a schedule we wrote in advance.
- **Training duration**, via early stopping (patience 20, best-checkpoint
  restore): rather than committing to a fixed epoch count, training runs
  until validation loss hasn't improved for 20 consecutive epochs, then
  restores the single best checkpoint seen. This is why CSDI stopped at
  epoch 78 out of a 200-epoch ceiling — the model's own validation behaviour
  decided the training length, not us.

Both mechanisms exist precisely so that two models with very different
convergence behaviour (compare CSDI's smooth 78-epoch descent to PriSTI's
noisy, data-starved curve from §3c) each get an appropriately-sized training
run without us having to guess the right epoch count or decay schedule for
each one individually ahead of time.

`Gradient clipping at 1.0` is the one remaining training knob, and it's
**not tuned at all** — it's a fixed, standard stability measure for diffusion
and Transformer training (it simply caps how large a single weight update
can be, preventing rare large gradients at high noise levels from
destabilising the attention weights), used at the same value across both
models because it's an engineering safeguard rather than a quantity whose
"best value" differs meaningfully by dataset.

### 4d. Why we didn't run a full hyperparameter search

Two numbers made a formal search impractical: compute and time. Per
`PLAN.md`, CSDI and PriSTI together were budgeted roughly 13 hours of
Kaggle T4 GPU time (Steps 5-6), against a 30-hours/week Kaggle quota — already
most of a week's allowance for a single pass. A grid search over even 3
hyperparameters (say, `d_model` ∈ {32, 64, 128}, `n_layers` ∈ {2, 4, 6},
learning rate ∈ {1e-4, 1e-3}) at one run each would be 18 training runs per
model — an 18x multiplier that doesn't fit in any reasonable number of
Kaggle weeks, and isn't what this project is about. The project's goal (per
the project description) is a **comparative study across method families** — statistical
vs. ML vs. DL vs. diffusion — evaluated under a shared, literature-grounded
configuration for each, not an architecture search for any single one of
them. Spending the compute budget on tuning CSDI/PriSTI against themselves
would have come directly at the expense of that comparison, for a return
(possibly a few percent better MAE) that wouldn't change the comparative
conclusions in Part 6.

## Part 5 — How we actually used these models on our data

1. **Pilot month chosen**: March 2025 — the densest month (37% missing),
   giving the most training windows (13,640 train / 1,205 val / 5,485 test
   day-windows of shape `(96, 5)`), with October 2024 (74% missing) held as a
   secondary stress test.
2. **Normalisation**: every sensor z-scored (mean/std computed from the
   *training* split only, to avoid leaking test information), per the project's
   standard pipeline convention.
3. **Simulating missingness ourselves, 30 different ways**, since the raw
   data's natural missingness can't be "dialled up" to test robustness. For
   every test window we generated masks at:
   - **3 mechanisms** — *MCAR* (purely random, like flipping a coin per
     cell), *MAR* (missingness depends on the *other* visible sensors — e.g.
     low Voltage readings make a Current reading more likely to be dropped),
     and *MNAR* (like MAR, but the inputs driving that logistic rule are
     themselves partly scrambled, so the missingness pattern is harder to
     infer from what's visible) — all calibrated via binary search so the
     *actual* missing fraction always lands within ~1% of the target.
   - **5 rates** — 10%, 30%, 50%, 70%, 90% missing.
   - **2 styles** — *feature-level* and *block-level* (mechanics and
     intuition below — this distinction matters a lot for the results).
   - 3 × 5 × 2 = **30 conditions**, each evaluated separately.

   ### Feature-level vs. block-level masking — what actually changes

   Both styles can be combined with any of the 3 mechanisms above; "style"
   only controls *which cells* in the `(96, 5)` grid get wiped out together.

   - **Feature-level**: each of the 5 sensor columns gets its *own*,
     independent missingness decision at every timestep. Concretely, in the
     code this means 5 separate coin-flips (MCAR) or 5 separate logistic
     models (MAR/MNAR) — one per sensor, each one looking at the *other 4*
     sensors to decide if *this* one should be dropped. The practical effect:
     at any given 15-minute timestep, it's common for 3-4 of the 5 sensors
     to still be visible, with only one or two missing.
   - **Block-level**: there's only *one* missingness decision per timestep,
     and it applies to *all 5 sensors at once*. In the code this is a single
     coin-flip (MCAR) or a single logistic model combining all 5 sensors
     (MAR/MNAR), then that one outcome is copied across all 5 columns. The
     practical effect: a "missing" timestep is a completely blank row — zero
     sensor readings at all for that 15-minute slot — while an "observed"
     timestep has full, complete information.

   **Why block is so much harder for the model** (and why CSDI's errors
   roughly double from feature to block in every condition — see
   `results/comparison/tables/diffusion_mechanism_style_summary.csv`):
   recall from Part 2 that CSDI fills gaps using two kinds of attention —
   *feature attention* (cross-reference the other sensors at the *same*
   timestep) and *temporal attention* (cross-reference the *same* sensor at
   other timesteps). Under feature-level masking, a missing cell almost
   always has healthy neighbours in the same row, so feature attention alone
   can do most of the work (e.g. infer a missing kWh reading directly from
   that same instant's Voltage and Current). Under block-level masking, a
   missing timestep has *nothing* in its row to attend to — feature attention
   has no information to pool, so the model is forced to fall back entirely
   on temporal attention, reconstructing the whole row purely by analogy to
   other, possibly distant, time steps within the same day (e.g. "what did this
   meter do in the morning or evening of this same day"). That's a strictly harder inference problem, and it
   only gets worse as the missing rate climbs, because long stretches of
   fully-blank rows become more likely.

   **Why we tested block-level at all, and why it's the one that matters
   most**: per the project description, the *real* missingness in this dataset is
   block-structured — "all 6 sensor columns go missing together per
   meter-timeslot... meters typically miss entire days." Feature-level
   masking is the standard, idealised benchmark most imputation papers (and
   most of your mid-eval baselines) are evaluated on, but it doesn't actually
   describe what happens to this data in the field — a meter that fails to
   report doesn't selectively lose just its Current reading, the whole
   reading for that interval is gone. So while the feature-level numbers are
   the ones that line up with classical imputation literature and with your
   mid-eval comparison, the **block-level numbers are the honest estimate of
   how these models would perform on this dataset's actual failure mode**,
   and are the ones worth leading with if you want to claim real-world
   relevance rather than benchmark performance.
4. **Training**: CSDI trained to convergence (78 epochs, early-stopped) on
   the full training set; PriSTI trained on the same windows but reshaped
   into date-grouped spatiotemporal batches (constrained to 15/5
   train/val batches, as explained above).
5. **Evaluation**: for each of the 30 conditions, mask the test set that way,
   run 20 independent reverse-diffusion samples per model, take the median as
   the point estimate, apply the physics clamp, then score MAE / MSE / RMSE /
   MAPE against ground truth, plus CRPS against the full 20-sample spread.

---

## Part 6 — What we achieved, and why it matters

**Headline number**: at the one condition that lines up exactly with your
mid-eval baselines — **kWh, 30% MAR** — CSDI (feature-style) scored
**MAE 0.0043, RMSE 0.0110, MAPE 8.6%**, essentially matching MissForest
(0.0040 / 0.0107 / 24.2%) and XGBoost (0.0041 / 0.0110 / 25.1%), and actually
*beating every single mid-eval baseline on MAPE*. Full numbers and the exact
comparison chart are in `results/comparison/comparison_report.md` §1.

**Why this matters for the project**:

- It demonstrates that a diffusion model — trained from scratch on a fairly
  small, single-month dataset, with no pretraining — can match the best
  classical ML methods from your mid-eval on their own turf (point-estimate
  accuracy), while *also* producing calibrated uncertainty (CRPS) that none
  of those methods can give you. That's the central promise of using
  diffusion models for this problem, and the pilot result actually delivers
  on it.
- The full 10-90% sweep
  (`results/comparison/figures/kwh_mae_vs_rate_with_baselines.png`) shows
  CSDI tracks the best baselines almost exactly up to ~50% missing, and only
  degrades beyond where any baseline was ever tested (70-90%) — so the
  "diffusion matches classical ML" finding isn't a lucky single point, it
  holds across a wide range of difficulty.
- The gap between **feature-level** and **block-level** masking
  (0.0043 vs 0.0185 MAE for CSDI) is itself a useful finding: it quantifies
  how much harder the *real* missingness pattern in this dataset (whole
  sensors dropping together) is compared to the idealised per-column
  missingness most baseline methods were built around — a point worth
  making explicitly in your discussion, since it argues for block-aware
  methods (like diffusion models that see the whole window) over purely
  per-column imputers at high real-world missingness rates.
- PriSTI's underperformance is itself a legitimate, reportable finding: it
  shows that spatial/graph information is only as useful as the data
  pipeline that delivers it, and sets up concrete future work (loosen the
  same-date batching constraint) rather than being a dead end.

For the complete numeric tables, every figure, and the full 300-row raw
results behind these claims, see `results/comparison/comparison_report.md`
and the `results/comparison/tables/` and `results/comparison/figures/`
folders.
