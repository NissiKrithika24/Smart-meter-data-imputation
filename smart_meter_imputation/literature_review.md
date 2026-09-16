# Phase 1 — Literature Review: Deep Learning Architectures for Smart Meter Time-Series Missing Value Imputation

**Author / Project:** Research & Coursework on Smart Meter Missing Value Imputation  
**Dataset Context:** Indore Smart Meter Electricity Consumption (15-minute intervals, November 2024)  
**Target Variable:** Active Energy Consumption (kWh) under MCAR, MAR, and MNAR Missingness  

---

## 1. Executive Summary & Problem Formulation

Smart grid infrastructure relies on Advanced Metering Infrastructure (AMI) to collect electricity consumption data at regular intervals (typically 15-minute to 1-hour resolution). In real-world deployment, smart meter data streams are frequently compromised by missing values stemming from:
1. **Communication channel failures:** GPRS/cellular fading, packet collisions in mesh networks, or gateway timeouts (generating bursts of missingness, often conditioned on external factors like weather or time-of-day).
2. **Meter hardware outages:** Power brownouts, hardware resets, or sensor battery depletion.
3. **Data collection system maintenance:** Server downtimes and database synchronization anomalies.

Accurate missing value imputation is vital because downstream utility analytics—including non-technical loss (theft) detection, short-term load forecasting (STLF), tariff billing, and distribution transformer load management—require continuous, uncorrupted time-series matrices.

Mathematically, let an original univariate smart meter sequence be X = [x_1, x_2, ..., x_T]^T in R^T. Due to missingness, we observe an incomplete sequence X_tilde = [x_tilde_1, ..., x_tilde_T]^T, accompanied by a binary indicator mask M = [m_1, ..., m_T]^T in {0, 1}^T, defined as:
m_t = 1 if x_t is observed, and 0 if x_t is missing.

The objective of the imputation model f_theta(.) is to reconstruct an estimate X_hat = [x_hat_1, ..., x_hat_T]^T such that the error metric between x_t and x_hat_t for all timesteps where m_t = 0 is minimized.

---

## 2. Recurrent Network Architectures for Imputation

Traditional recurrent neural networks (standard LSTM and GRU) are formulated for sequential autoregression (x_t = g(x_{t-1}, x_{t-2}, ...)). When confronted with missing values, standard RNNs fail because:
- They assume uniform, uninterrupted temporal spacing between consecutive inputs (Delta t = const).
- They cannot inherently differentiate between a true observed zero consumption value (x_t = 0.0 kWh) and a zero-filled missing value (x_t = NaN -> 0).
- Standard causal RNNs only process unidirectional past context, completely disregarding future observed context that provides critical boundary constraints for interpolation.

To overcome these deficiencies, several seminal architectures have been introduced in the literature:

### 2.1 GRU-D (Gated Recurrent Unit with Temporal Decay)
* **Citation:** Che, Z., Purushotham, S., Cho, K., Sontag, D., & Liu, Y. (2018). 'Recurrent Neural Networks for Multivariate Time Series with Missing Values.' Scientific Reports (Nature), 8(1), 6085.
* **Mechanism for Masks & Time Gaps:**
  GRU-D explicitly introduces two auxiliary inputs: the missingness mask m_t and the continuous elapsed time since the last valid observation delta_t. It defines a continuous, monotonically decreasing exponential decay factor:
  gamma_t = exp(-max(0, W_gamma * delta_t + b_gamma)), gamma_t in (0, 1]
  GRU-D incorporates two distinct decay operations:
  1. **Input Decay:** When an input variable has been missing for a long duration, its imputed replacement decays from its last observed value toward the global empirical baseline mean:
     x_hat_t = m_t * x_t + (1 - m_t) * (gamma_t^(x) * x_{last} + (1 - gamma_t^(x)) * x_mean)
  2. **Hidden State Decay:** As the gap delta_t grows larger, past historical memory in the hidden state gradually fades toward zero:
     h_hat_{t-1} = gamma_t^(h) * h_{t-1}
* **Distinction from Plain GRU:** Plain GRU forces zero-imputation or forward-filling without decaying confidence, causing hidden state drift. GRU-D adapts its gating dynamics directly to the observation cadence.

### 2.2 BRITS (Bidirectional Recurrent Imputation for Time Series)
* **Citation:** Cao, W., Wang, D., Li, J., Zhou, H., Li, L., & Li, Y. (2018). 'BRITS: Bidirectional Recurrent Imputation for Time Series.' Advances in Neural Information Processing Systems (NeurIPS 2018), 31, 6775–6785.
* **Mechanism for Masks & Bidirectionality:**
  BRITS eliminates heuristic pre-imputation by treating missing entries as internal variables within a bidirectional recurrent computation graph:
  - **Forward Direction:** Estimates missing value x_hat_t using decayed history.
  - **Backward Direction:** Processes the sequence in reverse temporal order, estimating x_hat conditioned on future observed trajectory.
  - **Consistency Regularization:** BRITS introduces a dual-directional consistency loss forcing the forward and backward recurrent estimations to agree at unobserved points.
* **Distinction from Plain Bi-RNN:** A plain Bi-RNN requires complete input vectors. BRITS dynamically couples imputation estimation into the recurrence step, updating the hidden states using the model's own predictions at missing timesteps.

### 2.3 M-RNN (Multi-Directional Recurrent Neural Network)
* **Citation:** Yoon, J., Zame, W. R., & van der Schaar, M. (2019). 'Estimating Missing Data in Temporal Data Streams Using Multi-Directional Recurrent Neural Networks.' IEEE Transactions on Biomedical Engineering, 66(5), 1477–1490.
* **Mechanism:** Operates across two orthogonal coordinate axes:
  1. Temporal Dimension: Recurrent interpolation along individual feature trajectories across time steps.
  2. Feature Dimension: Multi-stream cross-sectional feedforward mapping capturing static inter-feature correlations (e.g. active power vs. reactive power vs. current).
* **Distinction from Plain RNN:** Plain RNN models only process temporal sequences and fail to leverage cross-sensor covariance structures.

### 2.4 SAITS (Self-Attention-based Imputation for Time Series)
* **Citation:** Du, W., Cote, D., & Liu, Y. (2023). 'SAITS: Self-Attention-based Imputation for Time Series.' Expert Systems with Applications, 219, 119619.
* **Mechanism:** A non-recurrent architecture based on modified Transformer attention. Replaces recurrence with Diagonally-Masked Self-Attention (DMSA) blocks to prevent timesteps attending to their own masked input, learning bidirectional temporal context and feature correlations.

### 2.5 MIDA (Multiple Imputation using Denoising Autoencoders)
* **Citation:** Gondara, L., & Wang, K. (2018). 'MIDA: Multiple Imputation using Denoising Autoencoders.' PAKDD 2018, LNCS 10939, 560–572.
* **Mechanism:** Overcomplete denoising autoencoder generating multiple imputations through stochastic dropout during inference. Lacks explicit inductive bias for continuous sequential time dynamics.

---

## 3. Comparative Literature Summary Matrix

| Metric / Dimension | GRU-D (Che et al., 2018) | BRITS (Cao et al., 2018) | M-RNN (Yoon et al., 2019) | SAITS (Du et al., 2023) | MIDA (Gondara et al., 2018) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Base Architecture** | Unidirectional GRU + Decay | Bidirectional RNN + Graph | Multi-Directional RNN | Diagonally-Masked Self-Attention | Overcomplete Denoising Autoencoder |
| **Missingness Modeling** | Mask m_t + Gap delta_t + Decay gamma_t | Graph-embedded variables + Cycle consistency | Intra-temporal + Inter-feature 2-stage | Diagonal masking + Joint objective | Dropout injection + Denoising objective |
| **Handling of Time Gaps** | Exponential decay gamma(delta_t) | Recurrent decay gamma(delta_t) | Sequence padding / fixed step | Positional embeddings | None (Feature agnostic) |
| **Bidirectional Context?** | No (Forward only) | Yes (Forward & Backward) | Yes (Bi-directional interpolation) | Yes (All-to-all attention) | No (Static reconstruction) |
| **Primary Datasets** | PhysioNet 2012, MIMIC-III, Air Quality | Air Quality (KDD Cup), PhysioNet, Human Activity | PhysioNet 2012, Synthetic clinical streams | PhysioNet 2012, Air Quality, Electricity | UCI benchmarks (Wine, Boston, etc.) |
| **Reported Metrics** | AUC-ROC, PR-AUC, Cross-entropy | MAE, MRE (Relative Error) | RMSE, MAE | MAE, RMSE, MRE | RMSE |
| **Computational Complexity** | O(T) sequential | O(2T) sequential | O(T * D) sequential | O(T^2) quadratic | O(D) feedforward |
| **Smart Meter Applicability** | Good for irregular arrival; lacks future context | High imputation accuracy; slow sequential training | High on multi-channel meters (V, I, kWh) | Strong on long contexts; compute heavy | Weak on sharp periodic diurnal motifs |

---

## 4. Application to Smart Meter & Electricity Consumption Data

### 4.1 Domain-Specific Properties of Smart Meter Time Series
Electricity consumption data recorded at 15-minute intervals exhibits distinct structural characteristics:
1. **Diurnal Seasonality (24-Hour Cycle):** Exactly 96 timesteps constitute one day (96 * 15 min = 1440 min = 24 h). Household demand follows routine human activity patterns: morning wake-up ramps, daytime lull, evening cooking/lighting peaks, and nocturnal dormant baseload.
2. **Weekly Periodicity (168-Hour Cycle):** Systematic variance between weekdays (Monday-Friday work schedules) and weekends (Saturday-Sunday shifted peaks).
3. **High Non-Linear Volatility:** Sharp, instantaneous spikes caused by cycling high-wattage inductive appliances (water pumps, air conditioners, geysers, induction cooktops).
4. **Physical Non-Negativity:** Active energy consumption is strictly bounded: x_t >= 0. Solar net-metered prosumers export power (kWh EXP > 0), which must be isolated from pure consumption meters.

### 4.2 Missingness Mechanisms in Smart Metering
* **MCAR (Missing Completely At Random):** Probability of missingness is independent of observed or unobserved data. In smart meters, this models isolated transmission packet loss or random network buffer dropouts.
* **MAR (Missing At Random):** Probability of missingness depends on observed auxiliary covariates (such as time of day, day of week, or scheduled utility maintenance windows), but not on the unobserved consumption level itself. In smart grids, cellular network congestion and maintenance outages frequently peak during nocturnal or early-morning windows.
* **MNAR (Missing Not At Random):** Probability of missingness is directly correlated with the underlying unobserved consumption value. In distribution networks, this represents meter overload dropouts, high-current breaker trips during maximum demand peaks, or deliberate tampering where meters are disconnected during high-consumption intervals.

### 4.3 Standard Evaluation Metrics in Energy Literature
1. **Root Mean Squared Error (RMSE):** Measures peak-load reconstruction fidelity with quadratic error penalization.
2. **Mean Absolute Error (MAE):** Linear error penalization representing expected physical billing error in kWh.
3. **Mean Absolute Percentage Error (MAPE):** Scale-independent relative error, guarded with an epsilon threshold to prevent division by zero during zero-consumption intervals.

---

## 5. Architectural Motivation: Why 1D Dilated ResNet Outperforms Recurrent Baselines

While recurrent architectures (LSTM, GRU, BRITS) have historically dominated sequence imputation, recent literature demonstrates that **1D Dilated Residual Convolutional Networks (ResNet-1D)** offer decisive theoretical and practical advantages for high-frequency smart meter imputation:

1. **Multi-Scale Temporal Receptive Field without Information Bottleneck:**
   Recurrent networks force temporal information through a single fixed-size hidden vector h_t, causing gradual attenuation of distant past signals across 96 timesteps. In contrast, a 1D ResNet with dilated convolutions (dilations: 1, 2, 4, 1) expands its effective receptive field exponentially, enabling simultaneous learning of fine-grained local transitions (adjacent 15-minute readings) and macroscopic diurnal shifts (morning to evening peak relationships).
2. **Mitigation of Vanishing Gradients via Residual Skip Connections:**
   In standard CNNs or deep RNNs unrolled across 96 steps, backpropagated gradients decay or explode over extended gaps. The ResNet residual mapping y = F(x) + x creates identity gradient highways, ensuring strong gradient flow directly to early feature extractors, even when imputing wide contiguous gaps.
3. **Elimination of Sequential Recurrence Bottlenecks (Massive Parallelism):**
   RNN step computation is strictly sequential (t -> t+1), precluding efficient hardware parallelization across time. 1D Convolutions execute parallel tensor operations over the entire 96-timestep window simultaneously, yielding substantial training and inference speedups on standard hardware.
4. **Direct Temporal Translation Invariance:**
   Daily energy consumption waveforms share invariant localized shapes (e.g. a 45-minute cooking load spike looks identical whether it happens at 19:15 or 20:00). 1D convolutional filter kernels naturally exploit translation invariance to detect and reconstruct these archetypal consumption patterns regardless of exact temporal onset.

---

## 6. References

1. Che, Z., Purushotham, S., Cho, K., Sontag, D., & Liu, Y. (2018). 'Recurrent Neural Networks for Multivariate Time Series with Missing Values.' Scientific Reports (Nature), 8(1), 6085.
2. Cao, W., Wang, D., Li, J., Zhou, H., Li, L., & Li, Y. (2018). 'BRITS: Bidirectional Recurrent Imputation for Time Series.' Advances in Neural Information Processing Systems (NeurIPS 2018), 31, 6775–6785.
3. Yoon, J., Zame, W. R., & van der Schaar, M. (2019). 'Estimating Missing Data in Temporal Data Streams Using Multi-Directional Recurrent Neural Networks.' IEEE Transactions on Biomedical Engineering, 66(5), 1477–1490.
4. Du, W., Cote, D., & Liu, Y. (2023). 'SAITS: Self-Attention-based Imputation for Time Series.' Expert Systems with Applications, 219, 119619.
5. Gondara, L., & Wang, K. (2018). 'MIDA: Multiple Imputation using Denoising Autoencoders.' PAKDD 2018, LNCS 10939, 560–572.
6. Wang, Z., Yan, W., & Oates, T. (2017). 'Time series classification from scratch with deep neural networks: A strong baseline.' IJCNN 2017, pp. 1578–1585.
7. Bai, S., Kolter, J. Z., & Koltun, V. (2018). 'An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling.' arXiv:1803.01271.
8. Tashiro, Y., Song, J., Yang, P., & Ermon, S. (2021). 'CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation.' NeurIPS 2021, 34, 24804–24816.
