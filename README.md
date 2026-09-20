# WFB Evaluation and the Motivation for WSP

WSP is discussed as a research motivation, not as an evaluated model in these results. No original experiment files or manuscript text were changed to produce this report.

## Executive Summary

The experiments support WFB as a trainable, interval-conditioned wave representation, but do not establish an accuracy advantage over strong, parameter-matched feed-forward models or demonstrate that correctly aligned intervals cause its gains. Real-time WFB improves mean ADE over the small position-only FFN by **7.0%** and over the tested random Fourier, learned Fourier, and Time2Vec controls by **21.9%, 13.9%, and 16.2%**, respectively. However, it is approximately tied with matched FFNs and is less accurate than the SIREN control and the sequence models.

These findings motivate a precise next question: **can the wave representation become more useful when elapsed time propagates a carried hidden state, instead of only conditioning an activation at each observation?** That is the motivation for Wave State Propagation (WSP). The results do not prove that recurrence is necessary, that sequence learning is the sole missing ingredient, or that WSP will outperform existing models.

## 1. What Was Evaluated

| Item | Completed protocol |
|---|---|
| Data | Windows derived from the ETH/UCY/JAAD-labeled annotation collection |
| Input state | Position only, `[x, y]`; no velocity or acceleration inputs |
| Observation / prediction length | 8 observed positions / 12 future positions |
| Splits | 322,160 train; 61,794 validation; 69,384 test windows |
| Temporal input | Stored timestamp differences; real, within-window shuffled, or constant intervals for WFB |
| Normalization | Shared cache with training-split normalization; a common cache fingerprint across models |
| Model families / seeds | 15 models; seeds 1, 2, 3, 4, 5 |
| Hyperparameter selection | Mean validation ADE over the same five seeds; test evaluated for the selected configuration |
| Search budget | 315 candidate/seed training runs; 75 selected test runs |
| Training | AdamW; up to 100 epochs; patience 15; batch size 512 |
| Learning-rate search | `1e-4`, `3e-4`, `1e-3` for each model family |
| Periodic controls | Fourier scales `0.1, 1, 10`; SIREN frequency settings `1, 10, 30`, selected on validation data |
| WFB configuration | **Standard updates, lambda = 0, full wave parameters, MLP decoder** |
| Hardware | Two B200 devices used to schedule separate model runs; not two-GPU inference for each model |
| ADE / FDE units | Normalized image coordinates, not meters |

All matched controls have 397,816 to 402,711 trainable parameters, within 1% of WFB's 401,560. The small position-only and time-input FFNs have 142,104 and 144,152 parameters. Parameter matching improves the comparison but does not make architectures, initialization, regularization, or optimization landscapes identical. For example, the local SIREN control does not use the WFB MLP's dropout layers.

This is a **standard-WFB representation evaluation**. It does not evaluate the Laplacian or combined update, a linear-decoder WFB, robustness, or WSP itself. The presence of lambda search options in configuration files does not mean a Laplacian sweep ran: all completed candidates in this suite have lambda zero.

### Three Different Evaluation Questions

| Question | Appropriate comparison | What this suite can establish |
|---|---|---|
| Does the representation help? | WFB versus matched FFN and periodic feed-forward controls with the same inputs | Relative performance of the tested parameterizations |
| Does the update rule help? | The same WFB architecture with standard, Laplacian, and combined updates | Not answered by this standard-only suite |
| Is the model competitive on trajectories? | WFB versus GRU, LSTM, and Transformer | Application-level performance, not an isolated test of backpropagation |

Sequence benchmarks are useful context, but their advantage cannot be attributed solely to a different learning rule. Conversely, the feed-forward controls remain relevant even when WFB's intended contribution is not a new sequence architecture.

## 2. Accuracy Results

<img width="3080" height="1628" alt="01_accuracy_comparisons" src="https://github.com/user-attachments/assets/290d19ec-9339-470a-8b32-681d7bfb5e7e" />

**Figure 1.** Test ADE for feed-forward representation controls and application-level sequence benchmarks. Bars show mean +/- standard deviation across seeds, not confidence intervals. All completed models are shown; WFB-real is repeated in the sequence panel as a reference. The panels answer different questions and use the same ADE scale.

| Model | ADE mean +/- SD | FDE mean | Parameters |
|---|---:|---:|---:|
| GRU + dt | 0.009254 +/- 0.000122 | 0.014162 | 402,558 |
| LSTM + dt | 0.009265 +/- 0.000054 | 0.014208 | 402,568 |
| GRU, position only | 0.009491 +/- 0.000258 | 0.014613 | 401,931 |
| SIREN + dt | 0.009556 +/- 0.000135 | 0.014363 | 402,452 |
| Transformer + dt | 0.009804 +/- 0.001014 | 0.014610 | 397,816 |
| WFB, shuffled dt | 0.010996 +/- 0.000174 | 0.015500 | 401,560 |
| Matched FFN + dt | 0.011175 +/- 0.000679 | 0.015273 | 402,452 |
| Matched FFN, position only | 0.011212 +/- 0.000346 | 0.015885 | 400,753 |
| **WFB, real dt** | **0.011230 +/- 0.000223** | **0.015786** | **401,560** |
| Small FFN, position only | 0.012079 +/- 0.000873 | 0.016376 | 142,104 |
| Small FFN + dt | 0.012442 +/- 0.000780 | 0.016379 | 144,152 |
| WFB, constant dt | 0.012488 +/- 0.000726 | 0.016957 | 401,560 |
| Learned Fourier + dt | 0.013047 +/- 0.000174 | 0.017799 | 402,711 |
| Time2Vec + dt | 0.013396 +/- 0.000581 | 0.017710 | 402,415 |
| Random Fourier + dt | 0.014379 +/- 0.000747 | 0.019369 | 402,327 |

The 7.0% gain over the small position-only FFN does not replicate the earlier approximately 20% result: the input features and experimental protocol differ. Real-time WFB is 0.5% worse in mean ADE than matched FFN + dt, with no established difference across seeds. Therefore, these results do not support an architecture-independent advantage from increasing the number of learnable wave parameters.

WFB does outperform three tested periodic/time-encoding alternatives. This is evidence that these particular ways of encoding time are not interchangeable in practice. It is not proof of general superiority to Fourier or sinusoidal representations: **SIREN, which is also feed-forward, has 14.9% lower ADE than real-time WFB**. Its selected frequency setting is 1 and learning rate is `1e-4`.

GRU + dt and LSTM + dt reduce mean ADE relative to WFB by 17.6% and 17.5%. Transformer + dt reduces mean ADE by 12.7%, but has greater seed variation. These comparisons motivate exploring sequence-aware wave models; they do not identify sequence memory as the sole cause of the difference.

## 3. What the Temporal Ablation Shows

<img width="2816" height="1078" alt="02_temporal_ablation" src="https://github.com/user-attachments/assets/a353eb7d-ff15-4d03-9a42-a234ca06e3ec" />

**Figure 2.** Left: the same seed labels across constant, real, and shuffled intervals, with mean +/- SD. Right: paired ADE differences against real-time WFB, with **unadjusted** 95% Student-t intervals. The displayed p-values are Holm-adjusted across all 14 comparisons with real-time WFB. An unadjusted interval excluding zero does not guarantee significance after this correction.

| Comparison | Point estimate | Paired t p-value | Holm-adjusted p-value |
|---|---|---:|---:|
| Real versus constant WFB | Real has 10.1% lower ADE | 0.0081 | 0.0568 |
| Shuffled versus real WFB | Shuffled has 2.1% lower ADE | 0.1489 | 0.4467 |
| Matched FFN + dt versus real WFB | Matched FFN has 0.5% lower ADE | 0.8917 | 1.0000 |

Real WFB beats constant WFB in all five seeds. Shuffled WFB beats real WFB in four of five seeds. The real-versus-constant pattern is suggestive, but neither temporal comparison passes the reported Holm correction at 0.05. Absence of significance is not evidence of equality.

The shuffle permutes the seven noninitial observation intervals within each window. It preserves their values, distribution, and total observed elapsed time, while breaking their assignment to consecutive position differences. The positions and their ordered slots remain unchanged. Constant time removes interval variation, using the training mean; the initial sentinel remains shared across conditions.

The supported interpretation is that **correct local interval assignment has not been established as the reason WFB improves over the small FFN**. The model could use window-level time statistics, generic nonlinear feature expansion, or other information. These are hypotheses, not mechanisms identified by this ablation. Learned nonzero omega values or prediction sensitivity to dt alone would not prove useful physical-time modeling.

### Statistical Scope

Under the paired t-tests and Holm correction, WFB outperforms the tested learned Fourier, random Fourier, and Time2Vec controls; GRU, LSTM, and SIREN outperform WFB. The Transformer comparison does not pass the correction (`p = 0.0907`), despite lower mean ADE.

These tests use five paired seed differences. The exact two-sided sign-flip test has a minimum attainable p-value of 0.0625 with five pairs, so conclusions relying on t-tests should state their small-sample assumptions. Source-bootstrap intervals resample 189 test sources after averaging over seeds. They measure source-sampling uncertainty conditional on the fitted models, not seed-to-seed uncertainty, and should not replace the seed tests merely because they give a more favorable conclusion.

In `paired_ADE.csv`, `mean_difference` is **candidate minus real-time WFB**. Its `relative_improvement_pct` also uses WFB as the reference denominator. Percentages in this README use the explicitly named comparator; a 7.0% WFB improvement over the small FFN is not the same denominator as that CSV's -7.56% candidate improvement relative to WFB.

## 4. Efficiency and Convergence

<img width="2244" height="1320" alt="03_accuracy_latency" src="https://github.com/user-attachments/assets/5583dfb3-3bbf-4c68-afac-f3c63fce5815" />

**Figure 3.** Selected controls illustrate the accuracy/latency trade-off. Lower and farther left is preferable. Error bars show seed SD. Measurements use B200 FP32 forward passes on GPU-resident batches of 512, with warmup, CUDA synchronization, and five trials of at least one second per selected run. They exclude data transfer and preprocessing. Models were scheduled across two devices rather than each being benchmarked on both under a dedicated cross-device protocol.

Real WFB averages **0.232 ms/batch and 2.209 million samples/s**. Matched FFN + dt averages **0.128 ms and 4.007 million samples/s**. Thus WFB has about 81% higher latency and 45% lower throughput, without better mean ADE. SIREN is both more accurate and faster than WFB in this implementation.

WFB is faster than the tested GRU, LSTM, and Transformer, but less accurate. This is a trade-off, not equal-accuracy efficiency. A linear readout, fewer parameters, or simultaneous training of wave parameters does not by itself imply lower runtime. This suite uses an MLP decoder and provides no evidence for linear-decoder WFB or WSP efficiency.

<img width="2904" height="968" alt="04_wfb_convergence" src="https://github.com/user-attachments/assets/c46644fd-b26e-4cd2-a62b-c37a85f0ddd1" />

**Figure 4.** Training MSE, validation MSE, and validation ADE for real-time standard WFB. Loss axes are logarithmic. Each line ends where that run stopped; no missing epochs are filled. Diamonds mark the validation-selected epoch, not the minimum test error.

The selected real-time WFB runs reach their best validation ADE at epochs 58, 85, 45, 60, and 76. Their final test ADE standard deviation is 0.000223. The records show finite metrics and sustained improvement, not an obvious failure to train. They do not prove global convergence or optimal hyperparameters. A ten-epoch comparison would not represent these selected results well.

## 5. From WFB to Sequence-Aware Wave Dynamics

### What Current WFB Does

The evaluated wave layer applies a learned projection to each observed position, then forms a wave activation. Schematically, with elementwise operations,

$$
s_i=\tanh\!\left(\mathrm{LN}(W x_i+b)\right),\qquad
h_i=A\odot\cos\!\left(k\odot s_i-\omega\,\overline{\Delta t}_i-\theta\right),
$$

$$
\widehat{Y}=g\!\left(\mathrm{concat}(h_1,\ldots,h_8)\right).
$$

Here the effective amplitude is the product of the implementation's spatial and temporal amplitudes, and the phase combines its two phase parameters. The bar on dt denotes the training-normalized interval used by the WFB layer. Standard WFB optimizes these parameters through the ordinary chain rule; the optional Laplacian variants alter selected wave-parameter updates.

The wave activation at observation i has no carried wave state from observation i-1. Nevertheless, the downstream FFN receives **ordered slots** and can learn cross-slot relationships. It is incorrect to say that an FFN sees only the current token or is inherently permutation-invariant. The missing property is an explicit, shared elapsed-time state-transition mechanism, not all access to sequence order.

### Why a Propagation Mechanism Is Worth Testing

For the stronger objective of modeling how prior information evolves through successive physical intervals, WFB needs an extension that carries or otherwise relates that information across observations. This could be recurrent, state-space, or attention-based; WSP is one proposed wave-based choice, not the only possible solution.

The motivation has three parts:

1. **A gap in the mechanism.** Current dt changes a local activation; it does not explicitly advance a previous wave state through physical time.
2. **A gap in the temporal evidence.** Correct interval assignment does not outperform shuffled assignment, so useful interval-specific evolution remains unverified.
3. **A gap in application performance.** Sequence models outperform WFB here, making structured temporal interaction a reasonable direction to investigate. SIREN's advantage also shows that representation and optimization deserve attention independently of memory.

### WSP as a Testable Extension

<img width="2904" height="1474" alt="05_wfb_to_wsp" src="https://github.com/user-attachments/assets/ba0c2086-b21a-4b84-b057-ccfbb94c89b2" />

**Figure 5.** The top row describes the evaluated WFB-FFN. The bottom row is a simplified WSP mechanism consistent with the local wave-state implementation. It is a conceptual illustration, not an experimental WSP result or a complete architecture diagram.

A minimal carried-state formulation is

$$
z_i=P_i z_{i-1}+u_i,\qquad
P_i=\mathrm{diag}\left[
e^{-\gamma\Delta t_i}\odot
e^{\mathrm{i}(\kappa\Delta p_i-\omega\Delta t_i)}
\right],\qquad \gamma\geq 0.
$$

The previous hidden wave state is z, the observation-dependent injection is u, and gamma controls damping. The positional increment dp identifies progression in the observation sequence, whereas dt measures physical elapsed time. Kappa is the sequence-position phase increment, distinct from WFB's wavenumber applied to projected spatial features. The cosine and sine state components can implement this complex rotation using real tensors.

Unlike the centered time feature in WFB, propagation time should retain its meaning as a nonnegative duration, optionally expressed in a documented positive reference unit. Centered intervals must not be silently used as negative physical durations in the damping operator.

The distinction becomes explicit by expanding three updates with zero initial state:

$$
z_3=P_3P_2u_1+P_3u_2+u_3.
$$

The contribution from observation 2 depends on the interval after that observation, not only on the sum of all intervals. Swapping intervals of the same total duration can therefore change the hidden state. However, pure fixed diagonal propagation **without intermediate observation injection** composes according to total time and total position increments; its final state need not distinguish interval order. Carrying a state alone does not guarantee useful order sensitivity. Observation injection, the readout, learned frequencies, and possible aliasing all matter.

This produces the WSP research hypothesis:

> Promoting an interval-conditioned wave activation into an observation-driven, elapsed-time hidden-state transition may improve the use of temporally aligned information and prediction across irregular sampling intervals.

The hypothesis is narrower than claiming that WFB failed because it has no memory, that every application requires sequence learning, or that wave propagation necessarily improves accuracy or speed. No WSP checkpoint or WSP accuracy result is included in this report.

## 6. What a WSP Evaluation Should Demonstrate

| Question | Required evidence |
|---|---|
| Does propagation help beyond the wave basis? | Pointwise WFB versus propagated WSP with matched information, readout, training budget, and approximately matched capacity |
| Does physical time help beyond observation index? | Position-only phase, time-conditioned phase, and their combination within the same propagated architecture |
| Does the model learn event-specific elapsed-time effects? | Same initial conditions/content under controlled interval schedules, with physically consistent targets; optional inference-time timing interventions labeled as such |
| Does it generalize beyond observed intervals? | Held-out interval interpolation/extrapolation, irregular sampling, and longer-horizon evaluation |
| Is it competitive with strong alternatives? | Tuned matched SIREN and sequence/time-aware baselines, not only the small FFN |
| Is it more efficient at comparable accuracy? | Accuracy/latency curves over capacity and sequence length; matched precision, batches, timing scope, and device conditions |
| Does a linear readout suffice? | Linear versus nonlinear readouts in otherwise controlled WSP models |

For a state-transition predictor, the requested future duration should be supplied explicitly when it is part of the task definition. Observation intervals and future prediction horizons are different quantities. Their roles should not be conflated when constructing physical-time experiments.

If gradient-rule novelty remains a WFB claim, it still requires a separate fixed-architecture standard/Laplacian/combined comparison, with validation-selected lambda and numerical gradient checks. Adding WSP does not retroactively answer that question.

## 7. Data and Generalization Boundaries

- **Window counts:** 226,669 distinct source/track/start combinations, with two irregular samples per start, yield 453,338 windows. The 425,090 raw-observation count is historical preprocessing metadata, not an independently re-counted raw annotation total in this bundle.
- **Splits:** recorded source/agent pairs do not overlap across splits. However, 177 of 189 test sources also occur in training, accounting for 95.5% of test windows. This is track-disjoint by recorded IDs, not scene-disjoint; original physical identities still need provenance verification.
- **Timing:** all stored observation intervals match differences of float32-rounded frame/2.5 timestamps within 1e-7 seconds. The maximum discrepancy from exact frame-gap/2.5 intervals is 0.003125 seconds. This explains values near, rather than exactly equal to, 0.4 seconds. A zero mismatch against stored timestamps does not verify original camera FPS or recover prior precision.
- **Horizon:** the target is 12 subsequent positions, not a fixed physical duration. Recorded test horizons span approximately 4.80 to 20.40 seconds. Future target intervals are not inputs to the evaluated predictors.
- **Units and protocol:** normalized-image ADE is not comparable directly with meter-based official ETH/UCY benchmark scores. The annotation collection, tracking procedure, FPS provenance, and source overlap must be disclosed before making external benchmark claims.

These issues do not invalidate the internally shared comparison, but they limit physical-time interpretation and generalization claims. Correcting the timing provenance and adding source-disjoint evaluation are important before attributing an improvement to a physical propagation mechanism.

[Wave Function Backpropagation with Explicit Temporal-Interval Dynamics](https://arxiv.org/pdf/2609.00503) The 38th IEEE International Conference on Tools with Artificial Intelligence, Nov. 2026
