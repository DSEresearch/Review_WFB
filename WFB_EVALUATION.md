# WFB evaluation

Run commands below from the project root on the Linux server. Copy the updated
`src/`, `scripts/`, `tests/`, and `requirements.txt` together. 

## What each experiment resolves

| Reviewer concern | Experiment/output |
|---|---|
| Time input differs between FFN and WFB | FFN-x, FFN-(x,dt), and WFB receive identical position windows/targets |
| WFB has more parameters | Capacity-matched FFN-x, FFN-dt, SIREN-style, Fourier, Time2Vec, GRU, LSTM, Transformer |
| Improvement may be generic periodic expansion | Fixed Gaussian Fourier mapping, learned Fourier mapping, SIREN-style network, Time2Vec |
| Correct time is not established | Real, fixed per-window shuffled, and train-mean constant WFB; paired seed statistics |
| Motion features already encode time | Position-only primary experiment; separate motion-feature experiment |
| Weak trajectory baseline | GRU without/with dt, LSTM with dt, Transformer with dt and learned positions |
| Laplacian reliability | Standard/combined/Laplacian branches, lambda=0 in combined, validation lambda sweep, correction/task gradient norms |
| Window counts and inconsistent seconds | Source/track counts, repeated start counts, hashes, timestamp and frame-gap audit |
| Derivative reliability | Double-precision central-difference checks of task/correction gradients; optional noise/missing/jitter tests |
| Nonlinear decoder confound | Original WFB decoder vs linear decoder, trained under the same protocol |
| Which wave terms contribute? | Optional without-A, without-k, without-omega, without-theta suite |

These controls address the current trajectory claim. They do not establish a
universal replacement for neural networks or reproduce native Latent ODE,
ContiFormer, or pretrained time-series foundation models. Those need their own
faithful training protocols if included in a later broader evaluation.

## 1. Dependencies

```bash
python3 -m pip install -r requirements-wfb-review.txt
python3 -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0)); print(torch.cuda.get_arch_list())"
```

Use a PyTorch/CUDA installation that supports the B200. Installing Python
requirements alone does not verify GPU architecture support. The scripts use
FP32, explicitly disable TF32, and report the runtime and device.

## 2. Audit timing and prepare a new cache

For a frame-based dataset, verify that stored frame indices are actual video
frame numbers and that FPS describes those indices. Do not assume that all
ETH/UCY/JAAD sources share one frame rate. For source-specific rates supply a
JSON object such as `{"labels/sequence_a": 25.0, "labels/sequence_b": 30.0}`
using exact source names, and replace `--fps` with `--fps_json source_fps.json`.
The audit cannot establish the physical frame rate from label filenames alone.

The following command reproduces the historical 2.5-FPS assumption and writes
an audit, without changing old data:

```bash
python3 scripts/prepare_wfb_reviewer_data.py \
  --processed_dir outputs/preprocessed_irregular \
  --output_dir outputs/wfb_review_audit \
  --time_basis frames --fps 2.5 --coordinate_unit normalized_image
```

After confirming the time basis, create a separate cache (the example again
uses 2.5 FPS; replace it with verified rates):

```bash
python3 scripts/prepare_wfb_reviewer_data.py \
  --processed_dir outputs/preprocessed_irregular \
  --output_dir outputs/wfb_review_cache \
  --time_basis frames --fps 2.5 --coordinate_unit normalized_image --prepare
```

Use `--time_basis timestamps` for trustworthy original timestamps in seconds.
Already-rounded historical timestamps cannot be repaired by casting back to
float64; reconstruct from verified frame indices/FPS or preprocess raw data.
Training refuses caches with split-track overlap, invalid intervals, invalid
future chronology, missing data, or changed array hashes. A valid cache manifest
is written only after the audit succeeds.

### Confirmed findings in the local historical outputs

The audit in `outputs/wfb_reviewer_audit/data_audit.json` found:

- 322,160 train + 61,794 validation + 69,384 test = 453,338 windows.
- 226,669 distinct starts, with exactly two irregular windows per start.
- All stored dt values for observation positions 1 onward match subtraction
  of absolute timestamps after float32 conversion. This explains values such
  as 0.3984375. The largest discrepancy from frame-gap/2.5 is 0.003125 seconds.
- The first dt previously used a fallback median, which can be 0.6 seconds;
  it does not represent a measured within-window interval. The new protocol
  uses a zero sentinel, excludes it from dt fitting, and never shuffles it.
- The historical 425,090 observations include tracks too short to yield a
  window; raw observations and augmented windows are different counts.
- No shared source-agent pairs, but 199 train/validation, 177 train/test, and
  125 validation/test shared sources. This is not scene-disjoint evaluation.
- Fixed target length does not mean fixed physical horizon: with the 2.5-FPS
  assumption, the test horizon spans 4.8 to 20.4 seconds.
- Duplicate sampled windows exist (72 train, 16 validation, 13 test). They are
  reported and retained for reproduction; they are not independent trials.

New raw preprocessing keeps absolute timestamps in float64 before subtraction
and emits `track_window_counts.csv`, including short tracks, eligible starts,
requested windows, and generated windows. Use the saved audit for the response
to reviewer 1, while describing FPS as an assumption until verified.

If labels contain only bounding boxes/classes, the existing loader links
detections into tracks heuristically. Report this explicitly; class IDs and row
numbers are not ground-truth persistent identities. Prefer original track IDs
for a stronger benchmark. Normalized image-coordinate ADE is not in meters;
do not pool pixel, normalized-image, and world-coordinate data under one unit.

### Stronger split (recommended as a separate experiment)

Regenerate into a new directory using the existing preprocessor's
`--split_unit source`. For example, after verifying the frame-rate assumption:

```bash
python3 scripts/preprocess.py --data_dir eth_ucy_jaad \
  --output_dir outputs/preprocessed_review_scene \
  --fps 2.5 --obs_len 8 --pred_len 12 --irregular_obs \
  --max_obs_skip 3 --irregular_samples 2 --split_unit source
python3 scripts/prepare_wfb_reviewer_data.py \
  --processed_dir outputs/preprocessed_review_scene \
  --output_dir outputs/wfb_review_scene_cache \
  --time_basis frames --fps 2.5 --coordinate_unit normalized_image \
  --require_source_disjoint --prepare
```

Use separate output roots for the historical and scene-disjoint experiments.
Confirm source labels really distinguish scenes (multiple clips from one scene
should be assigned together). Split logic cannot infer physical scene identity.

## 3. Quick pipeline check

```bash
python3 -m unittest tests.test_wfb_reviewer tests.test_proof_of_concept -v
python3 scripts/check_wfb_gradients.py
python3 scripts/run_wfb_reviewer_suite.py \
  --cache_dir outputs/wfb_review_cache --output_dir outputs/wfb_review_pilot \
  --suites core sequence --devices cuda:0 cuda:1 \
  --seeds 1 --epochs 10 --patience 5 --lrs 0.001 --batch_size 512
```

This is a pipeline/pilot run; one seed cannot estimate between-seed variation.
The numerical gradient check validates selected implementation derivatives,
not general approximation accuracy or a new proof of the theory.

## 4. Main reviewer experiments on two B200 GPUs

Preview the exact workload before running:

```bash
python3 scripts/run_wfb_reviewer_suite.py \
  --cache_dir outputs/wfb_review_cache --output_dir outputs/wfb_review_main \
  --suites core sequence laplacian decoder --devices cuda:0 cuda:1 \
  --seeds 1 2 3 4 5 --epochs 100 --dry_run
```

Then run:

```bash
python3 scripts/run_wfb_reviewer_suite.py \
  --cache_dir outputs/wfb_review_cache --output_dir outputs/wfb_review_main \
  --suites core sequence laplacian decoder --devices cuda:0 cuda:1 \
  --seeds 1 2 3 4 5 --epochs 100 --patience 15 --batch_size 512 \
  --lrs 0.0001 0.0003 0.001 --threads 4 --workers 0 \
  --fourier_scales 0.1 1 10 --sine_omegas 1 10 30
```

One process runs on each GPU, distributing model families rather than using
DDP for these small models. Four CPU compute threads per process fit the
20-CPU server. Array caching avoids pandas row conversion on every epoch.
Do not run another GPU workload concurrently with inference timing.

There are 18 jobs and 405 candidate fits with the default five seeds,
three learning rates, and lambda grid. Early stopping can reduce actual epochs.
The expanded periodic grids in the main command increase this to 495 fits;
they give the Fourier/SIREN controls a broader declared tuning budget.
Use `--epochs 50` for all models if the budget is limited; document that common
budget and inspect convergence before reporting final superiority.
Use `--suites core sequence` first if compute time is limited.

Resume with the same command plus `--resume`. Completed candidates are reused;
an interrupted candidate restarts training from its seed. Configuration, source
code, and cache hashes must match; changes require a new output directory.
Each running job has a `.log` file next to `suite_manifest.json`.

Add motion or component ablations in separate roots:

```bash
python3 scripts/run_wfb_reviewer_suite.py \
  --cache_dir outputs/wfb_review_cache --output_dir outputs/wfb_review_motion \
  --suites motion --devices cuda:0 cuda:1 --seeds 1 2 3 4 5 --epochs 100
python3 scripts/run_wfb_reviewer_suite.py \
  --cache_dir outputs/wfb_review_cache --output_dir outputs/wfb_review_wave \
  --suites wave --devices cuda:0 cuda:1 --seeds 1 2 3 4 5 --epochs 100
```

## Selection and fairness

- All candidates share windows, train-only normalization, optimizer, batch size,
  epoch limit, early stopping, and seed lists. Seeded data-loader order does not
  depend on how many parameters were initialized.
- Position-only uses `[x,y]`. Motion uses derivatives recomputed from the
  observed positions and intervals; first unavailable derivatives are zero.
  Finite-difference velocities in existing CSVs are not reused at skipped points.
- Temporal controls change only the explicit dt branch. In motion experiments,
  velocity/acceleration stay fixed, so this tests the *additional* dt signal.
  Such motion features are not train/test leakage, but do already contain timing.
- Flattened FFNs can be sensitive to input order. A failed shuffle contrast is
  not proof that FFNs inherently ignore order; shuffled dt breaks alignment with
  positions while preserving each window's interval distribution.
- Shuffling is deterministic per window and shared across seeds and models;
  it is fixed across train/validation/test passes, not stochastic test-time noise.
- Mean validation ADE across all seeds selects one learning rate per family.
  Only selected candidates are evaluated on test data. Combined WFB includes
  lambda=0; standard WFB is the supervised reference. Lambda sweeps are shown
  on validation, with only the selected setting entering the main test table.
- The existing Laplacian branch replaces gradients only for `At_raw`,
  `omega_raw`, and `theta_t`. Other parameters still use supervised gradients.
  Therefore the name does not mean the entire network trains without labels.
- Parameters are matched by width. Exact counts and residual mismatch are
  recorded; default matched controls are within about 1% of standard WFB.
  Fourier mappings add trigonometric operations not counted as linear MACs.
- Periodic controls are supervised trajectory adaptations, not claims of a
  complete reproduction of the authors' original experiments. Fourier scale
  and SIREN frequency are exposed by both scripts; `--fourier_scales` and
  `--sine_omegas` select them on validation along with learning rate. The saved
  selection is also used during checkpoint reload and robustness testing.
  Disclose these grids and the unequal number of tuning candidates per family.
- Linear WFB has fewer parameters by design. Its comparison measures the
  decoder trade-off; it does not prove capacity-independent superiority.

## 5. Robustness without retraining

```bash
python3 scripts/evaluate_wfb_reviewer_robustness.py \
  --results_dir outputs/wfb_review_main --cache_dir outputs/wfb_review_cache \
  --device cuda:0 --noise_std 0.005 0.01 0.02 \
  --missing_rates 0.1 0.25 0.5 --jitter_std 0.05 0.1 0.25
```

Gaussian noise is measured in raw coordinate units; the defaults suit normalized
image coordinates only. Missing observations use causal carry-forward while
retaining the original timestamps and prediction horizon. This is distinct from
changing the frame-skip rate and target horizon. Timestamp uncertainty is modeled
as positive multiplicative interval noise. Derivatives are recomputed when
observations/timing are corrupted. Future targets are never changed.

## 6. Tables, curves and interpretation

The launcher automatically summarizes successful jobs. Regenerate with:

```bash
python3 scripts/summarize_wfb_reviewer.py --results_dir outputs/wfb_review_main
```

Key files in `summary/`:

- `performance.csv`, `performance_table.tex`: ADE/FDE/MSE/RMSE, seed SD,
  parameter counts and synchronized inference timing.
- `paired_ADE.csv`: paired Student-t confidence intervals and p-values,
  Holm correction, standardized paired effects, exact sign-flip p-values,
  and source-cluster bootstrap intervals.
- `ade_position.pdf/png`: selected test errors across models.
- `lambda_*.pdf/png`: validation ADE vs lambda, including zero for combined.
- `convergence_*.pdf/png`: train MSE, validation MSE, validation ADE per seed.
- Per-candidate histories contain task and weighted-correction gradient norms.
- Per-seed test errors are also grouped by source/track for auditing overlap.

In `paired_ADE.csv`, real-time standard WFB is the reference. A positive
candidate-minus-reference difference favors real WFB. In particular, compare
the shuffled and constant rows with real WFB before claiming a temporal benefit.
Source-cluster bootstrap conditions on the trained models and complements,
rather than replaces, the seed-level intervals. Overlapping windows are never
treated as independent training replicates. Five-seed exact two-sided sign-flip
tests cannot achieve p < 0.05 (minimum 0.0625); report uncertainty instead of
changing the test after looking at results.

Do not claim speed benefits from fewer parameters alone. Latency/throughput
measure resident-input inference at the reported batch size, with synchronization,
30 warmup passes and five duration-based trials. They exclude CPU preprocessing
and host-to-device transfer. Compare accuracy and speed together.

### Baseline sources

- Fourier feature mapping: https://bmild.github.io/fourfeat/
- SIREN reference implementation: https://github.com/vsitzmann/siren
- Time2Vec: https://arxiv.org/abs/1907.05321

