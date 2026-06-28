# AI Unity Notebook

**Started:** 2026-06-27  
**Researcher:** Jim Dawdy  
**Agent:** Codex  
**Machine context:** `/home/jim/Coding Projects/AI_Unity`

This notebook is both a project log and a scientist's working journal. I will write down raw findings, interpretation, doubts, and next decisions as the experiments progress. The goal is to preserve reasoning, not just final scores.

---

## 2026-06-27 00: Initial Handoff Review

### Instructions I Am Following

- Load the necessary skills named in the handoff before continuing experiment work.
- Read and review the files in this folder.
- Implement the staged AI Unity testing plan.
- Test first on MNIST.
- If usage and context remain, begin work toward a larger text dataset, probably `ddclm-baseline-1b` or `fineweb-edu-1b`.
- Log findings carefully and verbosely in this file.

### Skills Loaded

- `pytorch-lightning`: relevant for training structure, GPU training, checkpoints, and future LightningModule conversion (LightningModule means an organized PyTorch model/training class).
- `scikit-learn`: immediately relevant for confusion matrices and evaluation metrics.
- `statsmodels`: held for later multi-seed statistical comparison (formal checks for whether differences are likely meaningful).
- `shap`: held for later model explanation/feature attribution (feature attribution means estimating which inputs influenced predictions).
- `pymoo`: held for later hyperparameter optimization (automated search over settings such as learning rate, expert count, or top-k routing).

### Files Reviewed

I reviewed the project docs, source files, wrappers, and tests:

- `README.md`
- `UNITY_PLANV1.md`
- `UNITY_LOG.md`
- `AI_RESEARCH_UNIFY6-26-26.md`
- `ai unity chat.md`
- `pyproject.toml`
- `requirements.txt`
- `.gitignore`
- `ai_unity/__init__.py`
- `ai_unity/utils.py`
- `ai_unity/data.py`
- `ai_unity/ternary_moe.py`
- `ai_unity/complex_capsules.py`
- `ai_unity/training.py`
- `ai_unity/run_ternary.py`
- `ai_unity/run_complex.py`
- `ternary-moe/model.py`
- `ternary-moe/train.py`
- `ternary-moe/results.txt`
- `complex-capsules/model.py`
- `complex-capsules/train.py`
- `complex-capsules/HANDOVER.md`
- `tests/test_models_smoke.py`
- `tests/test_training_smoke.py`

I intentionally did not read virtual environments, downloaded datasets, checkpoints, or generated result folders as source material. Those are either binary/generated artifacts or dependency trees, not project logic.

### Repository State

`git status --short` reports the entire project as untracked. This matches the handoff: the repo was initialized recently, but no commit exists yet. I will avoid touching generated folders except through experiment outputs under ignored `results/` and `checkpoints/`.

### Scientific Context From The Docs

The core hypothesis is that current neural network signals are too poor: they are scalar-like quantities, while Jim's intuition is that useful signals should carry type, identity, phase, and routing authority. The staged test plan focuses on combinations that are compatible with modern PyTorch training:

1. Ternary weights + mixture of experts (MoE; a router chooses specialist subnetworks).
2. Complex capsules (capsules are vectors whose direction/pose encodes identity; complex values add magnitude and phase).
3. Complex capsule identity as an MoE routing signal.
4. Ternary integration if the earlier phases are promising.

The existing code has already consolidated historical `ternary-moe/` and `complex-capsules/` folders into the shared `ai_unity` package. The current implementation is suitable for GPU smoke tests and full MNIST runs.

### First Code Impressions

The training harness is simple but productive:

- It supports CUDA/CPU device selection.
- It writes comparison JSON, history CSV, confusion matrix JSON, checkpoints, and accuracy plots.
- It records MoE routing diagnostics: expert usage fraction, per-class expert usage, entropy (spread of router choices), collapse rate, and load-balance loss.
- It records complex phase diagnostics: phase mean, phase standard deviation, per-class spread, and collapse detection.

This is enough to run the first real MNIST experiments tonight.

### Important Concern: Option B Is Not Exactly The Plan Yet

The plan describes Complex Capsule Option B as "phase as spatial angle." The current `ComplexCapsuleNetB` uses a learned convolutional phase channel:

```python
phase = self.primary_conv_phase(h)
u = torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))
```

That means phase is learned from features, not explicitly derived from spatial angle. This is still a valid complex capsule experiment, but it is not yet the cleanest possible test of the specific "phase encodes spatial orientation" hypothesis. I should not overstate Option B results as proof of spatial-angle phase unless I later implement an explicit coordinate-derived phase variant.

### MNIST Suitability

Jim's suspicion that MNIST is too small/easy is likely correct. MNIST can verify that the models train and diagnostics are finite, but it may saturate too quickly and fail to reveal whether phase or expert routing helps. The historical handover also says rotated MNIST is the meaningful next gate, because phase should matter more under transformation.

My working standard:

- Plain MNIST: sanity check and regression baseline.
- Rotated MNIST: first scientific stress test for phase.
- DDCLM/FineWeb-Edu: larger-scale text direction, but it requires a different modeling head than the vision capsule code.

### Immediate Plan

1. Add small utilities if needed so results are easier to summarize from JSON.
2. Run the test suite again to verify local state before experiments.
3. Run 5-epoch complex capsule MNIST.
4. Run 5-epoch ternary/MoE MNIST.
5. Log all results and my interpretation here.
6. Run rotated MNIST for complex capsules if the first runs are stable.
7. If time and context remain, inspect and begin a DDCLM/FineWeb-Edu text-data path without pretending it is the same experiment as MNIST vision.

---

## 2026-06-27 01: Environment And Test Verification

### Environment

Verified inside `.venv`:

- `torch 2.11.0+cu130`
- `torchvision 0.26.0+cu130`
- `scikit-learn 1.7.2`
- CUDA available: yes
- GPU: `NVIDIA GeForce RTX 5060 Ti`

### Test Result

Command:

```bash
.venv/bin/python -m pytest
```

Result:

```text
21 passed, 1 skipped in 8.25s
```

### Interpretation

The repo is in a valid starting state. The CUDA path is available, the tests cover forward/backward behavior for the ternary, MoE, real capsule, and complex capsule families, and generated outputs can be written by the training harness. The skipped test is expected when a CUDA-specific check is not relevant on a device path; CUDA itself is available here.

Jim said longer epochs are acceptable because the machine is available overnight. I will treat 20 epochs as the first serious MNIST pass, then use rotated MNIST as the more meaningful stress test.

---

## 2026-06-27 02: 20-Epoch Complex Capsule MNIST Run

### Command

```bash
.venv/bin/python complex-capsules/train.py \
  --epochs 20 \
  --batch-size 128 \
  --device auto \
  --data-dir complex-capsules/data \
  --output-dir results/complex_mnist_20ep_seed123 \
  --checkpoint-dir checkpoints/complex_mnist_20ep_seed123 \
  --num-workers 4 \
  --seed 123
```

### Result Artifacts

- Comparison JSON: `results/complex_mnist_20ep_seed123/complex_capsules_comparison.json`
- Histories: `results/complex_mnist_20ep_seed123/*_history.csv`
- Confusion matrices: `results/complex_mnist_20ep_seed123/*_confusion.json`
- Accuracy plot: `results/complex_mnist_20ep_seed123/complex_capsules_accuracy.png`
- Checkpoints: `checkpoints/complex_mnist_20ep_seed123/`

### Summary

| Rank | Model | Params | Train | Test | Test loss | Time (s) | Phase spread |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | ComplexCapsuleA (full complex) | 2,767,568 | 99.95% | 99.15% | 0.4498 | 207.8 | 1.6366 |
| 2 | RealCapsule | 1,394,320 | 99.84% | 99.03% | 1.4728 | 160.5 | n/a |
| 3 | ComplexCapsuleB (phase=angle) | 2,767,568 | 99.96% | 98.98% | 0.4511 | 211.2 | 1.6908 |
| 4 | BaselineMLP | 269,322 | 99.68% | 97.97% | 0.1050 | 79.5 | n/a |

No complex phase classes were marked collapsed. Mean per-class phase spread:

- ComplexCapsuleA: `1.6366`
- ComplexCapsuleB: `1.6908`

### Interpretation

Plain MNIST is mostly saturated. The capsule family beats the baseline MLP, but the difference among RealCapsule, ComplexCapsuleA, and ComplexCapsuleB is small enough that I should not make a strong architecture claim from this single seed.

ComplexCapsuleA won this run by a small margin at 99.15% test accuracy. RealCapsule reached 99.03%, and ComplexCapsuleB reached 98.98%. That says complex capsules are trainable and competitive here. It does not prove that phase helps.

The most encouraging signal is that phase did not collapse. Both complex variants preserved a broad phase distribution after training. This matters because one failure mode I expected was for the model to effectively become real-valued by pushing phases toward a constant. That did not happen.

The Option B caveat remains important: the code names it `phase=angle`, but it currently learns phase through a convolutional branch rather than explicitly tying phase to spatial coordinates. The run supports "learned phase can train on MNIST," not "spatial angle phase solved rotation."

### Scientist's Note

I am more interested in the negative space around this result than in the leaderboard. MNIST did exactly what Jim feared: it validated the machinery but gave the models too little room to reveal deep differences. The complex models did not break, did not collapse, and did not lag badly despite having complex arithmetic and routing. That is meaningful engineering evidence. But the actual scientific question has moved immediately to rotated MNIST: if phase is doing something geometric, it should matter when geometry is disturbed.

---

## 2026-06-27 03: 20-Epoch Ternary/MoE MNIST Run

### Command

```bash
.venv/bin/python ternary-moe/train.py \
  --epochs 20 \
  --batch-size 128 \
  --device auto \
  --data-dir ternary-moe/data \
  --output-dir results/ternary_mnist_20ep_seed123 \
  --checkpoint-dir checkpoints/ternary_mnist_20ep_seed123 \
  --num-workers 4 \
  --seed 123 \
  --num-experts 4 \
  --top-k 1 \
  --hidden-dim 256
```

### Result Artifacts

- Comparison JSON: `results/ternary_mnist_20ep_seed123/ternary_moe_comparison.json`
- Histories: `results/ternary_mnist_20ep_seed123/*_history.csv`
- Confusion matrices: `results/ternary_mnist_20ep_seed123/*_confusion.json`
- Accuracy plot: `results/ternary_mnist_20ep_seed123/ternary_moe_accuracy.png`
- Checkpoints: `checkpoints/ternary_mnist_20ep_seed123/`

### Summary

| Rank | Model | Params | Train | Test | Test loss | Time (s) | Routing collapse |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | FloatSingle (baseline) | 269,322 | 99.77% | 97.93% | 0.1085 | 79.0 | n/a |
| 2 | FloatMoE (4 experts, top-1) | 817,260 | 99.20% | 97.62% | 0.0842 | 117.1 | 0.3464 |
| 3 | TernarySingle | 269,322 | 94.65% | 93.49% | 5.1021 | 85.3 | n/a |
| 4 | TernaryMoE (4 experts, top-1) | 817,260 | 91.16% | 90.60% | 2.0590 | 168.2 | 0.3712 |

Routing diagnostics:

- FloatMoE usage fraction: `[0.2476, 0.1666, 0.2394, 0.3464]`
- FloatMoE entropy: `1.3853`
- FloatMoE load-balance loss: `0.00409`
- TernaryMoE usage fraction: `[0.2219, 0.2213, 0.1856, 0.3712]`
- TernaryMoE entropy: `1.3850`
- TernaryMoE load-balance loss: `0.00511`

### Interpretation

The longer run does not rescue the TernaryMoE result. FloatSingle beats FloatMoE, and TernarySingle beats TernaryMoE. The specific hypothesis "MoE routing compensates for ternary precision loss" is not supported under this configuration.

This is not simply total expert collapse. The routing collapse rates are only about 0.35-0.37, meaning the dominant expert is used for roughly a third of top-1 assignments rather than nearly all assignments. Entropy is high for four experts. So the router is distributing work, but the distributed work is not producing better classification.

One possible explanation is that top-1 MoE makes each expert see a smaller, moving slice of the data, while ternary quantization already makes each expert harder to optimize. The two burdens may multiply rather than compensate. A more careful curriculum might train FloatMoE first, then ternarize experts after specialization stabilizes. Another possible next setting is top-k=2, because two active experts may soften the brittle routing boundary.

### Scientist's Note

This is a useful failure. It rules out the naive efficiency story: "just add experts and ternary weights will recover." It does not rule out ternary + routing in general, but it says the first simple version is not enough. I would not spend more overnight time brute-forcing this exact top-1 configuration before testing the phase/rotation question, because the complex capsule path is closer to Jim's original "signal carries phase/identity" intuition.

---

## 2026-06-27 04: Rotation Robustness Evaluation From Plain-MNIST Checkpoints

### Implementation Added

Added `ai_unity/evaluate_complex_rotations.py`.

Purpose: load the 20-epoch plain-MNIST complex capsule checkpoints and evaluate them on rotated MNIST test images without retraining. This is a sharper robustness check than training and testing on the same fixed rotation because it asks whether the learned representation generalizes across geometric change.

Validation:

- Full test suite after adding the script: `21 passed, 1 skipped in 8.35s`
- Smoke evaluation on one batch for 0 and 30 degrees succeeded.

### Command

```bash
.venv/bin/python -m ai_unity.evaluate_complex_rotations \
  --comparison-json results/complex_mnist_20ep_seed123/complex_capsules_comparison.json \
  --output-dir results/complex_mnist_20ep_seed123_rotation_eval \
  --data-dir complex-capsules/data \
  --device auto \
  --num-workers 4 \
  --batch-size 256 \
  --rotations 0 15 30 45 60 90
```

### Result Artifact

- `results/complex_mnist_20ep_seed123_rotation_eval/complex_rotation_eval.json`

### Accuracy By Rotation

| Rotation | BaselineMLP | RealCapsule | ComplexCapsuleB | ComplexCapsuleA |
|---:|---:|---:|---:|---:|
| 0 | 97.97% | 99.03% | 98.98% | 99.15% |
| 15 | 94.58% | 97.12% | 97.43% | 97.38% |
| 30 | 77.34% | 84.82% | 85.43% | 86.99% |
| 45 | 47.57% | 57.66% | 58.96% | 61.81% |
| 60 | 26.31% | 33.30% | 35.42% | 36.50% |
| 90 | 10.37% | 15.80% | 18.29% | 16.64% |

Phase standard deviation stayed broad for both complex models across rotations:

- ComplexCapsuleB phase std range: about `1.817` to `1.841`
- ComplexCapsuleA phase std range: about `1.802` to `1.843`

### Interpretation

This is the most promising result so far.

All models degrade under rotation, as expected. But the complex capsule models usually degrade less than RealCapsule, and all capsule models degrade less than the MLP. The largest useful margins:

- At 30 degrees, ComplexCapsuleA beats RealCapsule by `+2.17` percentage points.
- At 45 degrees, ComplexCapsuleA beats RealCapsule by `+4.15` percentage points.
- At 60 degrees, ComplexCapsuleA beats RealCapsule by `+3.20` percentage points.
- At 90 degrees, ComplexCapsuleB beats RealCapsule by `+2.49` percentage points.

This is still single-seed evidence, but it is aligned with the hypothesis that complex phase helps under geometric transformation. The fact that ComplexCapsuleA is best over the middle rotation range is especially interesting because A is the more general full-complex representation. B wins at 15 and 90 degrees but does not dominate consistently.

The MLP collapse is severe: 97.97% at 0 degrees to 10.37% at 90 degrees. RealCapsule also collapses, but more slowly. Complex models degrade more slowly still, especially around 30-60 degrees.

### Caveats

- These checkpoints were trained on unrotated MNIST only. That makes this a useful robustness test, but it does not show how models behave if trained with rotation augmentation.
- Rotations are fixed deterministic rotations, not random per-sample rotations.
- A single seed is not enough for a research claim.
- Complex models have about twice the parameters of RealCapsule (`2.77M` vs `1.39M`), so some robustness could come from capacity rather than phase. A fairer future experiment should parameter-match or include a larger real capsule control.
- ComplexCapsuleB is still not an explicit spatial-angle phase implementation. It learns phase from data.

### Scientist's Note

This is the first result that makes me lean forward. On plain MNIST, complex capsules merely looked viable. Under rotation, the complex variants begin to act like they have a useful extra degree of freedom. I do not want to romanticize this: it could be parameter count, architecture details, or seed noise. But the direction is exactly where the theory said to look. If I were designing the next controlled study, I would do three things: add seeds, parameter-match RealCapsule, and implement a true coordinate-phase Option B so we can separate "learned complex capacity" from "phase as geometric angle."

---

## 2026-06-27 05: Added True Spatial-Angle Phase Variant

### Implementation

Added `ComplexCapsuleNetSpatialB` in `ai_unity/complex_capsules.py`.

This model differs from the existing `ComplexCapsuleNetB`:

- Existing B learns phase with a convolutional branch.
- New SpatialB anchors primary capsule phase to the spatial angle of the capsule grid position.
- SpatialB keeps a learned per-capsule/per-dimension phase bias, so the model can adjust the coordinate phase without abandoning the spatial prior.

Because the primary capsule grid is currently only `3x3`, the spatial-angle signal is coarse. That is a limitation, but it makes the implementation directly test the "phase as spatial angle" hypothesis from `UNITY_PLANV1.md`.

### Files Updated

- `ai_unity/complex_capsules.py`
- `ai_unity/training.py`
- `ai_unity/__init__.py`
- `ai_unity/evaluate_complex_rotations.py`
- `tests/test_models_smoke.py`

### Validation

Command:

```bash
.venv/bin/python -m pytest
```

Result:

```text
23 passed, 1 skipped in 8.39s
```

### Interpretation

This closes an important conceptual gap in the experimental suite. The next complex run will now include:

1. BaselineMLP
2. RealCapsule
3. ComplexCapsuleB with learned phase
4. ComplexCapsuleSpatialB with coordinate-anchored phase
5. ComplexCapsuleA with full complex channels

This makes the phrase "Option B" less ambiguous: we can now distinguish learned phase from explicit spatial phase.

---

## 2026-06-27 06: Five-Way Complex MNIST Run With SpatialB

### Command

```bash
.venv/bin/python complex-capsules/train.py \
  --epochs 20 \
  --batch-size 128 \
  --device auto \
  --data-dir complex-capsules/data \
  --output-dir results/complex_mnist_spatial_20ep_seed123 \
  --checkpoint-dir checkpoints/complex_mnist_spatial_20ep_seed123 \
  --num-workers 4 \
  --seed 123
```

### Plain MNIST Summary

| Rank | Model | Params | Train | Test | Test loss | Time (s) | Phase spread |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | ComplexCapsuleA (full complex) | 2,767,568 | 99.94% | 99.12% | 0.4523 | 224.7 | 1.7358 |
| 2 | ComplexCapsuleB (phase=angle) | 2,767,568 | 99.98% | 99.04% | 0.4491 | 233.3 | 1.7203 |
| 3 | RealCapsule | 1,394,320 | 99.84% | 98.99% | 1.4732 | 164.3 | n/a |
| 4 | ComplexCapsuleSpatialB (phase=spatial_angle) | 1,440,464 | 99.92% | 98.96% | 0.4595 | 225.7 | 1.5403 |
| 5 | BaselineMLP | 269,322 | 99.74% | 97.97% | 0.1073 | 78.3 | n/a |

Result artifacts:

- `results/complex_mnist_spatial_20ep_seed123/complex_capsules_comparison.json`
- `checkpoints/complex_mnist_spatial_20ep_seed123/`

### Interpretation

The five-way plain MNIST run again clusters tightly among capsule models. ComplexCapsuleA is best, learned-phase B is second, RealCapsule third, and SpatialB fourth. SpatialB is important because it has about the same parameter count as RealCapsule (`1.44M` vs `1.39M`), unlike A/B learned complex models (`2.77M`). On plain MNIST, SpatialB is viable but not superior.

No complex model showed phase collapse.

---

## 2026-06-27 07: Rotation Robustness With SpatialB Included

### Command

```bash
.venv/bin/python -m ai_unity.evaluate_complex_rotations \
  --comparison-json results/complex_mnist_spatial_20ep_seed123/complex_capsules_comparison.json \
  --output-dir results/complex_mnist_spatial_20ep_seed123_rotation_eval \
  --data-dir complex-capsules/data \
  --device auto \
  --num-workers 4 \
  --batch-size 256 \
  --rotations 0 15 30 45 60 90
```

### Accuracy By Rotation

| Rotation | BaselineMLP | RealCapsule | Learned B | SpatialB | Full Complex A |
|---:|---:|---:|---:|---:|---:|
| 0 | 97.97% | 98.99% | 99.04% | 98.96% | 99.12% |
| 15 | 94.41% | 97.20% | 97.42% | 96.78% | 97.42% |
| 30 | 77.21% | 85.86% | 87.01% | 84.97% | 86.44% |
| 45 | 48.78% | 60.48% | 63.40% | 58.38% | 59.95% |
| 60 | 26.87% | 36.73% | 37.97% | 33.19% | 36.61% |
| 90 | 10.19% | 16.97% | 18.76% | 17.61% | 18.49% |

Result artifact:

- `results/complex_mnist_spatial_20ep_seed123_rotation_eval/complex_rotation_eval.json`

### Interpretation

Learned-phase ComplexCapsuleB is the strongest model on this rotation curve. It beats RealCapsule at every nonzero rotation:

- 15 degrees: `+0.22` points
- 30 degrees: `+1.15` points
- 45 degrees: `+2.92` points
- 60 degrees: `+1.24` points
- 90 degrees: `+1.79` points

Full Complex A is competitive but not consistently best in this run. SpatialB does not validate my first explicit spatial-angle prior. It trails RealCapsule at 15, 30, 45, and 60 degrees, and only beats RealCapsule at 90 degrees. Since SpatialB is parameter-matched close to RealCapsule, this is an important caution: simply injecting coordinate angle as phase is not automatically helpful.

My read: learned phase is more flexible and currently more useful than hard-coded coarse spatial phase. The spatial prior may be too crude because the primary capsule grid is only `3x3`; it may also be the wrong place to impose angle. A richer spatial phase model might need finer capsule grids, radial information, or rotation augmentation during training.

### Scientist's Note

This result trims the hypothesis. "Phase helps" still looks plausible. "Hand-authored spatial angle phase helps immediately" does not. I like this outcome scientifically because it separates romantic theory from implementation. The network seems to want phase freedom more than my crude coordinate badge. That does not kill the idea of phase as geometry; it says geometry has to be represented at the right resolution and with the right inductive bias.

---

## 2026-06-27 08: Began DDCLM/FineWeb-Edu Text Dataset Path

### Context

Jim suspected MNIST is too small and suggested `ddclm-baseline-1b` or `fineweb-edu-1b` as better datasets. I agree in principle: a 1B-token text corpus would be a much richer environment than MNIST. But it is also not a drop-in replacement for the current vision capsule code. It requires a language-modeling harness (tokenization, sequence batching, next-token loss, and likely transformer or sequence models).

### Local Text Stack

Installed in `.venv` / system site packages:

- `datasets 4.8.5`
- `transformers 5.10.2`
- `tokenizers 0.22.2`
- `tiktoken 0.12.0`

I added a text optional dependency group in `pyproject.toml`:

```toml
[project.optional-dependencies]
text = ["datasets>=4.0", "transformers>=5.0", "tokenizers>=0.22", "tiktoken>=0.12"]
```

### Implementation Added

Added `ai_unity/text_data.py`.

Capabilities:

- Dataset aliases:
  - `dclm-1b`, `ddclm-1b`, `dclm-baseline-1b` -> `codelion/dclm-baseline-1B`
  - `fineweb-edu-1b` -> `codelion/fineweb-edu-1B`
- Streaming row iteration through Hugging Face `datasets`.
- Inspection CLI that writes:
  - `text_dataset_inspection.json`
  - `samples.txt`

### Verification Commands

```bash
.venv/bin/python -m ai_unity.text_data \
  --dataset dclm-baseline-1b \
  --max-docs 3 \
  --output-dir results/text_dataset_inspection_dclm_1b

.venv/bin/python -m ai_unity.text_data \
  --dataset fineweb-edu-1b \
  --max-docs 3 \
  --output-dir results/text_dataset_inspection_fineweb_edu_1b
```

Artifacts written:

- `results/text_dataset_inspection_dclm_1b/text_dataset_inspection.json`
- `results/text_dataset_inspection_dclm_1b/samples.txt`
- `results/text_dataset_inspection_fineweb_edu_1b/text_dataset_inspection.json`
- `results/text_dataset_inspection_fineweb_edu_1b/samples.txt`

Sample observations:

- `codelion/dclm-baseline-1B` rows have key `text`; first sampled lengths were `2481`, `1606`, and `2287` characters.
- `codelion/fineweb-edu-1B` rows have key `text`; first sampled lengths were `920`, `949`, and `4359` characters.

### Important Stability Caveat

The inspection scripts successfully wrote output, but both Python processes ended with:

```text
terminate called without an active exception
```

and exit code `134`.

This appears to happen during native cleanup after streaming rows from `datasets`, not before data access. I should not launch a long 1B-token streaming/training job until this is resolved or isolated. A next engineering step would be to reproduce the abort in a minimal script, try non-streaming small slices, try disabling HF transfer/Xet paths, or isolate data streaming in a subprocess that writes shards and exits cleanly.

### Scientist's Note

Starting the text path is the right move, but I do not want to pretend it is just "bigger MNIST." MNIST tests phase/capsule geometry directly. DDCLM/FineWeb-Edu would test language modeling and routing/capacity under scale. That may be a better stress test for MoE and ternary ideas than for the current capsule geometry implementation. The honest next milestone is a small text harness: token stream -> fixed-length batches -> baseline tiny transformer -> MoE/ternary variants. Only after that should the 1B-token corpora become more than data plumbing.

---

## 2026-06-27 09: Second Seed For Complex/Spatial MNIST

### Command

```bash
.venv/bin/python complex-capsules/train.py \
  --epochs 20 \
  --batch-size 128 \
  --device auto \
  --data-dir complex-capsules/data \
  --output-dir results/complex_mnist_spatial_20ep_seed321 \
  --checkpoint-dir checkpoints/complex_mnist_spatial_20ep_seed321 \
  --num-workers 4 \
  --seed 321
```

### Plain MNIST Summary

| Rank | Model | Params | Train | Test | Test loss | Time (s) | Phase spread |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | ComplexCapsuleSpatialB (phase=spatial_angle) | 1,440,464 | 99.92% | 99.13% | 0.4524 | 223.6 | 1.7041 |
| 2 | ComplexCapsuleB (phase=angle) | 2,767,568 | 99.98% | 99.06% | 0.4485 | 229.4 | 1.6819 |
| 3 | ComplexCapsuleA (full complex) | 2,767,568 | 99.94% | 98.92% | 0.4564 | 221.0 | 1.6667 |
| 4 | RealCapsule | 1,394,320 | 99.81% | 98.83% | 1.4758 | 170.6 | n/a |
| 5 | BaselineMLP | 269,322 | 99.66% | 98.14% | 0.0912 | 76.7 | n/a |

Artifacts:

- `results/complex_mnist_spatial_20ep_seed321/complex_capsules_comparison.json`
- `checkpoints/complex_mnist_spatial_20ep_seed321/`

### Rotation Evaluation

Command:

```bash
.venv/bin/python -m ai_unity.evaluate_complex_rotations \
  --comparison-json results/complex_mnist_spatial_20ep_seed321/complex_capsules_comparison.json \
  --output-dir results/complex_mnist_spatial_20ep_seed321_rotation_eval \
  --data-dir complex-capsules/data \
  --device auto \
  --num-workers 4 \
  --batch-size 256 \
  --rotations 0 15 30 45 60 90
```

Seed 321 accuracy by rotation:

| Rotation | BaselineMLP | RealCapsule | Learned B | SpatialB | Full Complex A |
|---:|---:|---:|---:|---:|---:|
| 0 | 98.14% | 98.83% | 99.06% | 99.13% | 98.92% |
| 15 | 94.28% | 97.09% | 97.69% | 97.42% | 97.44% |
| 30 | 77.13% | 85.60% | 87.74% | 85.20% | 86.55% |
| 45 | 49.02% | 60.87% | 62.38% | 58.51% | 61.34% |
| 60 | 27.44% | 35.01% | 37.18% | 33.42% | 36.38% |
| 90 | 12.05% | 18.29% | 18.30% | 15.30% | 15.94% |

### Two-Seed Rotation Aggregate

Seeds included: `123`, `321`.

| Rotation | BaselineMLP | RealCapsule | Learned B | SpatialB | Full Complex A |
|---:|---:|---:|---:|---:|---:|
| 0 | 98.06% | 98.91% | 99.05% | 99.05% | 99.02% |
| 15 | 94.34% | 97.14% | 97.55% | 97.10% | 97.43% |
| 30 | 77.17% | 85.73% | 87.38% | 85.08% | 86.50% |
| 45 | 48.90% | 60.68% | 62.89% | 58.44% | 60.64% |
| 60 | 27.15% | 35.87% | 37.58% | 33.30% | 36.49% |
| 90 | 11.12% | 17.63% | 18.53% | 16.46% | 17.21% |

Mean nonzero-rotation advantage versus RealCapsule:

- BaselineMLP: `-7.67` percentage points
- ComplexCapsuleB learned phase: `+1.37` percentage points
- ComplexCapsuleSpatialB spatial angle: `-1.33` percentage points
- ComplexCapsuleA full complex: `+0.25` percentage points

### Interpretation

The second seed strengthens the main pattern:

- Learned-phase ComplexCapsuleB is consistently better than RealCapsule under rotation on average.
- SpatialB is not helping under rotation in this implementation, despite matching RealCapsule parameter count closely.
- Full Complex A is mildly positive on average but less consistent than learned B.
- Plain MNIST ranking changes by tiny amounts across seeds, so plain MNIST remains a weak decision surface.

Learned B's advantage is not huge, but it is now present across two seeds and most nonzero rotations. This is the strongest empirical signal from the night.

### Scientist's Note

I am starting to trust the learned-phase result as a real lead, not a fluke. I still would not call it a claim. Two seeds is not enough, and capacity remains a confound. But a pattern is forming: let the model learn phase, and it uses it in a way that gives modest rotation robustness. Hard-code a crude spatial phase on a `3x3` capsule grid, and it does not help. The theory should adapt to that evidence.

---

## 2026-06-27 10: Third Seed And Three-Seed Aggregate

### Seed 777 Command

```bash
.venv/bin/python complex-capsules/train.py \
  --epochs 20 \
  --batch-size 128 \
  --device auto \
  --data-dir complex-capsules/data \
  --output-dir results/complex_mnist_spatial_20ep_seed777 \
  --checkpoint-dir checkpoints/complex_mnist_spatial_20ep_seed777 \
  --num-workers 4 \
  --seed 777
```

### Seed 777 Plain MNIST Summary

| Rank | Model | Params | Train | Test | Test loss | Time (s) | Phase spread |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | ComplexCapsuleB (phase=angle) | 2,767,568 | 99.96% | 99.19% | 0.4501 | 228.6 | 1.6473 |
| 2 | ComplexCapsuleA (full complex) | 2,767,568 | 99.95% | 99.12% | 0.4508 | 224.9 | 1.7386 |
| 3 | ComplexCapsuleSpatialB (phase=spatial_angle) | 1,440,464 | 99.91% | 98.94% | 0.4556 | 222.0 | 1.6687 |
| 4 | RealCapsule | 1,394,320 | 99.83% | 98.93% | 1.4750 | 172.2 | n/a |
| 5 | BaselineMLP | 269,322 | 99.72% | 97.99% | 0.1004 | 78.1 | n/a |

### Seed 777 Rotation Summary

| Rotation | BaselineMLP | RealCapsule | Learned B | SpatialB | Full Complex A |
|---:|---:|---:|---:|---:|---:|
| 0 | 97.99% | 98.93% | 99.19% | 98.94% | 99.12% |
| 15 | 94.33% | 96.96% | 97.65% | 96.61% | 97.29% |
| 30 | 77.56% | 84.14% | 87.50% | 83.13% | 85.89% |
| 45 | 47.55% | 58.22% | 62.39% | 55.36% | 59.48% |
| 60 | 24.90% | 34.58% | 35.19% | 31.22% | 34.12% |
| 90 | 9.34% | 16.98% | 14.96% | 15.32% | 18.36% |

Artifacts:

- `results/complex_mnist_spatial_20ep_seed777/complex_capsules_comparison.json`
- `results/complex_mnist_spatial_20ep_seed777_rotation_eval/complex_rotation_eval.json`
- `checkpoints/complex_mnist_spatial_20ep_seed777/`

### Three-Seed Rotation Mean Accuracy

Seeds included: `123`, `321`, `777`.

| Rotation | BaselineMLP | RealCapsule | Learned B | SpatialB | Full Complex A |
|---:|---:|---:|---:|---:|---:|
| 0 | 98.03% | 98.92% | 99.10% | 99.01% | 99.05% |
| 15 | 94.34% | 97.08% | 97.59% | 96.94% | 97.38% |
| 30 | 77.30% | 85.20% | 87.42% | 84.43% | 86.29% |
| 45 | 48.45% | 59.86% | 62.72% | 57.42% | 60.26% |
| 60 | 26.40% | 35.44% | 36.78% | 32.61% | 35.70% |
| 90 | 10.53% | 17.41% | 17.34% | 16.08% | 17.60% |

Three-seed nonzero-rotation mean advantage versus RealCapsule:

- BaselineMLP: `-7.59 +/- 2.94` percentage points
- ComplexCapsuleB learned phase: `+1.37 +/- 1.46` percentage points
- ComplexCapsuleSpatialB spatial angle: `-1.50 +/- 1.29` percentage points
- ComplexCapsuleA full complex: `+0.45 +/- 1.02` percentage points

Plain MNIST final test accuracy mean:

- BaselineMLP: `98.03% +/- 0.08`
- RealCapsule: `98.92% +/- 0.07`
- ComplexCapsuleB learned phase: `99.10% +/- 0.07`
- ComplexCapsuleSpatialB spatial angle: `99.01% +/- 0.09`
- ComplexCapsuleA full complex: `99.05% +/- 0.09`

### Interpretation

With three seeds, the learned-phase ComplexCapsuleB result is the strongest current evidence. It has the best mean plain-MNIST accuracy and the best mean nonzero-rotation advantage versus RealCapsule.

The result is modest, not dramatic. The standard deviation of the rotation advantage is similar in size to the mean advantage. But the direction is stable enough to justify a next controlled phase:

1. Add a parameter-matched larger RealCapsule control.
2. Add at least two more seeds.
3. Evaluate rotation augmentation and random rotations, not only fixed deterministic rotations.
4. Rework SpatialB if the explicit spatial prior remains scientifically interesting; the current `3x3` grid is probably too coarse.

### Scientist's Note

This is now a real lead: learned phase seems to buy rotation robustness. The evidence is still early, but it is no longer just a single lucky run. SpatialB is a humbling counterexample to my own intuition; explicit coordinate phase sounded elegant, but the model did worse with it. The next truth test is whether learned B still wins against a capacity-matched real model. If it does, then phase is not just more parameters; it is doing representational work.

---

## 2026-06-27 11: Ternary/MoE Top-k=2 Follow-Up

### Why This Run

The top-k=1 ternary/MoE run failed: TernaryMoE was worse than TernarySingle. I suspected the routing boundary might be too brittle when each input is forced through only one ternary expert. This run tests top-k=2, where each input can use two experts.

### Command

```bash
.venv/bin/python ternary-moe/train.py \
  --epochs 20 \
  --batch-size 128 \
  --device auto \
  --data-dir ternary-moe/data \
  --output-dir results/ternary_mnist_20ep_seed123_topk2 \
  --checkpoint-dir checkpoints/ternary_mnist_20ep_seed123_topk2 \
  --num-workers 4 \
  --seed 123 \
  --num-experts 4 \
  --top-k 2 \
  --hidden-dim 256
```

### Summary

| Rank | Model | Params | Train | Test | Test loss | Time (s) | Routing collapse |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | FloatSingle (baseline) | 269,322 | 99.79% | 98.14% | 0.1011 | 79.7 | n/a |
| 2 | FloatMoE (4 experts, top-2) | 817,260 | 99.94% | 98.04% | 0.0905 | 123.4 | 0.4584 |
| 3 | TernaryMoE (4 experts, top-2) | 817,260 | 96.92% | 95.34% | 0.6892 | 178.2 | 0.5000 |
| 4 | TernarySingle | 269,322 | 95.35% | 93.93% | 5.1592 | 86.5 | n/a |

Routing diagnostics:

- FloatMoE top-2 usage fraction: `[0.4584, 0.0470, 0.3431, 0.1515]`
- FloatMoE entropy: `0.6262`
- FloatMoE load-balance loss: `0.02575`
- TernaryMoE top-2 usage fraction: `[0.0005, 0.00005, 0.5000, 0.49945]`
- TernaryMoE entropy: `0.6783`
- TernaryMoE load-balance loss: `0.06236`

### Interpretation

This changes the ternary/MoE story.

Top-k=1 did not help ternary. Top-k=2 did help on this seed:

- TernarySingle: `93.93%`
- TernaryMoE top-2: `95.34%`
- Improvement: `+1.41` percentage points

This still trails FloatSingle by `2.80` points, but it recovers meaningful accuracy compared with a single ternary network.

The caveat is expert usage. TernaryMoE top-2 effectively collapsed to two experts: experts 2 and 3 each got about half the selected routes, while experts 0 and 1 were almost unused. So the result is not "four balanced experts specialize cleanly." It is more like "letting the ternary model blend two active expert subnetworks works better than a single ternary path."

### Scientist's Note

This is exactly why long overnight runs help. The earlier failure was real, but it was not the whole story. The architecture was brittle with top-1 routing; with top-2, the combination becomes useful. I would next test a two-expert top-2 version, an 8-expert top-2 version with stronger load balancing, and a curriculum where FloatMoE specializes first and then experts are ternarized.

---

## 2026-06-27 12: FineWeb-Edu Tiny Transformer Baseline Started

### Research Decision

Jim asked whether ternary/MoE is worth chasing or whether the project should focus on learned-phase complex capsules.

My current answer:

- Main theory track: learned-phase complex capsules. This has the cleanest empirical signal so far and best matches the "signal carries phase/identity" thesis.
- Secondary engineering track: ternary/MoE. It is not a dead end because top-k=2 recovered meaningful accuracy, but it is more of an efficiency/routing engineering problem than the core representation thesis.
- FineWeb path: start with standard language modeling baselines. Capsules do not yet transfer cleanly to text, so the honest first FineWeb step is a tiny transformer baseline, then MoE/ternary variants.

### Implementation Added

Added `ai_unity/text_lm.py`.

Capabilities:

- `prepare`: stream FineWeb-Edu text and write a local token cache.
- `train`: train a tiny causal transformer from scratch on cached tokens.
- Uses `tiktoken` GPT-2 encoding.
- Uses a subprocess worker plus `os._exit(0)` after cache writing to avoid the native cleanup abort seen in Hugging Face `datasets` streaming.

Added test:

- `tests/test_text_lm_smoke.py`

Also fixed:

- `ai_unity/utils.py` now serializes `Path` objects in JSON output.
- Tiny transformer weights now initialize with small normal weights (`std=0.02`) instead of default embedding initialization, which made the first attempted loss enormous.

### Validation

Command:

```bash
.venv/bin/python -m pytest
```

Result:

```text
24 passed, 1 skipped, 1 warning in 8.86s
```

The warning is a PyTorch nested-tensor warning caused by `norm_first=True` in `TransformerEncoderLayer`; it does not block training.

### FineWeb-Edu Token Cache

Command:

```bash
.venv/bin/python -m ai_unity.text_lm prepare \
  --dataset fineweb-edu-1b \
  --max-tokens 250000 \
  --output-dir results/fineweb_edu_1b_tokens_250k
```

Result:

```text
Wrote 250,000 tokens from 233 docs to results/fineweb_edu_1b_tokens_250k/tokens.npy
Prepared token cache under results/fineweb_edu_1b_tokens_250k
```

### Tiny Transformer Smoke Training

Command:

```bash
.venv/bin/python -m ai_unity.text_lm train \
  --token-path results/fineweb_edu_1b_tokens_250k/tokens.npy \
  --output-dir results/fineweb_edu_1b_tiny_transformer_smoke \
  --device auto \
  --seed 123 \
  --steps 100 \
  --batch-size 32 \
  --seq-len 128 \
  --d-model 192 \
  --n-heads 6 \
  --n-layers 4 \
  --amp \
  --log-every 20 \
  --eval-batches 8
```

Model:

- Parameters: `11,453,760`
- Token cache: `250,000` tokens
- Device: CUDA on RTX 5060 Ti

Training trace:

| Step | Train loss | Eval loss | Eval perplexity | Token accuracy |
|---:|---:|---:|---:|---:|
| 1 | 10.8503 | 10.6342 | 41,529.62 | 3.94% |
| 20 | 9.5051 | 9.4282 | 12,434.07 | 3.70% |
| 40 | 8.5193 | 8.4475 | 4,663.42 | 3.92% |
| 60 | 7.9235 | 7.8185 | 2,486.20 | 4.09% |
| 80 | 7.5251 | 7.5411 | 1,883.89 | 7.16% |
| 100 | 7.4429 | 7.4838 | 1,779.05 | 7.51% |

Artifacts:

- `results/fineweb_edu_1b_tokens_250k/tokens.npy`
- `results/fineweb_edu_1b_tokens_250k/metadata.json`
- `results/fineweb_edu_1b_tiny_transformer_smoke/tiny_transformer_lm_results.json`
- `results/fineweb_edu_1b_tiny_transformer_smoke/tiny_transformer_lm.pt`

### Interpretation

This is not a meaningful language result yet; it is a working baseline path. The loss decreases cleanly, the save path works, and the data cache avoids the earlier streaming cleanup crash.

The next FineWeb step should be:

1. Prepare a larger local token cache, probably 5M-20M tokens.
2. Train the tiny transformer longer with train/validation split separation.
3. Add a tiny MoE feed-forward block variant.
4. Only then add ternary experts or phase-inspired routing.

### Scientist's Note

FineWeb changes the project from geometric representation learning to language modeling. I should resist importing the capsule idea too quickly. The right scientific move is to build the standard baseline first, then perturb one thing at a time. If ternary/MoE helps language, it will likely help as an efficiency/routing mechanism. If learned phase helps language, it probably needs to enter through positional/attention-style mechanisms rather than MNIST-style capsules.

---

## 2026-06-27 13: FineWeb-Edu 5M Token Baseline vs Tiny MoE

### Implementation Update

Expanded `ai_unity/text_lm.py`:

- Replaced the basic `TransformerEncoderLayer` path with explicit transformer blocks.
- Added `TokenMoEFeedForward`, a token-level MoE feed-forward block.
- Added `--model-kind transformer|moe`.
- Added `--num-experts`, `--top-k`, and `--lb-weight`.
- Added routing summaries for MoE language models.

Expanded tests:

- `tests/test_text_lm_smoke.py` now covers both standard and MoE language-model forward/backward paths.

Validation:

```text
25 passed, 1 skipped in 9.03s
```

### FineWeb-Edu 5M Token Cache

Command:

```bash
.venv/bin/python -m ai_unity.text_lm prepare \
  --dataset fineweb-edu-1b \
  --max-tokens 5000000 \
  --output-dir results/fineweb_edu_1b_tokens_5m
```

Result:

```text
Wrote 5,000,000 tokens from 4,539 docs to results/fineweb_edu_1b_tokens_5m/tokens.npy
```

Token cache size:

- `results/fineweb_edu_1b_tokens_5m/tokens.npy`: about `9.6M`

### Tiny Transformer Baseline

Command:

```bash
.venv/bin/python -m ai_unity.text_lm train \
  --token-path results/fineweb_edu_1b_tokens_5m/tokens.npy \
  --output-dir results/fineweb_edu_1b_tiny_transformer_5m_500s \
  --device auto \
  --seed 123 \
  --steps 500 \
  --batch-size 32 \
  --seq-len 128 \
  --d-model 192 \
  --n-heads 6 \
  --n-layers 4 \
  --amp \
  --log-every 50 \
  --eval-batches 16
```

Result:

- Parameters: `11,453,760`
- Runtime to final log: `26.5s`
- Final train loss: `6.7246`
- Final eval loss: `6.7317`
- Final eval perplexity: `838.53`
- Final token accuracy: `12.81%`

### Tiny MoE Transformer

Command:

```bash
.venv/bin/python -m ai_unity.text_lm train \
  --token-path results/fineweb_edu_1b_tokens_5m/tokens.npy \
  --output-dir results/fineweb_edu_1b_tiny_moe_5m_500s \
  --device auto \
  --seed 123 \
  --steps 500 \
  --batch-size 32 \
  --seq-len 128 \
  --d-model 192 \
  --n-heads 6 \
  --n-layers 4 \
  --model-kind moe \
  --num-experts 4 \
  --top-k 2 \
  --lb-weight 0.01 \
  --amp \
  --log-every 50 \
  --eval-batches 16
```

Result:

- Parameters: `15,007,312`
- Runtime to final log: `40.6s`
- Final train loss: `6.6601`
- Final eval loss: `6.6688`
- Final eval perplexity: `787.48`
- Final token accuracy: `13.28%`

### Comparison

| Model | Params | Eval loss | Perplexity | Token acc | Runtime |
|---|---:|---:|---:|---:|---:|
| Tiny Transformer | 11.45M | 6.7317 | 838.53 | 12.81% | 26.5s |
| Tiny MoE Transformer | 15.01M | 6.6688 | 787.48 | 13.28% | 40.6s |

The MoE model improved eval loss by `0.0628`, improved perplexity by about `51`, and improved token accuracy by `0.47` percentage points. It also used more parameters and took about `53%` longer.

### MoE Routing

Routing stayed reasonably distributed:

| Layer | Usage fraction | Entropy | Collapse rate |
|---:|---|---:|---:|
| 0 | `[0.2313, 0.2875, 0.2413, 0.2399]` | 0.8917 | 0.2875 |
| 1 | `[0.1835, 0.3302, 0.2645, 0.2218]` | 1.0317 | 0.3302 |
| 2 | `[0.2668, 0.3835, 0.1599, 0.1897]` | 1.1400 | 0.3835 |
| 3 | `[0.1940, 0.3070, 0.2399, 0.2592]` | 1.2417 | 0.3070 |

This is much healthier than the ternary top-k=2 MNIST routing, where only two experts were effectively used.

### Interpretation

The FineWeb path is working. On a small 5M-token cache and 500 training steps, the MoE feed-forward variant beats the same-size-shape transformer baseline modestly, but with more parameters and slower training.

This does not yet prove MoE is "worth it" in a strong sense, because the comparison is not parameter-matched. It does say the MoE path is viable on FineWeb and does not immediately collapse.

### Scientist's Note

This is the right kind of early language experiment: boring, controlled, and measurable. The MoE result is positive but not magical. I would not bring ternary into FineWeb until this baseline vs MoE comparison is repeated with parameter matching or equal compute. Learned-phase remains the more original thesis, but FineWeb is likely where MoE can earn its keep.

---

## 2026-06-27 14: FineWeb-Edu Parameter-Matched Dense Control

### Why This Run

The first FineWeb-Edu MoE run beat the `11.45M` dense transformer, but the MoE model had `15.01M` parameters. This run tests whether the gain was routing or just parameter count.

### Command

```bash
.venv/bin/python -m ai_unity.text_lm train \
  --token-path results/fineweb_edu_1b_tokens_5m/tokens.npy \
  --output-dir results/fineweb_edu_1b_dense_15m_5m_500s \
  --device auto \
  --seed 123 \
  --steps 500 \
  --batch-size 32 \
  --seq-len 128 \
  --d-model 240 \
  --n-heads 6 \
  --n-layers 4 \
  --amp \
  --log-every 50 \
  --eval-batches 16
```

### Result

| Run | Params | Eval loss | Perplexity | Token acc | Seconds |
|---|---:|---:|---:|---:|---:|
| 11.45M dense | 11,453,760 | 6.7317 | 838.53 | 12.81% | 26.5 |
| 14.87M dense | 14,870,160 | 6.6496 | 772.46 | 13.17% | 29.8 |
| 15.01M MoE | 15,007,312 | 6.6688 | 787.48 | 13.28% | 40.6 |

Validation after the run:

```text
25 passed, 1 skipped in 8.66s
```

### Interpretation

The parameter-matched dense transformer beats the MoE model on eval loss and perplexity:

- Dense 14.87M eval loss: `6.6496`
- MoE 15.01M eval loss: `6.6688`
- Dense advantage: `0.0192` loss

MoE has slightly higher token accuracy (`13.28%` vs `13.17%`), but it is slower (`40.6s` vs `29.8s`) and does not win the cleaner language-modeling objective.

This suggests the earlier MoE improvement over the 11.45M dense baseline was mostly parameter count, not routing. MoE remains viable, but it is not yet a win at equal parameter count on this small FineWeb-Edu setup.

### Scientist's Note

This was the right control. It prevents us from fooling ourselves. A bigger dense model absorbed the MoE advantage immediately. That does not kill MoE, because MoE is usually about scaling active compute differently from total parameters, and this toy implementation evaluates all experts before gathering. But for this repo, right now, the learned-phase capsule result is scientifically stronger than FineWeb MoE.

---

## 2026-06-27 15: Capacity-Matched Learned-Phase Capsule Control

### Why This Run

The earlier learned-phase capsule result compared `ComplexCapsuleB` against `RealCapsule`, but `ComplexCapsuleB` had about twice as many parameters. That is a dangerous confound (a confound is an alternate explanation for a result). I added `RealCapsuleLarge`, an exact parameter-matched real-valued capsule control.

The control is intentionally conservative: it keeps the real capsule formulation but gives it a second primary capsule bank and a second routing transform so the parameter count matches `ComplexCapsuleB`.

### Parameter Check

| Model | Parameters |
|---|---:|
| RealCapsule | 1,394,320 |
| RealCapsuleLarge | 2,767,568 |
| ComplexCapsuleB | 2,767,568 |

This is the cleanest control so far for asking whether learned phase is helping beyond just model capacity.

### Runs

I ran three 20-epoch MNIST seeds with the expanded model set:

| Seed | Output |
|---:|---|
| 123 | `results/complex_mnist_capacity_20ep_seed123/` |
| 321 | `results/complex_mnist_capacity_20ep_seed321/` |
| 777 | `results/complex_mnist_capacity_20ep_seed777/` |

Then I evaluated the saved checkpoints on rotated MNIST at `0`, `15`, `30`, `45`, `60`, and `90` degrees:

| Seed | Rotation Eval Output |
|---:|---|
| 123 | `results/complex_mnist_capacity_20ep_seed123_rotation_eval/` |
| 321 | `results/complex_mnist_capacity_20ep_seed321_rotation_eval/` |
| 777 | `results/complex_mnist_capacity_20ep_seed777_rotation_eval/` |

Validation after adding the control:

```text
27 passed, 1 skipped
```

### Plain MNIST Accuracy

| Model | Mean Test Acc | Std | Seed Values |
|---|---:|---:|---|
| BaselineMLP | 98.150% | 0.111 | `[98.17, 98.03, 98.25]` |
| RealCapsule | 98.890% | 0.115 | `[98.78, 98.88, 99.01]` |
| RealCapsuleLarge | 99.067% | 0.102 | `[99.11, 99.14, 98.95]` |
| ComplexCapsuleB | 98.957% | 0.165 | `[99.12, 98.96, 98.79]` |
| ComplexCapsuleSpatialB | 98.977% | 0.121 | `[98.99, 99.09, 98.85]` |
| ComplexCapsuleA | 99.097% | 0.064 | `[99.07, 99.05, 99.17]` |

Plain MNIST is saturated. It is not the right place to decide the research direction.

### Mean Rotated MNIST Accuracy Across 3 Seeds

| Rotation | BaselineMLP | RealCapsule | RealCapsuleLarge | ComplexB | SpatialB | ComplexA |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 98.150% | 98.890% | 99.067% | 98.957% | 98.977% | 99.097% |
| 15 | 94.693% | 96.977% | 97.057% | 97.447% | 97.013% | 97.413% |
| 30 | 78.167% | 84.367% | 84.997% | 86.720% | 84.723% | 86.183% |
| 45 | 48.747% | 56.990% | 59.260% | 61.430% | 60.053% | 59.950% |
| 60 | 26.033% | 31.687% | 35.397% | 35.977% | 36.600% | 35.293% |
| 90 | 9.817% | 13.987% | 16.503% | 15.990% | 15.383% | 15.553% |

### Nonzero Rotation Advantage Versus RealCapsuleLarge

This aggregates rotations `15`, `30`, `45`, `60`, and `90` degrees.

| Model | Mean Advantage | Std Across All Seed/Rotation Cells | Seed Mean Advantages |
|---|---:|---:|---|
| ComplexCapsuleB | +0.870 points | 1.592 | `[+1.362, +1.674, -0.426]` |
| ComplexCapsuleA | +0.236 points | 1.156 | `[+0.522, +0.726, -0.540]` |
| ComplexCapsuleSpatialB | +0.112 points | 1.600 | `[+0.424, +0.740, -0.828]` |
| RealCapsule | -1.841 points | 1.514 | `[-2.334, -1.658, -1.532]` |

Breakdown for `ComplexCapsuleB - RealCapsuleLarge`:

| Rotation | Mean Diff | Std | Seed Diffs |
|---:|---:|---:|---|
| 15 | +0.390 | 0.308 | `[+0.44, +0.67, +0.06]` |
| 30 | +1.723 | 1.963 | `[+2.74, +2.97, -0.54]` |
| 45 | +2.170 | 1.960 | `[+2.77, +3.76, -0.02]` |
| 60 | +0.580 | 1.650 | `[+0.91, +2.04, -1.21]` |
| 90 | -0.513 | 0.516 | `[-0.05, -1.07, -0.42]` |

### Interpretation

The capacity-matched result is more sober than the earlier result. `ComplexCapsuleB` still beats the exact-size `RealCapsuleLarge` control on average over nonzero rotations by `+0.87` percentage points, and its strongest wins are at `30` and `45` degrees. But the advantage is not stable across seeds: seed `777` favors the large real control overall.

This means learned phase is still worth chasing, but the claim must be narrower:

- Stronger claim rejected: "Complex capsules obviously beat real capsules."
- Better claim supported: "The learned-phase angle variant shows a small, repeatable-looking rotation robustness signal after parameter matching, especially at moderate rotations, but it needs more seeds and harder datasets."

`ComplexCapsuleSpatialB` is still not persuasive. Hard-coding the spatial angle into phase does not reliably help. I think the model wants to learn phase relationships rather than receive a crude coordinate-angle prior on a tiny capsule grid.

`ComplexCapsuleA` is interesting but less compelling than `ComplexCapsuleB` for rotation robustness. It has the best mean plain MNIST accuracy in this run, but that is probably not meaningful because MNIST is saturated.

### Scientist's Note

This was the most important control of the night. It weakened the story, but in a useful way. The learned-phase result survived, yet it no longer looks like a dramatic breakthrough. That is exactly what a good control should do: strip away the easy explanation and show what is left.

What is left is still nontrivial. `ComplexCapsuleB` matching parameter count and still gaining at `30` and `45` degrees suggests phase may be encoding something about deformation or pose that the real capsule needs more difficulty capturing. But the seed instability tells me not to trust a three-seed conclusion too much.

My current research judgment:

1. Learned-phase capsules remain the main originality lead.
2. Ternary/MoE remains useful as a secondary efficiency/routing track, but it is not yet producing stronger evidence.
3. FineWeb-Edu is now viable for language experiments, but the first parameter-matched dense control says MoE is not automatically worth chasing there.
4. The next capsule step should be either more seeds on rotated MNIST or a harder vision dataset where pose/deformation is less toy-like.

---

## 2026-06-27 16: Targeted Capsule Sweep Handoff Note

### What Changed

After the three-seed capacity-matched result, I tried to extend the full six-model sweep to seed `2024`. That was too slow for the value gained: the run completed `BaselineMLP` and `RealCapsule`, then was interrupted during `RealCapsuleLarge`. The interruption was intentional. It preserved partial artifacts and revealed that full all-model sweeps are not the right operational unit for the next phase.

I added a targeted complex experiment selector:

```bash
--models real-large complex-b
```

The default remains `--models all`, so existing behavior is unchanged.

### AMP Bug Found And Fixed

A targeted smoke run with `--amp` found a real bug:

```text
NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'ComplexFloat'
```

Interpretation: PyTorch's AMP gradient scaler (automatic mixed precision, lower-precision GPU math with scaled gradients) cannot handle complex-valued parameters in this path. I patched `train_models` so AMP remains enabled for real-valued models but is automatically disabled for models with complex parameters.

Smoke confirmation:

```bash
.venv/bin/python complex-capsules/train.py \
  --epochs 1 \
  --device auto \
  --seed 42 \
  --models real-large complex-b \
  --train-subset 16 \
  --test-subset 16 \
  --limit-train-batches 1 \
  --limit-test-batches 1 \
  --batch-size 4 \
  --amp \
  --output-dir results/smoke_complex_subset \
  --checkpoint-dir checkpoints/smoke_complex_subset
```

Result: completed successfully. The trainer printed `AMP disabled for complex-valued parameters` for `ComplexCapsuleB`.

### Active Run At Handoff

The active run at handoff is:

```bash
.venv/bin/python complex-capsules/train.py \
  --epochs 20 \
  --device auto \
  --seed 2024 \
  --models real-large complex-b \
  --amp \
  --output-dir results/complex_mnist_capacity_target_20ep_seed2024 \
  --checkpoint-dir checkpoints/complex_mnist_capacity_target_20ep_seed2024
```

At the time of this note:

- `RealCapsuleLarge` had completed and written its checkpoint/history/confusion artifacts.
- `ComplexCapsuleB` was still training.
- PID observed: `2198787`.
- The process was CPU-active and using CUDA memory, but GPU utilization was modest. This capsule code path appears CPU-side heavy.

Postscript: the targeted seed `2024` run finished at `11:51`.

Plain MNIST result:

| Model | Params | Final Test Acc | Final Test Loss | Time |
|---|---:|---:|---:|---:|
| RealCapsuleLarge | 2,767,568 | 99.030% | 1.472439 | 443.9s |
| ComplexCapsuleB | 2,767,568 | 98.830% | 0.456015 | 427.2s |

Plain MNIST favors `RealCapsuleLarge` on accuracy for this seed. This does not settle the phase question because the earlier signal is specifically rotation robustness, especially at moderate rotations. The next required command is still rotated evaluation for seed `2024`.

Final verification after the selector and AMP guard:

```text
28 passed, 1 skipped in 8.75s
```

---

## 2026-06-27 17: Dataset Path Forward For Learned-Phase Capsules

### Question

Jim asked what dataset and experiment path makes the most sense for learned-phase capsules, given the current notebook findings and a practical hardware budget of two RTX 5060 Ti GPUs, roughly `32GB` total VRAM but effectively `16GB` per card unless we implement real multi-GPU training.

### Current Judgment

Do not jump to FineWeb, DDCLM, or another large text corpus for the learned-phase capsule question.

The current evidence is geometric, not linguistic. `ComplexCapsuleB` showed its most interesting signal on rotated MNIST, especially at moderate rotations, after being compared to an exact parameter-matched `RealCapsuleLarge` control. A text dataset would force a different architecture and would test language modeling plus routing plus token representation before we have established whether learned phase is doing useful pose/deformation work.

The next path should be cleaner, not bigger.

### Recommended Dataset Ladder

1. Rotated MNIST targeted controls
2. AffNIST or synthetic affine-MNIST
3. dSprites as an interpretability/probing dataset
4. SmallNORB
5. CIFAR-10-C or rotated/natural-image datasets later

### Step 1: Finish Targeted Rotated MNIST

Run only the decisive parameter-matched pair:

- `RealCapsuleLarge`
- `ComplexCapsuleB`

Use more seeds and evaluate:

- `0`
- `15`
- `30`
- `45`
- `60`
- `75`
- `90`

The earlier three-seed aggregate found `ComplexCapsuleB` ahead of `RealCapsuleLarge` by about `+0.87` percentage points over nonzero rotations, but the result was seed-sensitive. The immediate job is not to add a new model family. It is to determine whether this modest phase advantage survives more seeds.

Also add random-rotation evaluation, not only deterministic fixed rotations. Fixed rotations are useful diagnostics, but random rotations better represent a distribution shift.

### Step 2: AffNIST / Synthetic Affine-MNIST

This is the best next dataset.

Reason: it stays in the digit domain but adds affine transforms (rotation, translation, scaling, shear). That directly targets the learned-phase hypothesis: phase may encode pose, deformation, or transformation structure better than a real-valued capsule at the same parameter count.

This avoids adding color, background clutter, texture, and natural-image semantic complexity too early. The goal is to test phase, not to accidentally test whether a small capsule net can handle natural image statistics.

Recommended experiment design:

- Train on clean MNIST, test on affine distortions.
- Train with mild affine augmentation, test on stronger held-out affine distortions.
- Compare same parameter budget:
  - `SmallCNN`
  - `RealCapsuleLarge`
  - `ComplexCapsuleB`

Scientific value: this separates general out-of-distribution robustness from transformation learning.

### Step 3: dSprites Probe

dSprites is not a main benchmark, but it is a very good diagnostic dataset because it has known latent factors such as shape, scale, rotation, and position.

Use it to ask:

- Does learned phase correlate with rotation?
- Does learned phase correlate with x/y position?
- Does learned phase track scale?
- Does phase collapse under simple generated factors?

This would make the learned-phase story more interpretable. A positive classification result is useful, but a measured relationship between phase and known generative factors would be much stronger evidence that the mechanism is doing something meaningful.

### Step 4: SmallNORB

SmallNORB is the first serious dataset I would care about for this idea.

Reason: capsules were originally motivated by pose and viewpoint. SmallNORB contains objects under viewpoint and lighting changes, so it is much closer to the actual claim: learned phase might help represent object pose or transformation.

Recommended comparison:

- `SmallCNN`
- `RealCapsuleLarge`
- `ComplexCapsuleB`

Do not try to beat ResNet yet. That is not the right claim. The right claim is narrower:

At similar parameter count, learned phase improves pose/viewpoint robustness over a real-valued capsule control.

### Step 5: CIFAR-10-C / Natural Images Later

CIFAR-10 is useful for credibility because people recognize it, but it is a poor discovery dataset for this specific hypothesis. It adds color, texture, clutter, and natural-image variation all at once. That makes failures hard to interpret and successes hard to attribute.

Use CIFAR-10-C, rotated CIFAR-10, or other natural-image variants only after the geometry result survives cleaner tests.

### Hardware Note

Two RTX 5060 Ti cards give useful throughput, but they are not automatically one `32GB` VRAM pool. Unless the code is changed for distributed or model-parallel execution, each run mostly gets one card's memory.

For these recommended datasets, VRAM is not the main blocker. The notebook already observed modest GPU utilization and heavy CPU activity during capsule runs. The capsule routing/training code appears more CPU-side bottlenecked than VRAM-limited.

The next engineering priority for longer experiments is therefore:

- targeted model subsets,
- per-epoch CSV flushing,
- recoverable checkpoints,
- resume support,
- better progress logging,
- possibly CPU/data-loader/routing profiling.

### Scientist's Note

The temptation is to go bigger because bigger datasets feel more serious. I think that is the wrong instinct here. Learned-phase capsules are making a geometric claim. The right next datasets are those that isolate geometry, pose, and transformation factors.

My current staircase:

```text
Rotated MNIST, more seeds
-> AffNIST / affine-MNIST
-> dSprites probe
-> SmallNORB
-> CIFAR-10-C or natural images
```

This gives each experiment one cleaner question than the last. If the phase signal dies on AffNIST or dSprites, we learn that quickly and cheaply. If it survives into SmallNORB, then we have a much stronger reason to invest in larger architectures or more expensive datasets.

### Next Step

When the targeted run finishes, run rotated evaluation:

```bash
.venv/bin/python -m ai_unity.evaluate_complex_rotations \
  --comparison-json results/complex_mnist_capacity_target_20ep_seed2024/complex_capsules_comparison.json \
  --output-dir results/complex_mnist_capacity_target_20ep_seed2024_rotation_eval \
  --data-dir complex-capsules/data \
  --device auto \
  --seed 2024
```

Then aggregate seed `2024` with seeds `123`, `321`, and `777`, focusing on `ComplexCapsuleB - RealCapsuleLarge` over nonzero rotations.

### Scientist's Note

The engineering lesson is as important as the model result: our trainer should persist per epoch, not just per completed model. A long capsule model currently has no recoverable mid-model state and stdout is buffered. Before serious overnight runs, add per-epoch CSV flushing, optional per-epoch checkpoints, and ideally resume support.

---

## 2026-06-27 18: Seed 2024 Rotation Eval And Four-Seed Targeted Aggregate

### Command

```bash
.venv/bin/python -m ai_unity.evaluate_complex_rotations \
  --comparison-json results/complex_mnist_capacity_target_20ep_seed2024/complex_capsules_comparison.json \
  --output-dir results/complex_mnist_capacity_target_20ep_seed2024_rotation_eval \
  --data-dir complex-capsules/data \
  --device auto \
  --seed 2024
```

### Seed 2024 Rotation Result

| Rotation | RealCapsuleLarge | ComplexCapsuleB | Diff |
|---:|---:|---:|---:|
| 0 | 99.03% | 98.83% | -0.20 |
| 15 | 97.25% | 96.99% | -0.26 |
| 30 | 84.49% | 84.92% | +0.43 |
| 45 | 57.90% | 58.19% | +0.29 |
| 60 | 33.70% | 32.39% | -1.31 |
| 90 | 15.26% | 16.01% | +0.75 |

`ComplexCapsuleB` phase standard deviation stayed broad, from `1.814` to `1.846`, and no phase collapse was reported.

### Four-Seed Plain MNIST Accuracy

This compares only the targeted pair across seeds `123`, `321`, `777`, and `2024`.

| Model | Mean Test Acc | Std | Seed Values |
|---|---:|---:|---|
| RealCapsuleLarge | 99.058% | 0.085 | `[99.11, 99.14, 98.95, 99.03]` |
| ComplexCapsuleB | 98.925% | 0.149 | `[99.12, 98.96, 98.79, 98.83]` |

Plain MNIST now slightly favors `RealCapsuleLarge`. This remains a low-value distinction because the task is saturated.

### Four-Seed Mean Rotated MNIST Accuracy

| Rotation | RealCapsuleLarge | ComplexCapsuleB | Diff |
|---:|---:|---:|---:|
| 0 | 99.058% | 98.925% | -0.133 |
| 15 | 97.105% | 97.333% | +0.228 |
| 30 | 84.870% | 86.270% | +1.400 |
| 45 | 58.920% | 60.620% | +1.700 |
| 60 | 34.973% | 35.080% | +0.107 |
| 90 | 16.193% | 15.995% | -0.198 |

### Nonzero Rotation Advantage Versus RealCapsuleLarge

This aggregates rotations `15`, `30`, `45`, `60`, and `90` degrees.

| Model | Mean Advantage | Std Across All Seed/Rotation Cells | Seed Mean Advantages |
|---|---:|---:|---|
| ComplexCapsuleB | +0.647 points | 1.470 | `[+1.362, +1.674, -0.426, -0.020]` |

Breakdown for `ComplexCapsuleB - RealCapsuleLarge`:

| Rotation | Mean Diff | Std | Seed Diffs |
|---:|---:|---:|---|
| 15 | +0.227 | 0.411 | `[+0.44, +0.67, +0.06, -0.26]` |
| 30 | +1.400 | 1.729 | `[+2.74, +2.97, -0.54, +0.43]` |
| 45 | +1.700 | 1.856 | `[+2.77, +3.76, -0.02, +0.29]` |
| 60 | +0.107 | 1.646 | `[+0.91, +2.04, -1.21, -1.31]` |
| 90 | -0.198 | 0.759 | `[-0.05, -1.07, -0.42, +0.75]` |

### Interpretation

Seed `2024` weakens the earlier three-seed result. The nonzero-rotation advantage for this seed is essentially flat at `-0.02` percentage points. Added to the previous seeds, the four-seed mean advantage drops from `+0.87` to `+0.65` percentage points.

The shape of the result is still not random-looking. `ComplexCapsuleB` remains better at the moderate rotations where the hypothesis most naturally points: `+1.40` at `30` degrees and `+1.70` at `45` degrees across four seeds. But the advantage is fragile. At `60` degrees it is almost gone, and at `90` degrees it slightly favors `RealCapsuleLarge`.

My updated judgment:

1. Learned-phase `ComplexCapsuleB` remains the best capsule lead, but the claim should now be even narrower.
2. The honest claim is: same-size learned-phase capsules show a small average robustness signal on moderate rotated MNIST, not a broad or decisive win.
3. The next useful test is not more celebration on MNIST. It is either more targeted seeds with `75` degrees and random rotations, or the AffNIST / synthetic affine-MNIST step documented above.

### Scientist's Note

This is a good example of why the fourth seed mattered. The original result survived, but it got smaller and less rhetorically convenient. That is useful evidence. If learned phase is real, it should survive being made less pretty.

I would still continue this line, but with discipline: keep the parameter-matched pair, add random affine stress, and look for whether the moderate-rotation signal generalizes beyond deterministic MNIST rotations.

---

## 2026-06-27 19: Extended Rotation Eval With 75 Degrees And Random Ranges

### Code Change

I extended `ai_unity.evaluate_complex_rotations` so it can now:

- evaluate a model subset with `--models`, for example `--models real-large complex-b`;
- include `75` degrees in the default deterministic rotation sweep;
- evaluate reproducible random-rotation ranges with repeated `--random-rotations MIN MAX`;
- run multiple independent random draws per range with `--random-samples`.

I also added an index-seeded random rotation dataset wrapper in `ai_unity.data`. This matters because random rotations should be reproducible and should assign the same random angle to the same example when evaluating flat and image-shaped model inputs.

Validation:

```text
30 passed, 1 skipped in 9.16s
```

### Command Pattern

I ran the extended targeted eval for seeds `123`, `321`, `777`, and `2024`, evaluating only the parameter-matched pair:

```bash
.venv/bin/python -m ai_unity.evaluate_complex_rotations \
  --comparison-json results/<seed-output>/complex_capsules_comparison.json \
  --output-dir results/<seed-output>_rotation_eval_extended \
  --data-dir complex-capsules/data \
  --device auto \
  --seed <seed> \
  --models real-large complex-b \
  --random-rotations -15 15 \
  --random-rotations -30 30 \
  --random-rotations -45 45 \
  --random-rotations -90 90
```

### Four-Seed Fixed Rotation Result

| Rotation | RealCapsuleLarge | ComplexCapsuleB | Diff |
|---:|---:|---:|---:|
| 0 | 99.058% | 98.925% | -0.132 |
| 15 | 97.105% | 97.333% | +0.227 |
| 30 | 84.870% | 86.270% | +1.400 |
| 45 | 58.920% | 60.620% | +1.700 |
| 60 | 34.973% | 35.080% | +0.107 |
| 75 | 21.228% | 21.008% | -0.220 |
| 90 | 16.193% | 15.995% | -0.198 |

Including `75` degrees, the fixed nonzero-rotation mean advantage becomes:

```text
ComplexCapsuleB - RealCapsuleLarge: +0.503 points, std 1.425
```

Seed mean advantages over nonzero fixed rotations including `75`:

| Seed | Mean Diff | Rotation Diffs |
|---:|---:|---|
| 123 | +1.012 | `[+0.44, +2.74, +2.77, +0.91, -0.74, -0.05]` |
| 321 | +1.407 | `[+0.67, +2.97, +3.76, +2.04, +0.07, -1.07]` |
| 777 | -0.567 | `[+0.06, -0.54, -0.02, -1.21, -1.27, -0.42]` |
| 2024 | +0.160 | `[-0.26, +0.43, +0.29, -1.31, +1.06, +0.75]` |

### Four-Seed Random Rotation Result

Each random range used one reproducible random draw of the MNIST test set per seed.

| Random Range | RealCapsuleLarge | ComplexCapsuleB | Diff | Diff Std |
|---|---:|---:|---:|---:|
| `[-15, 15]` | 98.345% | 98.135% | -0.210 | 0.155 |
| `[-30, 30]` | 94.520% | 94.710% | +0.190 | 0.194 |
| `[-45, 45]` | 85.440% | 85.792% | +0.353 | 0.604 |
| `[-90, 90]` | 56.752% | 56.737% | -0.015 | 0.702 |

Across all random-range seed cells:

```text
ComplexCapsuleB - RealCapsuleLarge: +0.079 points, std 0.481
```

### Interpretation

This result narrows the learned-phase story again.

The moderate deterministic rotations are still the strongest signal. `ComplexCapsuleB` remains ahead at `30` and `45` degrees by `+1.40` and `+1.70` percentage points across four seeds. But the new `75` degree point does not support a broad rotation-invariance claim, and random rotations mostly collapse the advantage to near zero.

The random-range result is especially important because it is closer to a distribution shift than a hand-picked diagnostic angle. `ComplexCapsuleB` is not clearly better over `[-15, 15]`, `[-30, 30]`, `[-45, 45]`, and `[-90, 90]` as distributions. It has a tiny positive average, but not enough to treat as a real win.

Updated research judgment:

1. Learned phase still looks useful at specific moderate rotations.
2. It does not yet look like general rotation robustness.
3. The next useful scientific move is affine-MNIST or AffNIST, not more deterministic MNIST rotation angles.
4. The next useful engineering move is per-epoch persistence and resume support before longer dataset runs.

### Scientist's Note

This is the kind of result that saves time. The original fixed-angle curve could tempt us into overfitting the story to `30` and `45` degrees. The random-range check says: be careful. There may be a phase advantage around particular deformation severities, but it is not yet a broad invariant representation.

I would now move to synthetic affine-MNIST because it can test rotation, translation, scale, and shear while staying close enough to MNIST that failures remain interpretable.

---

## 2026-06-27 20: Synthetic Affine-MNIST Evaluation Phase

### What I Implemented

I added synthetic affine-MNIST evaluation without introducing a new dataset download dependency.

Code changes:

- `ai_unity.data`
  - Added fixed affine transforms.
  - Added index-seeded random affine datasets.
  - Extended `get_vision_loaders` and `get_mnist_pair_loaders` to accept affine transform modes.
- `ai_unity.evaluate_complex_affines`
  - New evaluator for clean-trained checkpoints on fixed and random affine MNIST distortions.
  - Supports `--models`, `--fixed-scenarios`, `--random-scenarios`, and `--random-samples`.
- `tests/test_training_smoke.py`
  - Added parser coverage and deterministic random-affine checks.

Validation:

```text
33 passed, 1 skipped in 8.90s
```

### Eval Design

I evaluated only the parameter-matched pair:

- `RealCapsuleLarge`
- `ComplexCapsuleB`

Seeds:

- `123`
- `321`
- `777`
- `2024`

Command pattern:

```bash
.venv/bin/python -m ai_unity.evaluate_complex_affines \
  --comparison-json results/<seed-output>/complex_capsules_comparison.json \
  --output-dir results/<seed-output>_affine_eval \
  --data-dir complex-capsules/data \
  --device auto \
  --seed <seed> \
  --models real-large complex-b
```

This evaluates clean-trained MNIST checkpoints on synthetic affine distortions. It does not retrain with affine augmentation yet.

### Fixed Affine Results Across Four Seeds

| Scenario | RealCapsuleLarge | ComplexCapsuleB | Diff | Diff Std |
|---|---:|---:|---:|---:|
| clean | 99.058% | 98.925% | -0.133 | 0.096 |
| translate4 | 14.540% | 18.032% | +3.492 | 5.510 |
| translate8 | 4.875% | 11.818% | +6.943 | 4.302 |
| scale085 | 97.788% | 97.718% | -0.070 | 0.275 |
| scale115 | 98.028% | 97.588% | -0.440 | 0.302 |
| shear15 | 98.543% | 98.317% | -0.225 | 0.164 |
| affine_moderate | 13.120% | 16.288% | +3.167 | 6.348 |
| affine_strong | 10.765% | 14.727% | +3.962 | 4.098 |

Across fixed out-of-distribution affine scenarios, excluding clean:

```text
ComplexCapsuleB - RealCapsuleLarge: +2.404 points, std 4.313
```

Seed mean advantages across fixed OOD scenarios:

| Seed | Mean Diff | Scenario Diffs |
|---:|---:|---|
| 123 | -0.089 | `[-0.71, +1.02, +0.34, -0.14, -0.23, +0.43, -1.33]` |
| 321 | +3.126 | `[+0.83, +11.18, -0.22, -0.37, +0.00, +5.04, +5.42]` |
| 777 | +5.246 | `[+11.55, +7.05, -0.16, -0.86, -0.28, +11.00, +8.42]` |
| 2024 | +1.334 | `[+2.30, +8.52, -0.24, -0.39, -0.39, -3.80, +3.34]` |

### Random Affine Results Across Four Seeds

Each random scenario used one reproducible affine draw per seed.

| Scenario | RealCapsuleLarge | ComplexCapsuleB | Diff | Diff Std |
|---|---:|---:|---:|---:|
| random_translate | 38.578% | 38.523% | -0.055 | 0.929 |
| random_translate_scale | 50.850% | 49.685% | -1.165 | 1.584 |
| random_affine_mild | 60.855% | 59.365% | -1.490 | 1.172 |
| random_affine_strong | 29.102% | 29.015% | -0.088 | 0.787 |

Across random affine scenario seed cells:

```text
ComplexCapsuleB - RealCapsuleLarge: -0.699 points, std 1.228
```

Seed mean advantages across random affine scenarios:

| Seed | Mean Diff | Scenario Diffs |
|---:|---:|---|
| 123 | +0.065 | `[-0.27, -0.08, -0.10, +0.71]` |
| 321 | -1.825 | `[-0.84, -2.66, -2.63, -1.17]` |
| 777 | -1.265 | `[-0.40, -2.38, -2.27, -0.01]` |
| 2024 | +0.228 | `[+1.29, +0.46, -0.96, +0.12]` |

### Interpretation

This is not a clean win for learned phase.

The fixed affine table has a tempting signal: `ComplexCapsuleB` is substantially better on deterministic translations and deterministic combined affine transforms. The largest mean gap is `translate8`, where `ComplexCapsuleB` beats `RealCapsuleLarge` by `+6.94` percentage points.

But the random affine table cuts against a broad robustness claim. When affine transforms are sampled as distributions rather than fixed hand-picked transformations, `RealCapsuleLarge` slightly wins overall by about `0.70` percentage points. That is not a huge gap, but it is enough to say learned phase is not broadly better under synthetic affine distortion.

The fixed translation results are still interesting. Both models collapse badly under translation, which means clean MNIST training has not learned true translation robustness. `ComplexCapsuleB` sometimes fails less badly, especially on seed `777` and `translate8`. But this could be an artifact of the specific fixed translation direction, not a general phase advantage.

Updated research judgment:

1. Learned phase remains interesting, but the evidence is now highly conditional.
2. The strongest positive signal is specific deterministic transformation cases, not sampled transformation distributions.
3. The next real test should train with mild affine augmentation and evaluate on stronger held-out affine distributions.
4. If affine-augmented training does not produce a learned-phase advantage, the capsule originality lead should shift from "robustness" toward "diagnostics/interpretable phase behavior" or be deprioritized.

### Next Experimental Step

Train only the parameter-matched pair with mild affine augmentation:

- train: random affine mild
  - rotation `[-30, 30]`
  - translation up to `15%`
  - scale `[0.85, 1.15]`
  - x-shear `[-15, 15]`
- test:
  - clean MNIST
  - random affine mild
  - random affine strong
  - deterministic fixed translation and combined affine cases

This asks a better question than the current clean-trained OOD eval:

Does learned phase help after both models are actually trained to see affine variation?

### Scientist's Note

This result is more useful than it is flattering. Learned phase keeps showing pockets of advantage, but the broad version of the story keeps failing when the test becomes more distributional. That does not kill the idea, but it changes the burden of proof.

The right next move is affine-augmented training, not more post-hoc evaluation of clean-trained checkpoints. If `ComplexCapsuleB` is genuinely using phase to organize pose/deformation, it should learn from affine variation better than the same-size real capsule control.

---

## 2026-06-27 21: Mild Affine-Augmented Training

### What I Implemented

I extended complex capsule training with an affine augmentation preset:

```bash
--affine-augment random-affine-mild
```

The mild preset is:

- rotation `[-30, 30]`
- translation up to `15%`
- scale `[0.85, 1.15]`
- x-shear `[-15, 15]`
- y-shear fixed at `0`

Validation after the training CLI change:

```text
33 passed, 1 skipped in 8.84s
```

Smoke command:

```bash
.venv/bin/python complex-capsules/train.py \
  --epochs 1 \
  --device auto \
  --seed 42 \
  --models real-large complex-b \
  --affine-augment random-affine-mild \
  --train-subset 16 \
  --test-subset 16 \
  --limit-train-batches 1 \
  --limit-test-batches 1 \
  --batch-size 4 \
  --amp \
  --output-dir results/smoke_complex_affine_aug \
  --checkpoint-dir checkpoints/smoke_complex_affine_aug
```

Result: completed successfully. AMP was automatically disabled for `ComplexCapsuleB`, as expected.

### Training Runs

I trained only the parameter-matched pair:

- `RealCapsuleLarge`
- `ComplexCapsuleB`

Command pattern:

```bash
.venv/bin/python complex-capsules/train.py \
  --epochs 20 \
  --device auto \
  --seed <seed> \
  --models real-large complex-b \
  --affine-augment random-affine-mild \
  --amp \
  --output-dir results/complex_mnist_affine_aug_mild_20ep_seed<seed> \
  --checkpoint-dir checkpoints/complex_mnist_affine_aug_mild_20ep_seed<seed>
```

Seeds:

- `123`
- `321`
- `777`
- `2024`

### Final Training Eval On Mild-Affine Test Loader

| Seed | RealCapsuleLarge | ComplexCapsuleB | Diff |
|---:|---:|---:|---:|
| 123 | 96.060% | 95.610% | -0.450 |
| 321 | 95.960% | 96.740% | +0.780 |
| 777 | 95.870% | 96.870% | +1.000 |
| 2024 | 95.330% | 96.640% | +1.310 |

Mean:

```text
ComplexCapsuleB - RealCapsuleLarge: +0.660 points, std 0.771
```

This is already more favorable to learned phase than the clean-trained random affine evaluation.

### Post-Training Affine Eval

I evaluated the augmented checkpoints with the same affine suite:

```bash
.venv/bin/python -m ai_unity.evaluate_complex_affines \
  --comparison-json results/complex_mnist_affine_aug_mild_20ep_seed<seed>/complex_capsules_comparison.json \
  --output-dir results/complex_mnist_affine_aug_mild_20ep_seed<seed>_affine_eval \
  --data-dir complex-capsules/data \
  --device auto \
  --seed <seed> \
  --models real-large complex-b
```

### Fixed Affine Results Across Four Seeds

| Scenario | RealCapsuleLarge | ComplexCapsuleB | Diff | Diff Std |
|---|---:|---:|---:|---:|
| clean | 98.420% | 98.567% | +0.148 | 0.356 |
| translate4 | 94.290% | 95.990% | +1.700 | 0.628 |
| translate8 | 16.040% | 18.697% | +2.657 | 3.367 |
| scale085 | 97.123% | 97.490% | +0.367 | 0.990 |
| scale115 | 97.892% | 97.960% | +0.068 | 0.464 |
| shear15 | 97.942% | 98.075% | +0.132 | 0.758 |
| affine_moderate | 82.820% | 87.657% | +4.837 | 3.941 |
| affine_strong | 26.800% | 30.188% | +3.388 | 4.906 |

Across fixed OOD affine scenarios, excluding clean:

```text
ComplexCapsuleB - RealCapsuleLarge: +1.879 points, std 2.984
```

Seed mean advantages across fixed OOD scenarios:

| Seed | Mean Diff | Scenario Diffs |
|---:|---:|---|
| 123 | +0.190 | `[+1.22, +4.06, -1.08, -0.60, -0.89, -0.63, -0.75]` |
| 321 | +1.376 | `[+1.10, -2.34, +0.58, +0.13, +0.09, +8.62, +1.45]` |
| 777 | +2.164 | `[+2.17, +3.90, +0.85, +0.45, +0.43, +4.98, +2.37]` |
| 2024 | +3.784 | `[+2.31, +5.01, +1.12, +0.29, +0.90, +6.38, +10.48]` |

### Random Affine Results Across Four Seeds

| Scenario | RealCapsuleLarge | ComplexCapsuleB | Diff | Diff Std |
|---|---:|---:|---:|---:|
| random_translate | 86.630% | 89.782% | +3.153 | 1.285 |
| random_translate_scale | 93.745% | 95.060% | +1.315 | 1.136 |
| random_affine_mild | 95.767% | 96.575% | +0.807 | 0.858 |
| random_affine_strong | 74.892% | 79.925% | +5.033 | 2.019 |

Across random affine scenario seed cells:

```text
ComplexCapsuleB - RealCapsuleLarge: +2.577 points, std 2.123
```

Seed mean advantages across random affine scenarios:

| Seed | Mean Diff | Scenario Diffs |
|---:|---:|---|
| 123 | +0.763 | `[+1.43, -0.21, -0.31, +2.14]` |
| 321 | +2.913 | `[+3.56, +1.62, +0.93, +5.54]` |
| 777 | +2.723 | `[+3.12, +1.33, +0.83, +5.61]` |
| 2024 | +3.910 | `[+4.50, +2.52, +1.78, +6.84]` |

### Interpretation

This is the strongest learned-phase result so far.

The clean-trained post-hoc affine evaluation did not support broad robustness: `ComplexCapsuleB` was slightly behind on random affine distributions. After both models were trained with mild affine augmentation, the result flipped:

```text
clean-trained random affine:      -0.699 points
affine-augmented random affine:   +2.577 points
```

The important point is not just that `ComplexCapsuleB` wins after augmentation. It wins most clearly on the harder random affine stress:

- `random_translate`: `+3.15`
- `random_translate_scale`: `+1.32`
- `random_affine_mild`: `+0.81`
- `random_affine_strong`: `+5.03`

This suggests learned phase may not give automatic invariance when the model only sees clean MNIST. But when both models are trained on transformation variation, the complex learned-phase representation appears to use that training signal better than the same-size real capsule control.

This is a materially better hypothesis:

> Learned phase improves transformation learning under affine augmentation, rather than providing free out-of-distribution invariance from clean training.

### Cautions

The result is promising, not final.

- Seed `123` is weak and mixed.
- Final epoch accuracy is noisy; ComplexB sometimes peaked earlier than epoch `20`.
- The trainer still only saves final checkpoints, not best checkpoints.
- Random affine eval currently uses one sampled test set per scenario per seed. More random samples would reduce draw noise.
- We still need a true held-out dataset such as AffNIST, not only synthetic affine transforms.

### Next Experimental Step

The next high-value task is engineering, not a new model:

1. Add per-epoch CSV flushing.
2. Save best checkpoints by validation accuracy.
3. Add resume support.
4. Re-evaluate best checkpoints, not only final epoch checkpoints.
5. Run random affine eval with multiple samples per scenario.

After that, move to AffNIST or a synthetic affine benchmark with a held-out transform distribution.

### Scientist's Note

This changes my confidence. The earlier story was shrinking because clean-trained checkpoints did not generalize broadly. The augmented result gives the idea a better shape: phase may help the model learn transformation families when the training distribution actually contains those transformations.

That is a much more defensible research claim than "complex capsules are rotation invariant." It also fits the evidence: no magic invariance, but better use of geometric variation once exposed to it.

---

## 2026-06-27 22: Durable Training Checkpoints And Resume Support

### What I Implemented

I added the engineering support needed before the next long run:

- per-epoch history CSV flushing;
- per-epoch latest checkpoint saving;
- best checkpoint saving by test accuracy;
- `--resume` support for continuing from the latest checkpoint;
- result JSON fields for checkpoint selection:
  - `checkpoint`
  - `latest_checkpoint`
  - `best_checkpoint`
  - `best_test_acc`
  - `best_epoch`
- evaluator support for checkpoint selection:
  - `--checkpoint-key checkpoint`
  - `--checkpoint-key latest_checkpoint`
  - `--checkpoint-key best_checkpoint`

This keeps the existing final checkpoint path compatible while adding `_latest.pt` and `_best.pt` files next to it.

### Validation

Full test suite:

```text
34 passed, 1 skipped in 9.64s
```

I also ran a CLI resume smoke through the real complex capsule entrypoint:

```bash
.venv/bin/python complex-capsules/train.py \
  --epochs 1 \
  --device auto \
  --seed 99 \
  --models real-large complex-b \
  --affine-augment random-affine-mild \
  --train-subset 16 \
  --test-subset 16 \
  --limit-train-batches 1 \
  --limit-test-batches 1 \
  --batch-size 4 \
  --amp \
  --output-dir results/smoke_complex_resume \
  --checkpoint-dir checkpoints/smoke_complex_resume
```

Then resumed:

```bash
.venv/bin/python complex-capsules/train.py \
  --epochs 2 \
  --device auto \
  --seed 99 \
  --models real-large complex-b \
  --affine-augment random-affine-mild \
  --train-subset 16 \
  --test-subset 16 \
  --limit-train-batches 1 \
  --limit-test-batches 1 \
  --batch-size 4 \
  --amp \
  --resume \
  --output-dir results/smoke_complex_resume \
  --checkpoint-dir checkpoints/smoke_complex_resume
```

Both `RealCapsuleLarge` and `ComplexCapsuleB` resumed from epoch `1` and completed epoch `2`.

Artifacts confirmed:

- `checkpoints/smoke_complex_resume/complex_capsules_RealCapsuleLarge_seed99_latest.pt`
- `checkpoints/smoke_complex_resume/complex_capsules_RealCapsuleLarge_seed99_best.pt`
- `checkpoints/smoke_complex_resume/complex_capsules_ComplexCapsuleB_phase-angle_seed99_latest.pt`
- `checkpoints/smoke_complex_resume/complex_capsules_ComplexCapsuleB_phase-angle_seed99_best.pt`

### Important Limitation

The four full affine-augmented runs in section `21` were completed before best-checkpoint saving existed. Their `complex_capsules_comparison.json` files therefore do not contain `best_checkpoint`.

That means best-checkpoint evaluation requires one of:

1. rerun the affine-augmented seeds with the durable trainer; or
2. use future runs only, accepting that the current four-seed result is final-checkpoint-only.

The current result is still valid, but it may understate `ComplexCapsuleB` because several histories show `ComplexCapsuleB` peaking before epoch `20`.

### Next Experimental Step

Rerun the affine-augmented targeted pair with the durable trainer, then evaluate:

```bash
.venv/bin/python -m ai_unity.evaluate_complex_affines \
  --comparison-json results/<durable-run>/complex_capsules_comparison.json \
  --checkpoint-key best_checkpoint \
  --output-dir results/<durable-run>_best_affine_eval \
  --data-dir complex-capsules/data \
  --device auto \
  --seed <seed> \
  --models real-large complex-b \
  --random-samples 5
```

This will answer whether the section `21` result is robust to checkpoint selection and random affine draw noise.

---

## 2026-06-27 23: Durable Rerun With Best-Checkpoint Evaluation

### Why This Rerun

The section `21` affine-augmented result was promising, but it used final checkpoints only. Several histories showed the best epoch happened before epoch `20`, especially for `ComplexCapsuleB`.

After adding durable checkpoints in section `22`, I reran the four affine-augmented seeds under new output directories and evaluated the best checkpoints with five random affine samples per scenario.

### Durable Training Runs

Command pattern:

```bash
.venv/bin/python complex-capsules/train.py \
  --epochs 20 \
  --device auto \
  --seed <seed> \
  --models real-large complex-b \
  --affine-augment random-affine-mild \
  --amp \
  --output-dir results/complex_mnist_affine_aug_mild_durable_20ep_seed<seed> \
  --checkpoint-dir checkpoints/complex_mnist_affine_aug_mild_durable_20ep_seed<seed>
```

Seeds:

- `123`
- `321`
- `777`
- `2024`

Durable checkpointing worked as intended. During active runs, each model wrote:

- per-epoch history CSV,
- `_latest.pt`,
- `_best.pt`,
- final `.pt`.

This removed the earlier "no artifacts until model completion" failure mode.

### Final Versus Best Training Eval

This table uses the mild-affine validation/test loader used during training.

| Seed | Final Diff | Best Diff | Best Epochs `(Real, ComplexB)` |
|---:|---:|---:|---|
| 123 | +0.610 | +1.020 | `(19, 6)` |
| 321 | +0.880 | +1.010 | `(8, 8)` |
| 777 | -0.880 | +0.670 | `(10, 5)` |
| 2024 | +0.850 | +1.050 | `(5, 6)` |

Mean final-checkpoint advantage:

```text
ComplexCapsuleB - RealCapsuleLarge: +0.365 points, std 0.839
```

Mean best-checkpoint advantage:

```text
ComplexCapsuleB - RealCapsuleLarge: +0.937 points, std 0.179
```

This is important: best-checkpoint selection makes the result more stable across seeds, not just larger.

### Best-Checkpoint Affine Eval

Command pattern:

```bash
.venv/bin/python -m ai_unity.evaluate_complex_affines \
  --comparison-json results/complex_mnist_affine_aug_mild_durable_20ep_seed<seed>/complex_capsules_comparison.json \
  --checkpoint-key best_checkpoint \
  --output-dir results/complex_mnist_affine_aug_mild_durable_20ep_seed<seed>_best_affine_eval_5samples \
  --data-dir complex-capsules/data \
  --device auto \
  --seed <seed> \
  --models real-large complex-b \
  --random-samples 5
```

### Fixed Affine Best-Checkpoint Results

| Scenario | RealCapsuleLarge | ComplexCapsuleB | Diff | Diff Std |
|---|---:|---:|---:|---:|
| clean | 98.455% | 98.800% | +0.345 | 0.103 |
| translate4 | 94.650% | 96.392% | +1.742 | 0.605 |
| translate8 | 15.750% | 16.988% | +1.237 | 2.165 |
| scale085 | 97.390% | 97.840% | +0.450 | 0.205 |
| scale115 | 97.990% | 98.457% | +0.468 | 0.162 |
| shear15 | 98.035% | 98.575% | +0.540 | 0.197 |
| affine_moderate | 84.945% | 89.905% | +4.960 | 2.266 |
| affine_strong | 24.168% | 30.990% | +6.822 | 4.106 |

Across fixed OOD affine scenarios, excluding clean:

```text
ComplexCapsuleB - RealCapsuleLarge: +2.317 points, std 2.962
```

### Random Affine Best-Checkpoint Results

This aggregates four seeds and five sampled test sets per random scenario, so each row summarizes `20` seed/sample cells.

| Scenario | RealCapsuleLarge | ComplexCapsuleB | Diff | Diff Std |
|---|---:|---:|---:|---:|
| random_translate | 87.367% | 90.481% | +3.113 | 0.860 |
| random_translate_scale | 94.382% | 95.856% | +1.475 | 0.452 |
| random_affine_mild | 96.004% | 97.037% | +1.033 | 0.309 |
| random_affine_strong | 76.196% | 81.225% | +5.028 | 1.031 |

Across all `80` random affine seed/sample/scenario cells:

```text
ComplexCapsuleB - RealCapsuleLarge: +2.662 points, std 1.733
```

Seed mean advantages across random affine scenarios and samples:

| Seed | Mean Diff | Std |
|---:|---:|---:|
| 123 | +2.819 | 1.768 |
| 321 | +2.707 | 1.695 |
| 777 | +1.731 | 1.175 |
| 2024 | +3.392 | 1.901 |

### Interpretation

This is now a much stronger result than section `21`.

Three things changed:

1. Best-checkpoint selection removed final-epoch noise.
2. Five random samples per scenario reduced random draw noise.
3. The ComplexB advantage stayed positive for every seed in the random affine aggregate.

The strongest current claim is:

> Under mild affine augmentation, the learned-phase `ComplexCapsuleB` learns affine transformation families better than an exactly parameter-matched real capsule control.

This is not a generic "complex capsules are invariant" claim. It is narrower and better supported:

- Clean training does not produce broad free invariance.
- Affine augmentation does produce a reliable learned-phase advantage.
- The advantage is strongest under harder sampled affine transforms, especially `random_affine_strong`.

### Cautions

- This is still synthetic affine-MNIST, not AffNIST.
- The models are small and MNIST-derived.
- The random affine evaluator samples from our own transform generator, so it is not an independent benchmark.
- We should eventually compare against a small CNN trained with the same augmentation, not only real capsules.

### Next Step

Move to an independent affine/pose dataset or add a stronger baseline:

1. Add `SmallCNN` or `BaselineCNN` with similar parameter budget.
2. Evaluate best checkpoints on AffNIST or a held-out synthetic affine generator with transform parameters not identical to training.
3. Add phase-factor diagnostics: check whether `ComplexCapsuleB` phase correlates with sampled rotation, translation, scale, or shear.

### Scientist's Note

This is the first result I would preserve as a real research lead. The earlier signals were fragile and angle-specific. This one survives more careful checkpointing, repeated random evaluation samples, and the exact parameter-matched real capsule control.

The idea has also become more precise. Learned phase is not magic invariance. It appears to be a useful coordinate system for learning transformations when the model is trained on transformation variation. That is a claim worth testing on AffNIST, dSprites, and eventually SmallNORB.

## 24. Parameter-Matched CNN Baseline

### Why This Phase Was Needed

The section `23` result established that learned phase helps `ComplexCapsuleB` beat the exactly parameter-matched real capsule control under mild affine augmentation.

That was necessary, but not sufficient. A stronger baseline is a conventional CNN with a comparable parameter budget, trained with the same augmentation and evaluated with the same best-checkpoint affine suite.

### Implementation

Added `BaselineCNN` to `ai_unity.complex_capsules` and exposed it through the trainer and affine evaluator as model key `cnn`.

Parameter counts:

| Model | Parameters |
|---|---:|
| BaselineCNN | 2,768,502 |
| RealCapsuleLarge | 2,767,568 |
| ComplexCapsuleB | 2,767,568 |

`BaselineCNN` is within `0.034%` of the capsule parameter budget.

Validation:

```text
37 passed, 1 skipped
```

Smoke training and affine evaluation also completed successfully for `cnn`, `real-large`, and `complex-b`.

### Full Experiment

Four seeds were trained for `20` epochs with the same mild affine augmentation:

```bash
.venv/bin/python complex-capsules/train.py \
  --epochs 20 \
  --device auto \
  --seed <seed> \
  --models cnn real-large complex-b \
  --affine-augment random-affine-mild \
  --amp \
  --output-dir results/complex_mnist_affine_aug_mild_cnn_20ep_seed<seed> \
  --checkpoint-dir checkpoints/complex_mnist_affine_aug_mild_cnn_20ep_seed<seed>
```

Best checkpoints were evaluated with five random samples per random affine scenario:

```bash
.venv/bin/python -m ai_unity.evaluate_complex_affines \
  --comparison-json results/<run>/complex_capsules_comparison.json \
  --checkpoint-key best_checkpoint \
  --output-dir results/<run>_best_affine_eval_5samples \
  --data-dir complex-capsules/data \
  --device auto \
  --seed <seed> \
  --models cnn real-large complex-b \
  --random-samples 5
```

### Training Result

Best checkpoint test accuracy across seeds:

| Model | Mean | Std | Seed Values |
|---|---:|---:|---|
| BaselineCNN | 97.883% | 0.046 | 97.87, 97.85, 97.95, 97.86 |
| RealCapsuleLarge | 96.280% | 0.156 | 96.27, 96.27, 96.10, 96.48 |
| ComplexCapsuleB | 97.157% | 0.104 | 97.04, 97.24, 97.10, 97.25 |

On the mild affine validation distribution, the CNN is strongest. `ComplexCapsuleB` remains clearly ahead of `RealCapsuleLarge`.

### Fixed Affine Best-Checkpoint Results

Means across four seeds:

| Scenario | CNN | RealLarge | ComplexB | Winner | ComplexB - Real | CNN - ComplexB |
|---|---:|---:|---:|---|---:|---:|
| clean | 98.95% | 98.54% | 98.75% | CNN | +0.21 | +0.20 |
| translate4 | 97.53% | 94.49% | 96.21% | CNN | +1.72 | +1.32 |
| translate8 | 10.18% | 14.62% | 17.40% | ComplexB | +2.78 | -7.21 |
| scale085 | 98.11% | 97.47% | 97.88% | CNN | +0.41 | +0.24 |
| scale115 | 98.75% | 97.99% | 98.28% | CNN | +0.29 | +0.47 |
| shear15 | 98.60% | 98.20% | 98.46% | CNN | +0.26 | +0.14 |
| affine_moderate | 90.47% | 84.79% | 90.15% | CNN | +5.36 | +0.32 |
| affine_strong | 21.04% | 26.19% | 28.97% | ComplexB | +2.79 | -7.93 |

Across all fixed affine cells:

```text
ComplexCapsuleB - RealCapsuleLarge: +1.727 points
CNN - ComplexCapsuleB: -1.557 points
```

The fixed aggregate favors `ComplexCapsuleB` because the deterministic extreme cases `translate8` and `affine_strong` are very hard for the CNN.

### Random Affine Best-Checkpoint Results

Each row aggregates four seeds and five sampled test sets, for `20` cells per scenario.

| Scenario | CNN | RealLarge | ComplexB | Winner | ComplexB - Real | CNN - ComplexB |
|---|---:|---:|---:|---|---:|---:|
| random_translate | 91.81% | 87.36% | 90.52% | CNN | +3.17 | +1.29 |
| random_translate_scale | 96.85% | 94.38% | 95.82% | CNN | +1.44 | +1.03 |
| random_affine_mild | 97.76% | 96.13% | 97.03% | CNN | +0.90 | +0.73 |
| random_affine_strong | 83.22% | 76.09% | 81.39% | CNN | +5.30 | +1.83 |

Across all `80` random affine seed/sample/scenario cells:

```text
ComplexCapsuleB - RealCapsuleLarge: +2.703 points
CNN - ComplexCapsuleB: +1.218 points
```

This is the key update: the learned-phase capsule still beats the real capsule control, but the conventional CNN is the strongest model on sampled affine distributions.

### Updated Interpretation

The current evidence supports a narrower claim than section `23` alone suggested:

> Learned phase improves the capsule model relative to an exactly parameter-matched real capsule, but it does not yet beat a simple parameter-matched CNN on the sampled synthetic affine benchmark.

That means the result is real but not yet competitive as an architecture claim.

What survived:

- `ComplexCapsuleB` beats `RealCapsuleLarge` on mild affine training accuracy and random affine evaluation.
- The phase representation does not collapse during affine evaluation.
- `ComplexCapsuleB` is distinctively better than the CNN on some deterministic extreme OOD transforms, especially `translate8` and `affine_strong`.

What did not survive:

- A broad claim that learned-phase capsules are the best affine-robust model at this parameter count.
- A claim that the synthetic affine benchmark alone is enough to justify the architecture.

### Next Step

The next step should be independent validation, not another MNIST-derived leaderboard pass.

Recommended next experiment:

1. Evaluate `BaselineCNN`, `RealCapsuleLarge`, and `ComplexCapsuleB` on AffNIST or a held-out affine generator whose transform parameters are not the same as training/eval generator defaults.
2. Add phase-factor diagnostics: record sampled rotation, translation, scale, and shear, then test whether `ComplexCapsuleB` phase statistics or capsule-level phase directions correlate with those factors.
3. If phase correlates with transform factors while the CNN still wins accuracy, treat learned phase as an interpretable pose representation lead rather than a raw accuracy lead.

Scientific read: this is still a useful research lead. The phase machinery is doing something measurable inside the capsule family. The CNN result prevents overclaiming and makes the next question sharper: is learned phase useful because it improves performance on independent pose benchmarks, or because it yields a more interpretable transformation representation?

## 25. Held-Out Affine Generator and Phase-Factor Diagnostics

### Implementation

Added two held-out random affine scenarios to `ai_unity.evaluate_complex_affines`:

| Scenario | Rotation | Translation | Scale | Shear X | Shear Y |
|---|---:|---:|---:|---:|---:|
| heldout_affine_left_zoom | `[-60, -30]` | `25%` | `[0.70, 0.85]` | `[-30, -15]` | `[-15, 0]` |
| heldout_affine_right_zoom | `[30, 60]` | `25%` | `[1.15, 1.35]` | `[15, 30]` | `[0, 15]` |

These are deliberately outside the mild affine training augmentation:

```text
rotation [-30, 30], translation 15%, scale [0.85, 1.15], shear_x [-15, 15], shear_y 0
```

Also added `--phase-diagnostics` to the affine evaluator. For random affine datasets, the dataset now exposes each example's sampled transform parameters:

- `angle_degrees`
- `translate_x_frac`
- `translate_y_frac`
- `scale`
- `shear_x_degrees`
- `shear_y_degrees`

The diagnostic pass computes correlations between those factors and `sin`/`cos` projections of the true-class `digit_phase` dimensions for complex models. Results are written to:

```text
complex_affine_phase_diagnostics.json
```

Validation:

```text
38 passed, 1 skipped
```

### Experiment

Used the existing four trained mild-affine checkpoints:

```text
results/complex_mnist_affine_aug_mild_cnn_20ep_seed{123,321,777,2024}/
```

Evaluated `cnn`, `real-large`, and `complex-b` with best checkpoints:

```bash
.venv/bin/python -m ai_unity.evaluate_complex_affines \
  --comparison-json results/<run>/complex_capsules_comparison.json \
  --checkpoint-key best_checkpoint \
  --output-dir results/<run>_heldout_affine_phase_eval_5samples \
  --data-dir complex-capsules/data \
  --device auto \
  --seed <seed> \
  --models cnn real-large complex-b \
  --fixed-scenarios clean \
  --random-scenarios heldout_affine_left_zoom heldout_affine_right_zoom \
  --random-samples 5 \
  --phase-diagnostics
```

### Held-Out Accuracy

Each row aggregates four seeds and five sampled test sets, for `20` cells per scenario.

| Scenario | CNN | RealLarge | ComplexB | Winner | ComplexB - Real | CNN - ComplexB |
|---|---:|---:|---:|---|---:|---:|
| heldout_affine_left_zoom | 47.08% | 37.33% | 42.72% | CNN | +5.39 | +4.35 |
| heldout_affine_right_zoom | 53.96% | 45.09% | 49.83% | CNN | +4.75 | +4.13 |

Across all `40` held-out seed/sample/scenario cells:

```text
CNN: 50.518% mean, 3.687 std
RealCapsuleLarge: 41.210% mean, 4.016 std
ComplexCapsuleB: 46.276% mean, 3.736 std

ComplexCapsuleB - RealCapsuleLarge: +5.066 points
CNN - ComplexCapsuleB: +4.241 points
```

Per-seed summary:

| Seed | Scenario | CNN | RealLarge | ComplexB | ComplexB - Real | CNN - ComplexB |
|---:|---|---:|---:|---:|---:|---:|
| 123 | left_zoom | 48.34% | 37.58% | 41.95% | +4.37 | +6.40 |
| 123 | right_zoom | 54.05% | 44.68% | 50.26% | +5.58 | +3.79 |
| 321 | left_zoom | 47.26% | 35.73% | 44.11% | +8.38 | +3.15 |
| 321 | right_zoom | 53.67% | 44.65% | 51.00% | +6.35 | +2.67 |
| 777 | left_zoom | 48.14% | 38.25% | 42.04% | +3.79 | +6.10 |
| 777 | right_zoom | 54.38% | 45.88% | 48.34% | +2.46 | +6.04 |
| 2024 | left_zoom | 44.56% | 37.79% | 42.79% | +5.00 | +1.77 |
| 2024 | right_zoom | 53.74% | 45.14% | 49.72% | +4.58 | +4.01 |

### Phase-Factor Diagnostics

`ComplexCapsuleB` phase did not collapse:

```text
phase_std mean: 1.777
phase_std min/max: 1.655 / 1.932
```

Best absolute phase-factor correlations, averaged across the `40` held-out diagnostic cells:

| Factor | Mean Best Abs Corr | Max |
|---|---:|---:|
| angle_degrees | 0.084 | 0.171 |
| translate_y_frac | 0.079 | 0.139 |
| translate_x_frac | 0.057 | 0.098 |
| shear_x_degrees | 0.039 | 0.067 |
| scale | 0.025 | 0.054 |
| shear_y_degrees | 0.024 | 0.058 |

The strongest individual diagnostic cells were angle correlations on seed `123` right-zoom:

```text
angle_degrees abs_corr ~= 0.16 to 0.17, mostly sin_dim_6
```

The repeated secondary signal was vertical translation:

```text
translate_y_frac abs_corr ~= 0.12 to 0.14 on the stronger cells
```

### Interpretation

This held-out generator confirms the post-CNN interpretation:

- The CNN remains the strongest accuracy baseline.
- `ComplexCapsuleB` remains consistently better than the exactly parameter-matched real capsule.
- The ComplexB margin over RealLarge is larger on these held-out transforms than it was on the earlier sampled affine suite.
- The phase representation remains high-variance, but the simple global phase-factor correlations are weak to modest.

The phase diagnostic is not yet strong evidence that the current learned phase cleanly encodes pose. It is evidence that:

1. Phase does not collapse under difficult held-out transforms.
2. Some phase dimensions repeatedly show modest association with angle or vertical translation.
3. A better diagnostic should inspect capsule-level or class-conditioned structure, not only global Pearson correlation over true-class digit phase.

### Next Step

The next useful step is to make the phase diagnostic more local and more falsifiable.

Recommended:

1. Add class-conditioned phase-factor diagnostics, because pooling all digits together may hide digit-specific pose encoding.
2. Add linear probes from `digit_phase` and `primary_phase` to affine factors, with train/test splits inside each held-out scenario.
3. Compare probe quality against CNN hidden activations. If phase is useful, it should predict pose factors better or with fewer probe parameters even when CNN wins accuracy.
4. After that, move to AffNIST as the first independent dataset rather than continuing to tune synthetic MNIST.

## 26. Held-Out Affine Linear Probes

### Implementation

Added `ai_unity.evaluate_affine_probes`.

The probe evaluator loads best checkpoints, builds the held-out affine datasets, extracts fixed representations, and trains Ridge probes to predict the sampled affine factors:

- `angle_degrees`
- `translate_x_frac`
- `translate_y_frac`
- `scale`
- `shear_x_degrees`
- `shear_y_degrees`

Feature blocks:

| Model | Feature Block | Dims |
|---|---|---:|
| BaselineCNN | `cnn_penultimate` | 382 |
| RealCapsuleLarge | `digit_caps` | 80 |
| ComplexCapsuleB | `digit_caps` | 240 |
| ComplexCapsuleB | `digit_phase_all` | 160 |
| ComplexCapsuleB | `digit_phase_true` | 16 |
| ComplexCapsuleB | `primary_phase_stats` | 32 |

`primary_phase_stats` contains mean/std summaries of `sin(primary_phase)` and `cos(primary_phase)` over primary capsules. This is a compact probe, not a flattened high-dimensional primary phase probe.

Validation:

```text
40 passed, 1 skipped
```

### Experiment

For each seed and each held-out affine scenario, the probe used:

```text
2,000 train examples
2,000 test examples
Ridge alpha = 10.0
```

Command shape:

```bash
.venv/bin/python -m ai_unity.evaluate_affine_probes \
  --comparison-json results/<run>/complex_capsules_comparison.json \
  --checkpoint-key best_checkpoint \
  --output-dir results/<run>_heldout_affine_probe_eval \
  --data-dir complex-capsules/data \
  --device auto \
  --seed <seed> \
  --models cnn real-large complex-b \
  --random-scenarios heldout_affine_left_zoom heldout_affine_right_zoom \
  --probe-train-examples 2000 \
  --probe-test-examples 2000
```

### Mean Probe Performance

Mean across four seeds, two held-out scenarios, and six affine factors:

| Feature Block | Mean R2 | Mean Normalized MAE |
|---|---:|---:|
| CNN hidden | 0.128 | 0.672 |
| Real caps | 0.049 | 0.838 |
| ComplexB caps | 0.129 | 0.789 |
| ComplexB phase all | 0.105 | 0.798 |
| ComplexB phase true | 0.012 | 0.857 |
| ComplexB primary phase | 0.277 | 0.688 |

The best compact representation by mean R2 is `ComplexB primary_phase_stats`.

The best representation by normalized MAE is still CNN hidden, but only narrowly:

```text
CNN hidden normalized MAE: 0.672
ComplexB primary phase normalized MAE: 0.688
```

### R2 by Factor

| Factor | CNN Hidden | Real Caps | CxB Caps | CxB Phase All | CxB Phase True | CxB Primary Phase |
|---|---:|---:|---:|---:|---:|---:|
| angle_degrees | -0.005 | 0.047 | 0.078 | 0.050 | 0.024 | 0.162 |
| translate_x_frac | 0.732 | 0.060 | 0.194 | 0.206 | 0.016 | 0.566 |
| translate_y_frac | 0.918 | 0.182 | 0.503 | 0.438 | 0.044 | 0.858 |
| scale | -0.428 | 0.001 | 0.003 | -0.021 | -0.007 | 0.046 |
| shear_x_degrees | -0.186 | 0.006 | 0.007 | -0.019 | -0.001 | 0.028 |
| shear_y_degrees | -0.267 | -0.006 | -0.008 | -0.028 | -0.006 | 0.005 |

Important read:

- CNN hidden activations are excellent translation probes.
- ComplexB primary phase is the best compact probe for angle, scale, and shear, though scale/shear R2 values are still low.
- True-class digit phase alone is weak.
- All-digit phase is much better than true-class phase, which suggests pose information is distributed across class capsules rather than concentrated only in the target class.

### Class-Conditioned Phase Correlations

Class-conditioned correlations are much stronger than the earlier global diagnostic.

| Factor | Mean Class Best Abs Corr | Max |
|---|---:|---:|
| angle_degrees | 0.187 | 0.436 |
| translate_x_frac | 0.182 | 0.508 |
| translate_y_frac | 0.295 | 0.691 |
| scale | 0.073 | 0.190 |
| shear_x_degrees | 0.101 | 0.239 |
| shear_y_degrees | 0.072 | 0.159 |

Strongest class-conditioned cells were mostly vertical translation:

```text
translate_y_frac max abs corr = 0.691
angle_degrees max abs corr = 0.436
translate_x_frac max abs corr = 0.508
```

This confirms the suspicion from section `25`: pooling all digits together hid digit-specific phase structure.

### Interpretation

This is the strongest representation-level evidence so far.

The accuracy story is unchanged:

- CNN wins held-out affine classification.
- ComplexB beats RealCapsuleLarge but does not beat CNN.

The representation story is more favorable to learned phase:

- Compact `primary_phase_stats` are the strongest mean R2 probe block.
- ComplexB phase features contain substantial translation information and some angle information.
- Class-conditioned phase correlations are materially stronger than global correlations.

The current conclusion should be:

> Learned phase is not a raw accuracy advantage over CNN on this benchmark, but it appears to encode affine nuisance factors in a compact and partially class-conditioned way.

The weakest part of the evidence:

- Scale and shear probes remain poor across all representations.
- `digit_phase_true` alone is weak.
- The probe is still synthetic MNIST, not AffNIST.

### Next Step

Move to AffNIST with the current model set and the same probe machinery.

For AffNIST, keep two separate questions:

1. Accuracy: does `ComplexCapsuleB` close the CNN gap on an independent affine dataset?
2. Representation: do `primary_phase_stats` or class-conditioned phase features still predict pose-like factors better than real capsules or CNN hidden activations?

If AffNIST does not expose per-example transform factors, use it first for accuracy, then keep synthetic held-out affine for the pose probes.

## 27. Vision Transformer Baseline

### Implementation

Added `VisionTransformerBaseline`, exposed as model key `vit`.

Architecture:

- image-to-patch embedding with `4x4` patches over `28x28` MNIST, producing `49` patch tokens
- learned CLS token
- learned positional embedding
- `9` Transformer encoder layers
- model width `192`
- `4` attention heads
- feed-forward width `384`
- MLP classification head

Parameter counts:

| Model | Parameters |
|---|---:|
| VisionTransformerBaseline | 2,767,663 |
| BaselineCNN | 2,768,502 |
| RealCapsuleLarge | 2,767,568 |
| ComplexCapsuleB | 2,767,568 |

This is an actual ViT-style transformer baseline at the same parameter budget as the CNN and capsule models.

Validation:

```text
43 passed, 1 skipped
```

### Training

Completed one full seed, `123`, for `20` epochs with the same mild affine augmentation:

```bash
.venv/bin/python complex-capsules/train.py \
  --epochs 20 \
  --device auto \
  --seed 123 \
  --models vit \
  --affine-augment random-affine-mild \
  --amp \
  --output-dir results/complex_mnist_affine_aug_mild_vit_20ep_seed123 \
  --checkpoint-dir checkpoints/complex_mnist_affine_aug_mild_vit_20ep_seed123
```

The first seed took roughly `40-44` seconds per epoch. The full four-seed loop was stopped after seed `123` because this baseline is much slower than the CNN/capsule runs and the one-seed result was already directionally clear.

Best-checkpoint training accuracy for seed `123`:

| Model | Best Accuracy | Best Epoch |
|---|---:|---:|
| CNN | 97.87% | 19 |
| ViT | 94.54% | 19 |
| RealCapsuleLarge | 96.27% | 10 |
| ComplexCapsuleB | 97.04% | 6 |

The unpretrained ViT is not competitive with the CNN or ComplexB as a classifier under this training recipe.

### Standard Random Affine Eval

Seed `123`, best checkpoints, five samples per random scenario:

| Scenario | CNN | ViT | RealLarge | ComplexB | Winner |
|---|---:|---:|---:|---:|---|
| random_translate | 92.71% | 90.50% | 88.12% | 90.99% | CNN |
| random_translate_scale | 97.18% | 93.13% | 94.82% | 95.99% | CNN |
| random_affine_mild | 97.86% | 94.40% | 96.41% | 97.16% | CNN |
| random_affine_strong | 83.88% | 79.89% | 76.87% | 81.71% | CNN |

Overall across these random scenarios:

```text
CNN: 92.909%
ViT: 89.481%
RealCapsuleLarge: 89.056%
ComplexCapsuleB: 91.463%
```

### Fixed Affine Eval

Seed `123`, best checkpoints:

| Scenario | CNN | ViT | RealLarge | ComplexB | Winner |
|---|---:|---:|---:|---:|---|
| clean | 98.86% | 97.08% | 98.41% | 98.70% | CNN |
| translate4 | 97.81% | 95.35% | 95.22% | 96.65% | CNN |
| translate8 | 9.63% | 28.97% | 13.41% | 15.85% | ViT |
| scale085 | 98.12% | 94.61% | 97.42% | 98.01% | CNN |
| scale115 | 98.64% | 96.59% | 98.10% | 98.27% | CNN |
| shear15 | 98.49% | 96.95% | 98.19% | 98.31% | CNN |
| affine_moderate | 91.50% | 84.76% | 87.76% | 90.42% | CNN |
| affine_strong | 20.95% | 30.68% | 24.62% | 27.73% | ViT |

The ViT has a real advantage on the two deterministic extreme cases where the CNN previously failed hardest: `translate8` and `affine_strong`. That advantage does not carry over to the sampled random affine distributions.

### Held-Out Affine Eval

Seed `123`, best checkpoints, five samples per held-out scenario:

| Scenario | CNN | ViT | RealLarge | ComplexB | Winner |
|---|---:|---:|---:|---:|---|
| heldout_affine_left_zoom | 48.34% | 39.94% | 37.58% | 41.95% | CNN |
| heldout_affine_right_zoom | 54.05% | 51.29% | 44.68% | 50.26% | CNN |

Overall held-out accuracy:

```text
CNN: 51.197%
ViT: 45.619%
RealCapsuleLarge: 41.127%
ComplexCapsuleB: 46.105%
```

The ViT is close to ComplexB on held-out average, beats ComplexB on `right_zoom`, and loses to ComplexB on `left_zoom`.

### Probe Results

Seed `123`, held-out affine probes:

| Feature Block | Mean R2 | Mean Normalized MAE |
|---|---:|---:|
| CNN hidden | 0.249 | 0.662 |
| ViT CLS | 0.363 | 0.612 |
| Real caps | 0.055 | 0.835 |
| ComplexB primary phase | 0.273 | 0.690 |

R2 by factor:

| Factor | CNN Hidden | ViT CLS | Real Caps | CxB Primary Phase |
|---|---:|---:|---:|---:|
| angle_degrees | 0.195 | 0.318 | 0.057 | 0.154 |
| translate_x_frac | 0.753 | 0.807 | 0.076 | 0.549 |
| translate_y_frac | 0.922 | 0.958 | 0.184 | 0.858 |
| scale | -0.132 | 0.074 | 0.006 | 0.040 |
| shear_x_degrees | -0.085 | 0.021 | 0.011 | 0.024 |
| shear_y_degrees | -0.160 | -0.002 | -0.006 | 0.011 |

This is the strongest point for the transformer:

> The ViT CLS embedding is the strongest affine-factor probe representation on seed `123`, even though the ViT is not the strongest classifier.

### Interpretation

The actual transformer baseline changes the story in a useful way.

Accuracy:

- CNN remains the best classifier.
- ComplexB remains better than RealCapsuleLarge.
- The unpretrained parameter-matched ViT is worse than CNN and ComplexB on ordinary sampled affine accuracy.
- ViT has pockets of strength on deterministic extreme transforms.

Representation:

- ViT CLS is the best seed-123 probe for affine factors.
- ComplexB primary phase remains a compact useful pose representation, but the transformer representation is stronger on this seed.
- RealCapsuleLarge is still the weakest representation probe.

The fair claim now is:

> Learned phase is a useful capsule-family improvement and a compact pose signal, but modern transformer-style representations can encode affine factors even more strongly. The CNN still wins classification in this small synthetic benchmark.

### Caveat

This is a one-seed transformer result. The earlier CNN/capsule numbers are four-seed aggregates. The ViT is slow enough in this repo that a full four-seed run should be treated as a separate scheduled experiment rather than a quick baseline.

### Next Step

If we want a rigorous transformer comparison, run the remaining ViT seeds with `--resume` and aggregate all four seeds:

```bash
for seed in 321 777 2024; do
  .venv/bin/python complex-capsules/train.py \
    --epochs 20 \
    --device auto \
    --seed "$seed" \
    --models vit \
    --affine-augment random-affine-mild \
    --amp \
    --resume \
    --output-dir "results/complex_mnist_affine_aug_mild_vit_20ep_seed${seed}" \
    --checkpoint-dir "checkpoints/complex_mnist_affine_aug_mild_vit_20ep_seed${seed}"
done
```

Scientifically, the more important move is still AffNIST. The ViT result makes that benchmark more important, because we now have three serious baselines to compare: CNN classifier strength, ViT representation strength, and ComplexB capsule-phase structure.

## 28. AffNIST, Four-Seed ViT, and Residual CNN Validation

### Implementation

Added the independent AffNIST transformed-test evaluator and a stronger conventional CNN baseline.

Code changes:

- `AffNISTDataset` in `ai_unity.data`, using the official transformed test set: `https://www.cs.toronto.edu/~tijmen/affNIST/32x/transformed/test.mat.zip`
- `ai_unity.evaluate_affnist`, supporting multiple comparison JSONs, model subsets, resized `28x28`, and native-image `40x40` transfer evaluation
- checkpoint-compatible native `40x40` image paths:
  - `BaselineCNN` uses adaptive pooling to keep its classifier input shape
  - ViT interpolates patch positional embeddings when the patch grid changes
  - capsule models adaptively pool primary capsule maps to the trained routing shape
- `ResidualCNNBaseline`, exposed as model key `resnet`
- probe support for `resnet_features`

Validation:

```text
55 passed, 1 skipped
```

Important protocol caveat:

> Native `40x40` AffNIST here is a transfer test using checkpoints trained on `28x28` MNIST-style inputs. It is not a fresh canonical `40x40` AffNIST training protocol.

### Four-Seed Training Accuracy

Best checkpoint test accuracy after 20 epochs with mild affine augmentation:

| Model | Mean | Std |
|---|---:|---:|
| ResidualCNNBaseline | 98.905% | 0.068 |
| BaselineCNN | 97.883% | 0.040 |
| ComplexCapsuleB | 97.157% | 0.090 |
| RealCapsuleLarge | 96.280% | 0.135 |
| VisionTransformerBaseline | 94.133% | 0.562 |

The residual CNN is not parameter-matched to the large capsule budget:

| Model | Parameters |
|---|---:|
| BaselineCNN | 2,768,502 |
| VisionTransformerBaseline | 2,767,663 |
| RealCapsuleLarge | 2,767,568 |
| ComplexCapsuleB | 2,767,568 |
| ResidualCNNBaseline | 1,919,178 |

Despite having fewer parameters, the residual CNN is the strongest conventional baseline by a large margin.

### Synthetic Affine Aggregate

Four seeds, best checkpoints, five random samples per scenario.

In-distribution random affine scenarios:

| Model | Mean Accuracy |
|---|---:|
| ResidualCNNBaseline | 98.124% |
| BaselineCNN | 92.409% |
| ComplexCapsuleB | 91.191% |
| VisionTransformerBaseline | 88.853% |
| RealCapsuleLarge | 88.488% |

Held-out affine scenarios:

| Model | Mean Accuracy |
|---|---:|
| ResidualCNNBaseline | 78.358% |
| BaselineCNN | 50.518% |
| VisionTransformerBaseline | 46.411% |
| ComplexCapsuleB | 46.276% |
| RealCapsuleLarge | 41.210% |

Held-out breakdown:

| Scenario | CNN | ResNet | ViT | RealLarge | ComplexB |
|---|---:|---:|---:|---:|---:|
| heldout_affine_left_zoom | 47.075% | 79.329% | 41.663% | 37.334% | 42.721% |
| heldout_affine_right_zoom | 53.960% | 77.386% | 51.158% | 45.087% | 49.832% |

The residual CNN changes the practical conclusion: ordinary convolutional architecture strength, not capsules or transformers, is the dominant factor on these synthetic affine tests.

### AffNIST Accuracy

Official AffNIST transformed test set, four seeds.

Resized `28x28` protocol:

| Model | Mean | Std |
|---|---:|---:|
| ResidualCNNBaseline | 94.733% | 0.960 |
| BaselineCNN | 71.347% | 1.017 |
| VisionTransformerBaseline | 70.503% | 3.006 |
| RealCapsuleLarge | 62.323% | 0.854 |
| ComplexCapsuleB | 51.635% | 2.308 |

Native-image `40x40` transfer protocol:

| Model | Mean | Std |
|---|---:|---:|
| ResidualCNNBaseline | 97.459% | 1.559 |
| BaselineCNN | 74.695% | 0.570 |
| VisionTransformerBaseline | 74.409% | 3.459 |
| RealCapsuleLarge | 51.956% | 0.744 |
| ComplexCapsuleB | 37.309% | 1.855 |

This is the clearest falsification of the strong learned-phase robustness theory:

- On independent AffNIST, `ComplexCapsuleB` does not merely lose to CNN/ViT; it also loses to the real-valued capsule control.
- Native `40x40` transfer makes the capsule gap worse, not better.
- The residual CNN dominates every AffNIST protocol despite having fewer parameters than the parameter-matched CNN/ViT/capsule models.

### Probe Aggregate

Held-out affine probes, four seeds, two held-out scenarios, mean over factors:

| Feature Block | Mean R2 | Mean Normalized MAE |
|---|---:|---:|
| Residual CNN features | 0.399 | 0.606 |
| ViT CLS | 0.356 | 0.616 |
| ComplexB primary phase stats | 0.277 | 0.688 |
| Real capsule digit caps | 0.049 | 0.838 |

The representation story is now also weaker for learned phase. ComplexB primary phase is still a useful compact capsule-family signal, but it is not the best representation probe once stronger conventional baselines are included.

### Current Conclusion

The positive capsule claim should be narrowed substantially:

> Learned phase helps `ComplexCapsuleB` beat an exactly parameter-matched real capsule control on synthetic affine distributions, especially when the training data includes affine variation.

The broader theory is not supported:

> Learned phase does not beat ordinary CNN baselines, does not beat the transformer representation baseline on probes, and fails badly on independent AffNIST compared with CNN, ViT, and even the real capsule control.

The right paper framing is now a negative/diagnostic result:

- report the synthetic capsule-family improvement honestly,
- show how the result disappears or reverses under stronger baselines and AffNIST,
- use the phase probes as mechanistic diagnostics rather than proof of robustness,
- position the work as guidance for future capsule/pose-representation architecture, not as a state-of-the-art robustness claim.

### Next Step

Before writing the paper, run one canonical native AffNIST training protocol if we want to make claims about AffNIST rather than transfer:

- train all image models directly on `40x40` AffNIST-compatible inputs where possible,
- include `ResidualCNNBaseline` as the main conventional baseline,
- keep `BaselineCNN`, `VisionTransformerBaseline`, `RealCapsuleLarge`, and `ComplexCapsuleB`,
- report both accuracy and probe-style diagnostics where transform metadata exists.

If time is limited, write the paper now as a negative result and make canonical `40x40` AffNIST training future work.
