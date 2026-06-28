# AI Unity

AI Unity started from a simple but specific question: can a neural signal carry more than just magnitude?

In ordinary neural nets, a value is mostly just a number. The idea behind this project was to ask whether the signal itself could carry an additional meaningful state, like identity, phase, or type, so that the next layer could treat
the same value differently depending on what kind of signal it was. That line of thinking led to capsule networks, complex-valued activations, and routing mechanisms that try to preserve more structure than a standard scalar pipeline.

This repository tests that idea in a controlled way. The core experiments compare learned-phase complex capsules against real-valued capsule controls, parameter-matched CNN and ViT baselines, and a stronger residual CNN baseline on
synthetic affine MNIST and AffNIST. The result is nuanced: learned phase does help within the capsule family, but stronger conventional models do better overall, and the broad robustness claim does not hold up. That makes the project
less a victory lap and more a clean diagnostic study of what learned phase can and cannot do.

The work started with trying to determine if learned phase in a capsule network actually make a vision model more robust to geometric change?

The short answer from these experiments is: a little, but not enough.

`ComplexCapsuleB`, the learned-phase capsule model, does beat an exactly parameter-matched real capsule control on some synthetic affine tests. That is the useful signal in this repo. But stronger conventional baselines beat it, and an independent AffNIST evaluation reverses even the capsule-family advantage. So this repository is mainly a controlled negative result, not a claim of state-of-the-art robustness.

## What’s Here

- Capsule models with real and complex-valued routing
- A parameter-matched CNN baseline
- A parameter-matched Vision Transformer baseline
- A stronger residual CNN baseline
- Training and evaluation scripts for MNIST-derived affine experiments
- AffNIST evaluation support
- Probe scripts that ask whether learned phase actually stores pose-like information
- Experiment logs, summaries, and a first paper draft

## What The Experiments Say

At a high level:

1. `ComplexCapsuleB` is better than `RealCapsuleLarge` on synthetic affine data.
2. A normal CNN does better than `ComplexCapsuleB` on sampled synthetic affine tests.
3. A residual CNN does better than everything else on both synthetic affine tests and AffNIST.
4. AffNIST shows that the complex capsule is not the robust winner we hoped for.
5. Learned phase still seems to encode some affine information, but not enough to beat stronger models.

If you want the detailed numbers, start with [FINDINGS_SUMMARY.md](FINDINGS_SUMMARY.md) and then read the latest notebook section in [UNITY_NOTEBOOK.md](UNITY_NOTEBOOK.md).

## Repo Layout

- `ai_unity/` - shared training, evaluation, and model code
- `complex-capsules/` - the capsule experiment entry points
- `ternary-moe/` - the ternary/MoE experiment entry points
- `tests/` - smoke tests and parser coverage
- `results/` - saved logs, metrics, plots, and evaluation JSON/CSV outputs
- `PAPER_DRAFT.md` - first-pass manuscript draft
- `UNITY_NOTEBOOK.md` - running experiment notes
- `FINDINGS_SUMMARY.md` - distilled conclusions and tables

## Data Policy

The repository does not include datasets or checkpoints.

- Datasets are expected to be downloaded locally when needed.
- If derived datasets need to be shared, a separate Hugging Face dataset or equivalent is a better place for them than this repo.
- Checkpoints are intentionally left out of the public repository.

## Environment

This codebase was developed with Python 3.12 and PyTorch on CUDA-enabled hardware.

Create a local environment and install dependencies:

```bash
python3.12 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Smoke Tests

Run the test suite:

```bash
source .venv/bin/activate
pytest
```

The tests cover:

- model forward/backward smoke checks
- ternary weight constraints
- router and phase diagnostics
- training resume and logging
- AffNIST parser and dataset loading

## Training

The main training entry points are:

```bash
CUDA_VISIBLE_DEVICES=0 python complex-capsules/train.py --epochs 5 --device auto --output-dir results/complex
CUDA_VISIBLE_DEVICES=0 python ternary-moe/train.py --epochs 5 --device auto --output-dir results/ternary
```

Useful flags:

- `--device auto|cpu|cuda`
- `--gpu-index 0`
- `--seed 123`
- `--amp`
- `--compile`
- `--limit-train-batches`
- `--limit-test-batches`
- `--train-subset`
- `--test-subset`

The training harness writes history CSVs, confusion matrices, checkpoints, and accuracy plots when enabled.

## Evaluation

Relevant evaluators:

- `ai_unity.evaluate_complex_rotations`
- `ai_unity.evaluate_complex_affines`
- `ai_unity.evaluate_affine_probes`
- `ai_unity.evaluate_affnist`

The current paper draft argues for a cautious interpretation:

> Learned phase is a useful capsule-family mechanism, but it is not enough on its own to beat stronger conventional architectures on affine robustness.

## Current Paper State

The first draft is in [PAPER_DRAFT.md](PAPER_DRAFT.md).

The strongest result is negative:

- synthetic affine gains exist within the capsule family
- AffNIST and stronger baselines do not support a broad robustness claim
- residual CNNs are the strongest model family in this repo’s current experiments

That is the version worth writing up.
