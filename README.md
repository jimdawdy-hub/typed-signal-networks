# Typed Signal Networks

Typed Signal Networks is a small research codebase built around a simple question: can a neural signal carry more than magnitude?

The idea behind the project was to ask whether a signal could also carry identity, phase, or type, so the next layer would treat the same value differently depending on what kind of signal it was. That line of thinking led to capsule networks, complex-valued activations, and routing mechanisms that preserve more structure than a standard scalar pipeline.

The repository tests that idea in a controlled way. The core experiments compare learned-phase complex capsules against real-valued capsule controls, parameter-matched CNN and ViT baselines, and a stronger residual CNN baseline on synthetic affine MNIST and AffNIST. The result is nuanced: learned phase helps within the capsule family, but stronger conventional models do better overall, and the broad robustness claim does not hold up. That makes the project a diagnostic study of what learned phase can and cannot do.

## arXiv Endorsement Request (cs.LG)

The following is formatted for direct pasting into an arXiv endorsement request.

```text
I am seeking endorsement for arXiv category cs.LG (Machine Learning) for the following manuscript.

Title: When Learned Phase Is Not Enough: A Diagnostic Evaluation of Typed Neural Signals for Affine Robustness
Author: James Dawdy

Abstract:
Capsule networks are motivated by the idea that structured activations can represent pose more explicitly than scalar features. I test whether learned complex phase in a capsule network improves affine robustness, comparing a learned-phase complex capsule, ComplexCapsuleB, against a parameter-matched real capsule control, a parameter-matched CNN and Vision Transformer, and a stronger residual CNN. An earlier control, RealCapsuleLarge, matched the complex model’s parameter count but had two confounds: its extra parameters were functionally degenerate, and its readout had a smaller dynamic range under cross-entropy. I replace it with RealCapsuleControlV2, matched in count, width, and readout form, and isolate each confound with single-fix ablations. The confounds explain roughly half the originally reported held-out gap, but a smaller, seed-consistent effect remains on synthetic affine data. That improvement does not carry over to stronger baselines: a parameter-matched CNN beats the complex capsule on sampled affine tests, and a smaller residual CNN dominates every synthetic and AffNIST evaluation. On independent AffNIST, ComplexCapsuleB loses to the corrected control. Linear probes show that learned phase carries compact affine information, but residual CNN and ViT features encode affine factors more strongly. The result is narrow but useful: learned phase improves a capsule model relative to a carefully matched real control, yet the current architecture is not competitive with strong conventional models for affine robustness.

The work is an empirical machine-learning study of neural-network representations, controlled robustness evaluation, baseline design, and negative results. The repository contains the implementation, tests, experiment logs, and manuscript source.
```

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

## Foundations

This project is built on ideas that came from earlier work:

- Capsules and routing by agreement: Hinton et al. on transforming auto-encoders, then Sabour, Frosst, and Hinton on dynamic routing, and Hinton et al. on matrix capsules.
- Mixture of experts: Jacobs, Jordan, Nowlan, and Hinton on adaptive mixtures of local experts, then Shazeer et al. on sparsely-gated MoE.
- Ternary weights: Li et al. on ternary weight networks.
- Complex-valued networks: Trabelsi et al. on deep complex networks.
- Residual networks: He et al. on deep residual learning.
- Vision Transformers: Dosovitskiy et al. on ViT.

Those ideas are cited in the paper draft and are the intellectual backdrop for the experiments here.

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
