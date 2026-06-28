# AI Unity Findings Summary

**Date:** 2026-06-28  
**Scope:** Complex learned-phase capsules versus real capsule controls, parameter-matched CNN/ViT baselines, a stronger residual CNN baseline, synthetic affine MNIST tests, and AffNIST transfer evaluation.

## Current Best Finding

The strongest result so far is now negative.

The strongest supported positive claim is narrow:

> Under mild affine augmentation, learned-phase `ComplexCapsuleB` learns affine transformation families better than an exactly parameter-matched real capsule control, but a conventional parameter-matched CNN is stronger on sampled synthetic affine distributions.

The important capsule-family comparison remains `ComplexCapsuleB` versus `RealCapsuleLarge`, because both have exactly `2,767,568` parameters. The new `BaselineCNN` has `2,768,502` parameters, within `0.034%` of that budget.

The latest AffNIST and residual-CNN validation substantially weakens the broader theory:

- On independent AffNIST, `ComplexCapsuleB` loses to `RealCapsuleLarge`, CNN, ViT, and residual CNN.
- Native `40x40` AffNIST transfer makes the capsule gap worse.
- A fewer-parameter residual CNN is the strongest model on synthetic affine tests and AffNIST.
- Four-seed probes now put residual CNN features and ViT CLS ahead of `ComplexCapsuleB` primary phase summaries.

The correct paper framing is no longer "complex capsules beat conventional baselines." It is:

> Learned phase creates an interpretable capsule-family improvement on synthetic affine data, but that improvement is not sufficient for robust generalization once stronger baselines and AffNIST are included.

## What We Tested

### 1. Plain MNIST

Plain MNIST is saturated. It is useful for sanity checks, but not decisive.

Across the four-seed targeted comparison, plain MNIST slightly favored `RealCapsuleLarge`:

| Model | Mean Test Accuracy |
|---|---:|
| RealCapsuleLarge | 99.058% |
| ComplexCapsuleB | 98.925% |

Interpretation: no meaningful learned-phase claim should rest on plain MNIST.

### 2. Fixed Rotated MNIST

Clean-trained checkpoints showed a small, angle-specific learned-phase signal.

With seeds `123`, `321`, `777`, and `2024`, `ComplexCapsuleB` beat `RealCapsuleLarge` most clearly at moderate deterministic rotations:

| Rotation | ComplexB - RealLarge |
|---:|---:|
| 30 degrees | +1.400 points |
| 45 degrees | +1.700 points |

But the broader fixed-rotation result was fragile:

```text
Nonzero fixed rotations including 75 degrees:
ComplexCapsuleB - RealCapsuleLarge = +0.503 points
```

Random rotation ranges mostly collapsed the advantage:

```text
Random rotation ranges:
ComplexCapsuleB - RealCapsuleLarge = +0.079 points
```

Interpretation: clean-trained learned phase does not provide broad rotation invariance.

### 3. Clean-Trained Synthetic Affine-MNIST Evaluation

Clean-trained checkpoints were then evaluated on synthetic affine distortions: translation, scale, shear, and combined affine transforms.

Fixed affine scenarios showed pockets of advantage:

```text
Fixed OOD affine scenarios:
ComplexCapsuleB - RealCapsuleLarge = +2.404 points
```

But random affine distributions favored the real control slightly:

```text
Random affine scenarios:
ComplexCapsuleB - RealCapsuleLarge = -0.699 points
```

Interpretation: learned phase sometimes fails less badly on hand-picked deterministic transforms, but still does not give broad out-of-distribution robustness from clean training.

### 4. Mild Affine-Augmented Training

The important shift came after training both models with the same mild affine augmentation:

- rotation `[-30, 30]`
- translation up to `15%`
- scale `[0.85, 1.15]`
- x-shear `[-15, 15]`

Initial final-checkpoint result:

```text
Affine-augmented random affine eval:
ComplexCapsuleB - RealCapsuleLarge = +2.577 points
```

This was the first broad distributional result favoring `ComplexCapsuleB`.

### 5. Durable Rerun With Best Checkpoints

The trainer was improved to support:

- per-epoch history flushing,
- `_latest.pt` checkpoints,
- `_best.pt` checkpoints,
- `--resume`,
- evaluator `--checkpoint-key best_checkpoint`.

Then the affine-augmented experiment was rerun over four seeds and evaluated using best checkpoints with five random samples per scenario.

Best checkpoint training metric:

```text
ComplexCapsuleB - RealCapsuleLarge = +0.937 points, std 0.179
```

Best checkpoint random affine eval across `80` cells:

```text
ComplexCapsuleB - RealCapsuleLarge = +2.662 points, std 1.733
```

Random affine breakdown:

| Scenario | ComplexB - RealLarge |
|---|---:|
| random_translate | +3.113 points |
| random_translate_scale | +1.475 points |
| random_affine_mild | +1.033 points |
| random_affine_strong | +5.028 points |

Every seed had a positive mean advantage in the random affine aggregate:

| Seed | Mean Advantage |
|---:|---:|
| 123 | +2.819 |
| 321 | +2.707 |
| 777 | +1.731 |
| 2024 | +3.392 |

### 6. Parameter-Matched CNN Baseline

The next phase added `BaselineCNN`, exposed as model key `cnn`, and reran four seeds with the same mild affine augmentation, durable checkpoints, and best-checkpoint affine evaluator.

Validation:

```text
37 passed, 1 skipped
```

Best checkpoint training accuracy:

| Model | Mean | Std |
|---|---:|---:|
| BaselineCNN | 97.883% | 0.046 |
| RealCapsuleLarge | 96.280% | 0.156 |
| ComplexCapsuleB | 97.157% | 0.104 |

Fixed affine eval across four seeds:

```text
ComplexCapsuleB - RealCapsuleLarge = +1.727 points
CNN - ComplexCapsuleB = -1.557 points
```

Random affine eval across `80` seed/sample/scenario cells:

```text
ComplexCapsuleB - RealCapsuleLarge = +2.703 points
CNN - ComplexCapsuleB = +1.218 points
```

Random affine breakdown:

| Scenario | CNN | RealLarge | ComplexB | Winner |
|---|---:|---:|---:|---|
| random_translate | 91.81% | 87.36% | 90.52% | CNN |
| random_translate_scale | 96.85% | 94.38% | 95.82% | CNN |
| random_affine_mild | 97.76% | 96.13% | 97.03% | CNN |
| random_affine_strong | 83.22% | 76.09% | 81.39% | CNN |

Interpretation: `ComplexCapsuleB` still beats the exact real capsule control, but the conventional CNN is the stronger general sampled-affine baseline.

### 7. Held-Out Affine Generator and Phase Diagnostics

The evaluator now includes two held-out random affine scenarios outside the mild training ranges:

| Scenario | Rotation | Scale | Shear |
|---|---:|---:|---:|
| heldout_affine_left_zoom | `[-60, -30]` | `[0.70, 0.85]` | negative x/y shear |
| heldout_affine_right_zoom | `[30, 60]` | `[1.15, 1.35]` | positive x/y shear |

The affine dataset also exposes per-example transform parameters, and the evaluator supports:

```bash
--phase-diagnostics
```

This writes `complex_affine_phase_diagnostics.json` with correlations between affine factors and `sin`/`cos` projections of true-class `digit_phase`.

Validation:

```text
38 passed, 1 skipped
```

Held-out accuracy across `40` seed/sample/scenario cells:

| Model | Mean | Std |
|---|---:|---:|
| BaselineCNN | 50.518% | 3.687 |
| RealCapsuleLarge | 41.210% | 4.016 |
| ComplexCapsuleB | 46.276% | 3.736 |

Aggregate margins:

```text
ComplexCapsuleB - RealCapsuleLarge = +5.066 points
CNN - ComplexCapsuleB = +4.241 points
```

Scenario breakdown:

| Scenario | CNN | RealLarge | ComplexB | Winner |
|---|---:|---:|---:|---|
| heldout_affine_left_zoom | 47.08% | 37.33% | 42.72% | CNN |
| heldout_affine_right_zoom | 53.96% | 45.09% | 49.83% | CNN |

Phase diagnostics:

```text
ComplexB phase_std mean = 1.777
ComplexB phase_std min/max = 1.655 / 1.932
```

Best absolute phase-factor correlations were weak to modest:

| Factor | Mean Best Abs Corr | Max |
|---|---:|---:|
| angle_degrees | 0.084 | 0.171 |
| translate_y_frac | 0.079 | 0.139 |
| translate_x_frac | 0.057 | 0.098 |
| shear_x_degrees | 0.039 | 0.067 |
| scale | 0.025 | 0.054 |
| shear_y_degrees | 0.024 | 0.058 |

Interpretation: phase remains high-variance and non-collapsed, but simple global Pearson correlations do not yet prove clean pose encoding.

### 8. Held-Out Affine Linear Probes

Added `ai_unity.evaluate_affine_probes`.

The probe evaluator extracts fixed representations from best checkpoints, trains Ridge probes on `2,000` held-out affine examples, and tests on another `2,000` examples per scenario.

Feature blocks:

| Model | Feature Block | Dims |
|---|---|---:|
| BaselineCNN | `cnn_penultimate` | 382 |
| RealCapsuleLarge | `digit_caps` | 80 |
| ComplexCapsuleB | `digit_caps` | 240 |
| ComplexCapsuleB | `digit_phase_all` | 160 |
| ComplexCapsuleB | `digit_phase_true` | 16 |
| ComplexCapsuleB | `primary_phase_stats` | 32 |

Validation:

```text
40 passed, 1 skipped
```

Mean probe performance across four seeds, two held-out scenarios, and six affine factors:

| Feature Block | Mean R2 | Mean Normalized MAE |
|---|---:|---:|
| CNN hidden | 0.128 | 0.672 |
| Real caps | 0.049 | 0.838 |
| ComplexB caps | 0.129 | 0.789 |
| ComplexB phase all | 0.105 | 0.798 |
| ComplexB phase true | 0.012 | 0.857 |
| ComplexB primary phase | 0.277 | 0.688 |

R2 by factor:

| Factor | CNN Hidden | CxB Primary Phase |
|---|---:|---:|
| angle_degrees | -0.005 | 0.162 |
| translate_x_frac | 0.732 | 0.566 |
| translate_y_frac | 0.918 | 0.858 |
| scale | -0.428 | 0.046 |
| shear_x_degrees | -0.186 | 0.028 |
| shear_y_degrees | -0.267 | 0.005 |

Class-conditioned phase correlations were stronger than global correlations:

| Factor | Mean Class Best Abs Corr | Max |
|---|---:|---:|
| angle_degrees | 0.187 | 0.436 |
| translate_x_frac | 0.182 | 0.508 |
| translate_y_frac | 0.295 | 0.691 |
| scale | 0.073 | 0.190 |
| shear_x_degrees | 0.101 | 0.239 |
| shear_y_degrees | 0.072 | 0.159 |

Interpretation: learned phase is not a raw accuracy advantage over CNN on this benchmark, but `primary_phase` appears to encode affine nuisance factors in a compact and partially class-conditioned way.

### 9. Vision Transformer Baseline

Added `VisionTransformerBaseline`, exposed as model key `vit`.

Parameter counts:

| Model | Parameters |
|---|---:|
| VisionTransformerBaseline | 2,767,663 |
| BaselineCNN | 2,768,502 |
| RealCapsuleLarge | 2,767,568 |
| ComplexCapsuleB | 2,767,568 |

Validation:

```text
43 passed, 1 skipped
```

Completed one full seed, `123`, for `20` epochs with the same mild affine augmentation. The full four-seed loop was stopped after seed `123` because ViT training is much slower in this setup, roughly `40-44` seconds per epoch.

Seed `123` best-checkpoint training accuracy:

| Model | Best Accuracy | Best Epoch |
|---|---:|---:|
| BaselineCNN | 97.87% | 19 |
| VisionTransformerBaseline | 94.54% | 19 |
| RealCapsuleLarge | 96.27% | 10 |
| ComplexCapsuleB | 97.04% | 6 |

Seed `123` random affine eval, averaged over four random scenarios:

| Model | Mean Accuracy |
|---|---:|
| BaselineCNN | 92.909% |
| VisionTransformerBaseline | 89.481% |
| RealCapsuleLarge | 89.056% |
| ComplexCapsuleB | 91.463% |

Seed `123` held-out affine eval:

| Model | Mean Accuracy |
|---|---:|
| BaselineCNN | 51.197% |
| VisionTransformerBaseline | 45.619% |
| RealCapsuleLarge | 41.127% |
| ComplexCapsuleB | 46.105% |

The ViT does win two deterministic fixed affine cases:

| Scenario | CNN | ViT | RealLarge | ComplexB |
|---|---:|---:|---:|---:|
| translate8 | 9.63% | 28.97% | 13.41% | 15.85% |
| affine_strong | 20.95% | 30.68% | 24.62% | 27.73% |

But that advantage does not carry over to sampled random affine distributions.

Seed `123` held-out affine probe:

| Feature Block | Mean R2 | Mean Normalized MAE |
|---|---:|---:|
| CNN hidden | 0.249 | 0.662 |
| ViT CLS | 0.363 | 0.612 |
| Real caps | 0.055 | 0.835 |
| ComplexB primary phase | 0.273 | 0.690 |

Interpretation: the parameter-matched unpretrained ViT is not the best classifier, but its CLS representation is the strongest affine-factor probe on seed `123`.

## 2026-06-28 Validation Update

Latest verification:

```text
55 passed, 1 skipped
```

### Four-Seed Training Accuracy

Best checkpoint test accuracy after 20 epochs with mild affine augmentation:

| Model | Mean | Std |
|---|---:|---:|
| ResidualCNNBaseline | 98.905% | 0.068 |
| BaselineCNN | 97.883% | 0.040 |
| ComplexCapsuleB | 97.157% | 0.090 |
| RealCapsuleLarge | 96.280% | 0.135 |
| VisionTransformerBaseline | 94.133% | 0.562 |

### Synthetic Affine Accuracy

Four seeds, best checkpoints, five random samples per scenario:

| Evaluation Group | CNN | ResNet | ViT | RealLarge | ComplexB |
|---|---:|---:|---:|---:|---:|
| Random affine, in distribution | 92.409% | 98.124% | 88.853% | 88.488% | 91.191% |
| Held-out affine | 50.518% | 78.358% | 46.411% | 41.210% | 46.276% |

`ComplexCapsuleB` still beats `RealCapsuleLarge`, but only inside the capsule-family comparison. The residual CNN is far stronger than every capsule and transformer result on this benchmark.

### AffNIST Accuracy

Official AffNIST transformed test set:

| Protocol | CNN | ResNet | ViT | RealLarge | ComplexB |
|---|---:|---:|---:|---:|---:|
| Resized `28x28` | 71.347% | 94.733% | 70.503% | 62.323% | 51.635% |
| Native `40x40` transfer | 74.695% | 97.459% | 74.409% | 51.956% | 37.309% |

AffNIST is the decisive negative result:

- `ComplexCapsuleB` loses to `RealCapsuleLarge`.
- CNN and ViT are far ahead of both capsule models.
- `ResidualCNNBaseline` dominates all other models, despite having fewer parameters than the matched CNN/ViT/capsule models.

The native `40x40` result is a checkpoint-compatible transfer protocol, not canonical `40x40` training.

### Probe Aggregate

Held-out affine linear probes, four seeds, mean over two held-out scenarios and affine factors:

| Feature Block | Mean R2 | Mean Normalized MAE |
|---|---:|---:|
| Residual CNN features | 0.399 | 0.606 |
| ViT CLS | 0.356 | 0.616 |
| ComplexB primary phase stats | 0.277 | 0.688 |
| Real capsule digit caps | 0.049 | 0.838 |

The earlier phase-probe story still exists, but it is no longer the strongest representation result once residual CNN and four-seed ViT probes are included.

## Current Interpretation

Learned phase is not magic invariance.

Clean-trained `ComplexCapsuleB` does not robustly generalize to random rotations or random affine transforms. That stronger claim should be rejected for now.

The supported claim is narrower:

> Learned phase appears to help a capsule model learn transformation families when the model is trained on transformation variation, but that capsule-family improvement is not enough to beat a conventional CNN on this synthetic sampled-affine benchmark.

This is a better research lead than the original broad invariance story because it survived:

- exact parameter matching,
- four seeds,
- best-checkpoint selection,
- repeated random affine evaluation samples,
- comparison against a real-valued capsule with the same parameter count,
- and a CNN sanity baseline that prevents overclaiming.

The held-out generator adds one important nuance:

> The harder the affine shift gets, the more `ComplexCapsuleB` separates from `RealCapsuleLarge`, but the CNN still remains ahead on accuracy.

The probe result adds a second nuance:

> Learned phase is more promising as a compact pose representation than as an immediate CNN-beating classifier.

The ViT and residual CNN results add a third nuance:

> Transformer-style and residual-CNN representations encode affine factors more strongly than the current learned-phase capsule representation.

## Main Caveats

- The current benchmark is still MNIST-derived and synthetic.
- The random affine evaluator uses our own transform generator, not an independent dataset.
- The CNN baseline wins the sampled affine distributions, and the residual CNN wins by much more.
- Results are promising for transformation learning, not for general image recognition.
- `ComplexCapsuleB` is learned phase, not explicit coordinate-derived spatial phase.
- Probe results are representation-level evidence, not classification evidence.
- The native `40x40` AffNIST result is transfer evaluation from `28x28` training, not canonical native AffNIST training.

## Key Artifacts

Durable affine-augmented training outputs:

- `results/complex_mnist_affine_aug_mild_durable_20ep_seed123/`
- `results/complex_mnist_affine_aug_mild_durable_20ep_seed321/`
- `results/complex_mnist_affine_aug_mild_durable_20ep_seed777/`
- `results/complex_mnist_affine_aug_mild_durable_20ep_seed2024/`

Best-checkpoint affine eval outputs:

- `results/complex_mnist_affine_aug_mild_durable_20ep_seed123_best_affine_eval_5samples/`
- `results/complex_mnist_affine_aug_mild_durable_20ep_seed321_best_affine_eval_5samples/`
- `results/complex_mnist_affine_aug_mild_durable_20ep_seed777_best_affine_eval_5samples/`
- `results/complex_mnist_affine_aug_mild_durable_20ep_seed2024_best_affine_eval_5samples/`

CNN baseline training outputs:

- `results/complex_mnist_affine_aug_mild_cnn_20ep_seed123/`
- `results/complex_mnist_affine_aug_mild_cnn_20ep_seed321/`
- `results/complex_mnist_affine_aug_mild_cnn_20ep_seed777/`
- `results/complex_mnist_affine_aug_mild_cnn_20ep_seed2024/`

CNN baseline best-checkpoint affine eval outputs:

- `results/complex_mnist_affine_aug_mild_cnn_20ep_seed123_best_affine_eval_5samples/`
- `results/complex_mnist_affine_aug_mild_cnn_20ep_seed321_best_affine_eval_5samples/`
- `results/complex_mnist_affine_aug_mild_cnn_20ep_seed777_best_affine_eval_5samples/`
- `results/complex_mnist_affine_aug_mild_cnn_20ep_seed2024_best_affine_eval_5samples/`

Held-out affine + phase diagnostic outputs:

- `results/complex_mnist_affine_aug_mild_cnn_20ep_seed123_heldout_affine_phase_eval_5samples/`
- `results/complex_mnist_affine_aug_mild_cnn_20ep_seed321_heldout_affine_phase_eval_5samples/`
- `results/complex_mnist_affine_aug_mild_cnn_20ep_seed777_heldout_affine_phase_eval_5samples/`
- `results/complex_mnist_affine_aug_mild_cnn_20ep_seed2024_heldout_affine_phase_eval_5samples/`

Held-out affine probe outputs:

- `results/complex_mnist_affine_aug_mild_cnn_20ep_seed123_heldout_affine_probe_eval/`
- `results/complex_mnist_affine_aug_mild_cnn_20ep_seed321_heldout_affine_probe_eval/`
- `results/complex_mnist_affine_aug_mild_cnn_20ep_seed777_heldout_affine_probe_eval/`
- `results/complex_mnist_affine_aug_mild_cnn_20ep_seed2024_heldout_affine_probe_eval/`

ViT outputs:

- `results/complex_mnist_affine_aug_mild_vit_20ep_seed123/`
- `results/complex_mnist_affine_aug_mild_vit_20ep_seed321/`
- `results/complex_mnist_affine_aug_mild_vit_20ep_seed777/`
- `results/complex_mnist_affine_aug_mild_vit_20ep_seed2024/`
- `results/complex_mnist_affine_aug_mild_vit_20ep_seed123_best_affine_eval_5samples/`
- `results/complex_mnist_affine_aug_mild_vit_20ep_seed321_best_affine_eval_5samples/`
- `results/complex_mnist_affine_aug_mild_vit_20ep_seed777_best_affine_eval_5samples/`
- `results/complex_mnist_affine_aug_mild_vit_20ep_seed2024_best_affine_eval_5samples/`
- `results/complex_mnist_affine_aug_mild_vit_20ep_seed123_heldout_affine_probe_eval/`
- `results/complex_mnist_affine_aug_mild_vit_20ep_seed321_heldout_affine_probe_eval/`
- `results/complex_mnist_affine_aug_mild_vit_20ep_seed777_heldout_affine_probe_eval/`
- `results/complex_mnist_affine_aug_mild_vit_20ep_seed2024_heldout_affine_probe_eval/`

Residual CNN outputs:

- `results/complex_mnist_affine_aug_mild_resnet_20ep_seed123/`
- `results/complex_mnist_affine_aug_mild_resnet_20ep_seed321/`
- `results/complex_mnist_affine_aug_mild_resnet_20ep_seed777/`
- `results/complex_mnist_affine_aug_mild_resnet_20ep_seed2024/`
- `results/complex_mnist_affine_aug_mild_resnet_20ep_seed123_best_affine_eval_5samples/`
- `results/complex_mnist_affine_aug_mild_resnet_20ep_seed321_best_affine_eval_5samples/`
- `results/complex_mnist_affine_aug_mild_resnet_20ep_seed777_best_affine_eval_5samples/`
- `results/complex_mnist_affine_aug_mild_resnet_20ep_seed2024_best_affine_eval_5samples/`

AffNIST outputs:

- `results/affnist_resized28_seed123_full_eval/`
- `results/affnist_resized28_seed321_full_eval/`
- `results/affnist_resized28_seed777_full_eval/`
- `results/affnist_resized28_seed2024_full_eval/`
- `results/affnist_native40_seed123_full_eval/`
- `results/affnist_native40_seed321_full_eval/`
- `results/affnist_native40_seed777_full_eval/`
- `results/affnist_native40_seed2024_full_eval/`
- `results/affnist_resized28_resnet_seed123_full_eval/`
- `results/affnist_native40_resnet_seed123_full_eval/`
- matching `seed321`, `seed777`, and `seed2024` residual AffNIST directories

Main notebook:

- `UNITY_NOTEBOOK.md`

## Next Step

The next step is paper framing, not another quick baseline.

Write this as a negative/diagnostic result:

1. State the original hypothesis: learned complex phase might improve transformation robustness.
2. Show the supported positive result: `ComplexCapsuleB` beats the exactly parameter-matched `RealCapsuleLarge` on synthetic affine distributions.
3. Show the falsifying evidence: CNN, ViT, and especially residual CNN baselines beat `ComplexCapsuleB`; AffNIST reverses even the capsule-family advantage.
4. Use probes as mechanism analysis, not as proof of robustness.
5. Put canonical native `40x40` AffNIST training, dSprites, and SmallNORB in future work unless we want one more full experimental phase.

If one more experiment is required before writing, run canonical native `40x40` AffNIST training for all image models. Otherwise, start the paper now.
