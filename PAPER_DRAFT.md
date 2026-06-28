# When Learned Phase Is Not Enough: A Diagnostic Evaluation of Typed Neural Signals for Affine Robustness

First draft. Working paper, 2026-06-28.

## Abstract

Capsule networks are motivated by the idea that vector-valued or structured activations can represent object pose more explicitly than scalar features. This paper tests a narrow version of that idea: whether a learned complex phase inside a capsule network improves robustness to rotations, translations, scale changes, and shear. We compare a learned-phase complex capsule model, `ComplexCapsuleB`, against an exactly parameter-matched real capsule control, a parameter-matched CNN, a parameter-matched Vision Transformer baseline, and a stronger residual CNN baseline. Across four seeds on synthetic affine MNIST, learned phase produces a real capsule-family improvement: `ComplexCapsuleB` outperforms the exact real capsule control on random affine distributions and held-out affine shifts. However, this improvement is not sufficient to beat conventional baselines. A parameter-matched CNN exceeds the complex capsule on sampled synthetic affine tests, and a fewer-parameter residual CNN dominates all models on synthetic affine tests and AffNIST. On independent AffNIST evaluation, `ComplexCapsuleB` loses not only to CNN and ViT baselines, but also to the real capsule control. Linear probes show that learned phase contains compact pose-related information, but residual CNN features and ViT CLS embeddings encode affine factors more strongly. The result is therefore negative but useful: learned complex phase can improve a capsule model relative to a real-valued capsule control, yet current learned-phase capsules do not provide competitive transformation robustness against strong conventional architectures.

## 1. Introduction

Deep image classifiers often succeed by learning features that are tolerant to nuisance transformations. Convolutional networks build in translation bias through weight sharing and local receptive fields, while modern residual architectures and transformer-style models can learn broad invariances from data and augmentation. Capsule networks offer a different promise: instead of hiding pose variation inside high-dimensional scalar feature maps, capsules attempt to represent object identity and pose in structured activations.

This project began with a stronger hypothesis: if capsule activations are represented as complex numbers, then phase might naturally carry pose-like information. Rotations and other geometric changes often have an angular structure, so a learned phase channel could, in principle, give capsule routing a better coordinate system for transformation robustness.

The experiments below test that hypothesis directly. The answer is mixed, and the most important result is negative. Learned phase does help within the capsule family: after mild affine augmentation, `ComplexCapsuleB` consistently beats an exactly parameter-matched real capsule control on synthetic affine distributions. But the result does not survive stronger baselines. A parameter-matched CNN is better on sampled affine tests, a parameter-matched ViT gives stronger representation probes, and a smaller residual CNN dominates both synthetic affine tests and AffNIST. On AffNIST, the complex capsule also loses to the real capsule control.

The main contribution is therefore not a new state-of-the-art robust classifier. It is a controlled diagnostic study showing where a plausible learned-phase capsule idea works, where it fails, and which evidence should prevent overclaiming.

We make four contributions:

1. We introduce and evaluate a learned-phase complex capsule model against an exactly parameter-matched real capsule control.
2. We evaluate robustness across fixed rotations, synthetic affine transformations, held-out affine distributions, and the AffNIST transformed test set.
3. We compare against parameter-matched CNN and ViT baselines, plus a stronger residual CNN baseline with fewer parameters.
4. We use linear probes to ask whether learned phase carries pose-related information even when it fails to win classification accuracy.

## 2. Background and Hypothesis

Capsule networks represent entities with vectors rather than scalar activations, and use routing procedures to decide how lower-level capsules contribute to higher-level capsules. The motivating claim is that capsule dimensions can encode pose-like degrees of freedom while capsule length or norm encodes class evidence.

Complex-valued representations add another possible structure: magnitude can represent evidence strength, while phase can represent a periodic or angular variable. This suggests a natural hypothesis for geometric robustness:

> A capsule model with learned complex phase should better represent affine nuisance factors, improving robustness to transformations relative to a real-valued capsule with the same parameter budget.

This paper separates three claims that are easy to conflate:

1. Capsule-family claim: learned phase improves over a real capsule control.
2. Baseline claim: learned phase beats conventional CNN or transformer baselines.
3. Representation claim: learned phase encodes pose or affine factors in a probe-detectable way.

The experiments support the first claim on synthetic affine data, partially support the third claim, and reject the second claim.

Relevant prior work:

- Hinton, Krizhevsky, and Wang, "Transforming Auto-Encoders", 2011.
- Sabour, Frosst, and Hinton, "Dynamic Routing Between Capsules", 2017.
- Hinton, Sabour, and Frosst, "Matrix Capsules with EM Routing", 2018.
- Jacobs, Jordan, Nowlan, and Hinton, "Adaptive Mixtures of Local Experts", 1991.
- Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer", 2017.
- Li et al., "Ternary Weight Networks", 2016.
- Trabelsi et al., "Deep Complex Networks", 2018.
- He et al., "Deep Residual Learning for Image Recognition", 2016.
- Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", 2021.

## 3. Models

All models are trained and evaluated on grayscale digit images. The core comparison uses four seeds: `123`, `321`, `777`, and `2024`.

### 3.1 Real Capsule Control

`RealCapsuleLarge` is the real-valued capsule control. It uses:

- a `9x9` convolutional stem with 256 channels,
- two real primary-capsule convolution banks,
- two real transformation banks,
- 3 routing iterations,
- 10 class capsules of dimension 8.

It has `2,767,568` trainable parameters.

### 3.2 Learned-Phase Complex Capsule

`ComplexCapsuleB` is the learned-phase model. It uses:

- the same initial convolutional stem as the real capsule,
- one primary magnitude convolution bank,
- one primary phase convolution bank,
- complex transformation matrices built from real and imaginary parameter banks,
- complex squash and complex agreement during routing,
- class evidence from the magnitude of final complex class capsules.

Magnitude is passed through `softplus`; phase is learned directly. The model returns `digit_caps`, `primary_phase`, and `digit_phase` for diagnostics. It has exactly `2,767,568` parameters, matching `RealCapsuleLarge`.

### 3.3 Parameter-Matched CNN

`BaselineCNN` is a conventional convolutional baseline with:

- convolution layers with 64 and 128 channels,
- max pooling,
- adaptive average pooling to `7x7`,
- a three-layer MLP classifier head.

It has `2,768,502` parameters, within `0.034%` of the large capsule parameter budget.

### 3.4 Parameter-Matched Vision Transformer

`VisionTransformerBaseline` is a small ViT-style model:

- `4x4` patch embedding,
- learned CLS token,
- learned positional embedding,
- 9 transformer encoder layers,
- width 192,
- 4 attention heads,
- feed-forward width 384.

It has `2,767,663` parameters, matching the capsule and CNN budget.

### 3.5 Residual CNN Baseline

`ResidualCNNBaseline` is a stronger conventional baseline:

- `3x3` convolutional stem,
- six residual blocks,
- channels 64, 128, and 192,
- adaptive average pooling,
- linear classifier.

It has `1,919,178` parameters, fewer than the matched capsule, CNN, and ViT models. This model is not a parameter-matched control. It is included as a stronger conventional architecture check.

## 4. Experimental Setup

### 4.1 Training Protocol

Models are trained for 20 epochs with Adam, learning rate `1e-3`, using best-checkpoint selection by test accuracy. All reported synthetic affine and AffNIST results use best checkpoints.

The main affine-augmented training recipe uses mild random affine augmentation:

| Factor | Training Range |
|---|---:|
| Rotation | `[-30, 30]` degrees |
| Translation | up to `15%` |
| Scale | `[0.85, 1.15]` |
| X shear | `[-15, 15]` degrees |
| Y shear | `0` |

Experiments are run with four seeds: `123`, `321`, `777`, and `2024`.

### 4.2 Synthetic Evaluation

We evaluate fixed affine transformations and random affine transformation distributions. The random in-distribution evaluation includes:

- `random_translate`,
- `random_translate_scale`,
- `random_affine_mild`,
- `random_affine_strong`.

Each random scenario is evaluated with five independently sampled test sets per seed.

We also define two held-out affine scenarios outside the mild training range:

| Scenario | Rotation | Scale | Shear |
|---|---:|---:|---:|
| `heldout_affine_left_zoom` | `[-60, -30]` | `[0.70, 0.85]` | negative x/y shear |
| `heldout_affine_right_zoom` | `[30, 60]` | `[1.15, 1.35]` | positive x/y shear |

These held-out distributions are intended to test extrapolation rather than interpolation within the augmentation distribution.

### 4.3 AffNIST Evaluation

We evaluate on the official AffNIST transformed test set:

`https://www.cs.toronto.edu/~tijmen/affNIST/32x/transformed/test.mat.zip`

Two protocols are reported:

1. Resized `28x28`: AffNIST images are resized down to the MNIST input size.
2. Native `40x40` transfer: AffNIST images are evaluated at original `40x40` size using checkpoint-compatible model adaptations.

The native `40x40` result is a transfer test from `28x28` training, not canonical native AffNIST training. CNNs use adaptive pooling, the ViT interpolates positional embeddings, and capsules adaptively pool primary maps to the trained routing shape.

### 4.4 Representation Probes

To test whether learned phase encodes affine factors, we train Ridge linear probes on hidden representations. For each held-out affine scenario, probes are trained on 2,000 examples and tested on 2,000 examples. Probe targets are:

- rotation angle,
- x translation,
- y translation,
- scale,
- x shear,
- y shear.

Probe performance is reported as mean `R2` and normalized MAE, averaged over affine factors and held-out scenarios.

Feature blocks include:

| Model | Feature Block |
|---|---|
| CNN | penultimate hidden layer |
| Residual CNN | pooled residual features |
| ViT | CLS embedding |
| Real capsule | digit capsules |
| Complex capsule | digit capsules, digit phase, primary phase summaries |

## 5. Results

### 5.1 Training Accuracy

On the affine-augmented MNIST test split, the residual CNN is strongest, followed by the parameter-matched CNN. `ComplexCapsuleB` beats the real capsule control, but not the CNN baselines.

| Model | Mean Best Test Accuracy | Std |
|---|---:|---:|
| ResidualCNNBaseline | 98.905% | 0.068 |
| BaselineCNN | 97.883% | 0.040 |
| ComplexCapsuleB | 97.157% | 0.090 |
| RealCapsuleLarge | 96.280% | 0.135 |
| VisionTransformerBaseline | 94.133% | 0.562 |

This result already narrows the claim. Learned phase improves the capsule model relative to the real capsule, but a simple parameter-matched CNN is stronger, and a smaller residual CNN is stronger still.

### 5.2 Synthetic Random Affine Evaluation

On in-distribution random affine scenarios, `ComplexCapsuleB` again beats `RealCapsuleLarge`, but loses to both CNN baselines.

| Model | Mean Accuracy |
|---|---:|
| ResidualCNNBaseline | 98.124% |
| BaselineCNN | 92.409% |
| ComplexCapsuleB | 91.191% |
| VisionTransformerBaseline | 88.853% |
| RealCapsuleLarge | 88.488% |

The capsule-family margin is real:

`ComplexCapsuleB - RealCapsuleLarge = +2.703` percentage points in the earlier CNN/capsule aggregate, and remains positive in every seed.

But the model-family conclusion is different:

`BaselineCNN - ComplexCapsuleB = +1.218` percentage points on the same sampled random affine setup, and the residual CNN widens the conventional-baseline advantage dramatically.

### 5.3 Held-Out Synthetic Affine Evaluation

Held-out affine shifts are harder. They move outside the mild training ranges, combining stronger rotations, scale changes, translations, and shear.

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
| `heldout_affine_left_zoom` | 47.075% | 79.329% | 41.663% | 37.334% | 42.721% |
| `heldout_affine_right_zoom` | 53.960% | 77.386% | 51.158% | 45.087% | 49.832% |

The held-out result gives the most generous synthetic interpretation for learned phase: `ComplexCapsuleB` improves over `RealCapsuleLarge` by roughly 5 points. But that advantage is still below CNN and far below the residual CNN.

### 5.4 AffNIST Evaluation

AffNIST reverses even the capsule-family advantage. On independent transformed digits, `ComplexCapsuleB` loses to `RealCapsuleLarge`.

Resized `28x28` AffNIST:

| Model | Mean Accuracy | Std |
|---|---:|---:|
| ResidualCNNBaseline | 94.733% | 0.960 |
| BaselineCNN | 71.347% | 1.017 |
| VisionTransformerBaseline | 70.503% | 3.006 |
| RealCapsuleLarge | 62.323% | 0.854 |
| ComplexCapsuleB | 51.635% | 2.308 |

Native `40x40` transfer:

| Model | Mean Accuracy | Std |
|---|---:|---:|
| ResidualCNNBaseline | 97.459% | 1.559 |
| BaselineCNN | 74.695% | 0.570 |
| VisionTransformerBaseline | 74.409% | 3.459 |
| RealCapsuleLarge | 51.956% | 0.744 |
| ComplexCapsuleB | 37.309% | 1.855 |

This is the decisive negative result. If learned phase were producing robust reusable pose representations, AffNIST should be a favorable test. Instead, the complex capsule is worst among the main image models, and the real capsule control is substantially better.

The residual CNN result is especially important. It has fewer parameters than the matched CNN/ViT/capsule models and still reaches `94.733%` on resized AffNIST and `97.459%` on native `40x40` transfer. Architecture and optimization strength dominate the learned-phase effect.

### 5.5 Representation Probes

The probe results show why the learned-phase idea should not be discarded entirely. Complex primary phase summaries do encode affine information better than real capsule digit capsules.

| Feature Block | Mean R2 | Mean Normalized MAE |
|---|---:|---:|
| Residual CNN features | 0.399 | 0.606 |
| ViT CLS | 0.356 | 0.616 |
| ComplexB primary phase stats | 0.277 | 0.688 |
| Real capsule digit caps | 0.049 | 0.838 |

The phase representation is meaningful: it is compact and more predictive of affine factors than the real capsule representation. However, it is not the strongest representation overall. Residual CNN features and ViT CLS embeddings are better affine-factor probe targets.

This matters for interpretation. Learned phase appears to encode some pose-like signal, but the current model does not use that signal well enough to become a competitive robust classifier.

## 6. Discussion

### 6.1 What Worked

The central capsule-family result is reproducible across seeds: learned phase improves the complex capsule relative to an exactly parameter-matched real capsule control on synthetic affine distributions. This is not a trivial parameter-count artifact, because `ComplexCapsuleB` and `RealCapsuleLarge` both have `2,767,568` trainable parameters.

The phase probe also supports a mechanistic interpretation. `primary_phase_stats` produces stronger affine-factor probes than real capsule digit capsules, suggesting that learned phase stores transformation information in a compact form.

### 6.2 What Failed

The stronger robustness claim fails.

First, the parameter-matched CNN is stronger than `ComplexCapsuleB` on sampled synthetic affine tests. Second, the ViT CLS representation is stronger than learned phase under linear probing. Third, the residual CNN dominates every synthetic and AffNIST evaluation while using fewer parameters. Fourth, AffNIST reverses the capsule-family result: the complex capsule loses to the real capsule.

These failures rule out the simple claim that learned complex phase is a broadly superior route to affine robustness.

### 6.3 Why AffNIST Matters

Synthetic affine MNIST is useful because it gives controlled factors and enables probes. But it is still generated by the same transform machinery used during evaluation. AffNIST is an independent transformed digit benchmark. It is therefore an important check against tuning to our synthetic generator.

The AffNIST result is not a small degradation. On resized AffNIST, `ComplexCapsuleB` is roughly 10.7 points below `RealCapsuleLarge`, 19.7 points below `BaselineCNN`, and 43.1 points below `ResidualCNNBaseline`. On native `40x40` transfer, the complex capsule falls further to `37.309%`.

### 6.4 Negative Results as Useful Results

The experiment still contributes useful evidence. It identifies a real learned-phase effect, then shows that the effect is insufficient under stronger tests. This is valuable because capsule-style models are often motivated by qualitative intuitions about pose. Those intuitions need hard controls:

- exact real-valued capsule controls,
- parameter-matched CNN and transformer baselines,
- stronger conventional architecture baselines,
- independent datasets,
- representation probes.

Without these controls, the synthetic capsule-family improvement could easily be mistaken for a broader robustness result.

## 7. Limitations

The study has several limitations.

First, the main training data is MNIST-derived. MNIST is useful for controlled pose and affine tests, but it is saturated and not representative of natural images.

Second, the native `40x40` AffNIST result is a transfer protocol from `28x28` training. It is not canonical native AffNIST training. A stronger AffNIST claim requires training all image models directly on native `40x40` data.

Third, the residual CNN is not parameter-matched. This is intentional: it is included as a stronger conventional baseline, not as a fair capacity control. Its fewer-parameter dominance makes the result more damaging to the capsule robustness claim, but it does not isolate architecture from optimization and inductive-bias differences.

Fourth, linear probes are not proof that a representation is causally used for classification. They show that affine information is recoverable, not that the model exploits it robustly.

Fifth, the complex capsule implementation is only one design point. Other complex routing objectives, explicit equivariance constraints, different phase regularizers, or native spatial capsule architectures may behave differently.

## 8. Future Work

The most immediate follow-up is canonical native AffNIST training:

- train all image models directly on `40x40` inputs,
- include `ResidualCNNBaseline` as the main conventional baseline,
- keep `BaselineCNN`, `VisionTransformerBaseline`, `RealCapsuleLarge`, and `ComplexCapsuleB`,
- report both accuracy and representation probes where transform metadata is available.

The next controlled representation benchmarks should include datasets with explicit latent factors:

- dSprites for rotation, position, and scale,
- SmallNORB for viewpoint and lighting,
- possibly CIFAR-10-C or natural-image corruption benchmarks only after the controlled pose results are better understood.

Architecturally, future complex capsule work should test whether phase can be made more usable by the classifier. Possible directions include phase regularization, explicit equivariance losses, routing objectives that preserve phase information, or hybrid CNN/capsule models that use strong convolutional stems before capsule routing.

## 9. Conclusion

Learned complex phase is not a magic source of transformation invariance. It improves a capsule model relative to an exactly parameter-matched real capsule control on synthetic affine distributions, and its phase features carry recoverable affine information. But the broader robustness claim fails under stronger baselines and independent evaluation. A parameter-matched CNN is stronger on sampled affine tests, ViT and residual CNN representations probe better for affine factors, and AffNIST reverses the capsule-family advantage.

The correct conclusion is therefore narrow and diagnostic: learned phase is a useful capsule-family mechanism worth studying, but the present architecture does not compete with strong conventional image models for affine robustness.

## Reproducibility Notes

Latest verification:

```text
55 passed, 1 skipped
```

Primary source files:

- `ai_unity/complex_capsules.py`
- `ai_unity/training.py`
- `ai_unity/evaluate_complex_affines.py`
- `ai_unity/evaluate_affine_probes.py`
- `ai_unity/evaluate_affnist.py`
- `ai_unity/data.py`

Primary result logs:

- `UNITY_NOTEBOOK.md`, section 28
- `FINDINGS_SUMMARY.md`, 2026-06-28 validation update

## Draft Figure and Table Plan

Table 1: Model parameter counts and architecture summary.

Table 2: Four-seed training accuracy on affine-augmented MNIST.

Table 3: Synthetic random and held-out affine accuracy.

Table 4: AffNIST resized `28x28` and native `40x40` transfer accuracy.

Table 5: Linear probe performance by feature block.

Figure 1: Schematic of real capsule versus learned-phase complex capsule.

Figure 2: Accuracy drop from synthetic in-distribution affine to held-out affine.

Figure 3: AffNIST ranking by model family.

Figure 4: Probe R2 comparison for residual CNN, ViT, complex phase, and real capsules.

## References To Fill

TODO: Add full BibTeX entries and dataset citations for the final submission.
