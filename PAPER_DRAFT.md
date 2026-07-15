# When Learned Phase Is Not Enough: A Diagnostic Evaluation of Typed Neural Signals for Affine Robustness

Second draft. Working paper, 2026-07-14. (First draft 2026-06-28; this revision replaces the confounded capsule control, adds a confound ablation, and corrects the capsule-family margins.)

## Abstract

Capsule networks are motivated by the idea that vector-valued or structured activations can represent object pose more explicitly than scalar features. This paper tests a narrow version of that idea: whether a learned complex phase inside a capsule network improves robustness to rotations, translations, scale changes, and shear. We compare a learned-phase complex capsule model, `ComplexCapsuleB`, against a parameter-matched real capsule control, a parameter-matched CNN, a parameter-matched Vision Transformer baseline, and a stronger residual CNN baseline. An earlier version of this study used a control, `RealCapsuleLarge`, that matched the complex model's parameter count but harbored two confounds: its extra parameters were functionally degenerate (two summed linear banks collapse to one linear map), and its class readout had a smaller dynamic range than the complex model's under a shared cross-entropy loss. We replace it with `RealCapsuleControlV2`, a control matched in parameter count, representational width, and readout form, and we decompose the two confounds with readout-only and capacity-only ablations. The confounds account for roughly half of the originally reported held-out capsule-family gap (+5.07 to +2.56 points), but a genuine learned-phase improvement survives: across four seeds on synthetic affine MNIST, `ComplexCapsuleB` outperforms the corrected control on random affine distributions (+2.22 points) and held-out affine shifts (+2.56 points), positive in every seed. However, this improvement is not sufficient to beat conventional baselines. A parameter-matched CNN exceeds the complex capsule on sampled synthetic affine tests, and a fewer-parameter residual CNN dominates all models on synthetic affine tests and AffNIST. On independent AffNIST evaluation, `ComplexCapsuleB` loses not only to CNN and ViT baselines, but also to the corrected real capsule control — by roughly 11 points resized and 17 points at native size — so the decisive negative result is strengthened, not weakened, by removing the confounds. Linear probes show that learned phase contains compact pose-related information, but residual CNN features and ViT CLS embeddings encode affine factors more strongly. The result is therefore negative but useful: learned complex phase can improve a capsule model relative to a properly matched real-valued capsule control, yet current learned-phase capsules do not provide competitive transformation robustness against strong conventional architectures.

## 1. Introduction

Deep image classifiers often succeed by learning features that are tolerant to nuisance transformations. Convolutional networks build in translation bias through weight sharing and local receptive fields, while modern residual architectures and transformer-style models can learn broad invariances from data and augmentation. Capsule networks offer a different promise: instead of hiding pose variation inside high-dimensional scalar feature maps, capsules attempt to represent object identity and pose in structured activations.

This project began with a stronger hypothesis: if capsule activations are represented as complex numbers, then phase might naturally carry pose-like information. Rotations and other geometric changes often have an angular structure, so a learned phase channel could, in principle, give capsule routing a better coordinate system for transformation robustness.

The experiments below test that hypothesis directly. The answer is mixed, and the most important result is negative. Learned phase does help within the capsule family: after mild affine augmentation, `ComplexCapsuleB` consistently beats a parameter-matched real capsule control on synthetic affine distributions. Getting that comparison right turned out to be the hardest part of the study. Our first control matched the complex model's parameter count exactly but was confounded in two ways we did not initially recognize — its added parameters were provably redundant, and its class readout was handicapped under the shared loss. Fixing both confounds cut the capsule-family margin roughly in half; it did not eliminate it. But the result does not survive stronger baselines. A parameter-matched CNN is better on sampled affine tests, a parameter-matched ViT gives stronger representation probes, and a smaller residual CNN dominates both synthetic affine tests and AffNIST. On AffNIST, the complex capsule also loses to the corrected real capsule control.

The main contribution is therefore not a new state-of-the-art robust classifier. It is a controlled diagnostic study showing where a plausible learned-phase capsule idea works, where it fails, which control-design mistakes can silently inflate a capsule-family result, and which evidence should prevent overclaiming.

We make five contributions:

1. We introduce and evaluate a learned-phase complex capsule model against a real capsule control matched in parameter count, representational width, and readout form.
2. We identify two confounds — degenerate control capacity and asymmetric readout dynamic range — that inflated the initially reported capsule-family gap, and we quantify each with single-fix ablation controls.
3. We evaluate robustness across fixed rotations, synthetic affine transformations, held-out affine distributions, and the AffNIST transformed test set.
4. We compare against parameter-matched CNN and ViT baselines, plus a stronger residual CNN baseline with fewer parameters.
5. We use linear probes to ask whether learned phase carries pose-related information even when it fails to win classification accuracy.

## 2. Background and Hypothesis

Capsule networks represent entities with vectors rather than scalar activations, and use routing procedures to decide how lower-level capsules contribute to higher-level capsules. The motivating claim is that capsule dimensions can encode pose-like degrees of freedom while capsule length or norm encodes class evidence.

Complex-valued representations add another possible structure: magnitude can represent evidence strength, while phase can represent a periodic or angular variable. This suggests a natural hypothesis for geometric robustness:

> A capsule model with learned complex phase should better represent affine nuisance factors, improving robustness to transformations relative to a real-valued capsule with the same parameter budget.

Testing that hypothesis requires care about what "the same parameter budget" means. As Section 3 details, an equal parameter count can conceal an unequal capacity or an unequal readout, and either can masquerade as a phase effect.

This paper separates three claims that are easy to conflate:

1. Capsule-family claim: learned phase improves over a properly matched real capsule control.
2. Baseline claim: learned phase beats conventional CNN or transformer baselines.
3. Representation claim: learned phase encodes pose or affine factors in a probe-detectable way.

The experiments support the first claim on synthetic affine data (at roughly half the margin originally reported, after two control confounds are removed), partially support the third claim, and reject the second claim.

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

### 3.1 Learned-Phase Complex Capsule

`ComplexCapsuleB` is the learned-phase model under study. It uses:

- a `9x9` convolutional stem with 256 channels,
- one primary magnitude convolution bank,
- one primary phase convolution bank,
- complex transformation matrices built from real and imaginary parameter banks,
- complex squash and complex agreement during routing,
- 3 routing iterations,
- 10 class capsules of dimension 8 (complex),
- class evidence read out as the sum of per-dimension magnitudes of the squashed class capsule, bounded by `sqrt(8)`.

Magnitude is passed through `softplus`; phase is learned directly. The model returns `digit_caps`, `primary_phase`, and `digit_phase` for diagnostics. It has `2,767,568` trainable parameters by tensor-element count. (One footnote for exactness: the routing bias is stored as a complex tensor, so in raw real scalars the model has `2,767,648` — 80 more than the real controls, a `0.003%` difference we treat as immaterial.)

### 3.2 Real Capsule Control (Corrected): `RealCapsuleControlV2`

Designing the real-valued control is the crux of the capsule-family comparison, and our first attempt got it wrong (Section 3.3). The corrected control, `RealCapsuleControlV2`, is matched to the complex model on three axes rather than parameter count alone:

- **Parameter count.** Exactly `2,767,568` trainable parameters, equal to `ComplexCapsuleB`.
- **Representational width.** A complex 8-dimensional primary capsule is 16 real dimensions. The control therefore uses one genuinely wider primary bank producing 16-dimensional real primary capsules — spending its budget the same way the complex model does, on a single wide primary representation and a correspondingly larger transformation tensor, rather than on redundant duplicate banks.
- **Readout form.** The class score is the sum of per-dimension absolute values of the squashed class capsule — the identical functional form and identical `sqrt(8)` bound as the complex model's magnitude-sum readout.

Concretely: the same `9x9` stem, one primary convolution bank producing `8` capsule types of dimension 16, a single transformation tensor of shape `[10, 72, 8, 16]`, 3 routing iterations, and 10 class capsules of dimension 8.

The remaining differences between `ComplexCapsuleB` and `RealCapsuleControlV2` are intrinsic to the hypothesis being tested: the complex model constrains its 16 real primary dimensions into 8 magnitude-phase pairs (with `softplus` magnitudes) and its transformation obeys complex-multiplication weight tying, while the control's 16 dimensions and transformation are unconstrained. That is the comparison the learned-phase hypothesis calls for.

### 3.3 The Original Control and Its Two Confounds

The first draft of this study used `RealCapsuleLarge` as the control. It reached the same `2,767,568` parameters by adding a second primary convolution bank and a second transformation bank whose outputs were simply summed with the first. We retain it in the results as a cautionary baseline, because it carries two confounds that inflated the originally reported capsule-family gap:

1. **Degenerate capacity.** The sum of two linear maps is a single linear map. We verified numerically that merging the two banks reproduces `RealCapsuleLarge`'s outputs to `7e-10`: the control is functionally identical to the `1,394,320`-parameter small real capsule. Its parameter count matched the complex model; its expressive capacity did not.
2. **Readout dynamic range.** `RealCapsuleLarge` read out class evidence as the L2 norm of the squashed class capsule, which squash bounds below `1.0`, while the complex model's magnitude-sum readout is bounded by `sqrt(8) ~ 2.83`. Both scores feed a shared softmax cross-entropy, so the real control's maximum achievable logit separation — and with it its confidence ceiling and gradient signal — was structurally capped at roughly one third of the complex model's, purely by readout choice. (Sabour et al.'s margin loss avoids exactly this trap; under cross-entropy the readout forms must be matched instead.)

To attribute the two confounds separately we also train two single-fix ablations:

| Model | Architecture | Readout | Isolates |
|---|---|---|---|
| `RealCapsuleLargeL1` | original degenerate two-bank | magnitude-sum (`sqrt(8)` bound) | readout fix only |
| `RealCapsuleControlV2Norm` | corrected wide single-bank | L2 norm (bound `1.0`) | capacity fix only |
| `RealCapsuleControlV2` | corrected wide single-bank | magnitude-sum (`sqrt(8)` bound) | both fixes |

All three ablation controls also have exactly `2,767,568` trainable parameters.

### 3.4 Parameter-Matched CNN

`BaselineCNN` is a conventional convolutional baseline with:

- convolution layers with 64 and 128 channels,
- max pooling,
- adaptive average pooling to `7x7`,
- a three-layer MLP classifier head.

It has `2,768,502` parameters, within `0.034%` of the large capsule parameter budget.

### 3.5 Parameter-Matched Vision Transformer

`VisionTransformerBaseline` is a small ViT-style model:

- `4x4` patch embedding,
- learned CLS token,
- learned positional embedding,
- 9 transformer encoder layers,
- width 192,
- 4 attention heads,
- feed-forward width 384.

It has `2,767,663` parameters, matching the capsule and CNN budget.

### 3.6 Residual CNN Baseline

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

Two protocol details deserve explicit disclosure. First, best-checkpoint selection uses the MNIST test split — the same 10,000 images whose transformed versions the synthetic affine evaluations report on. Selection is over only 20 epochs and is applied identically to every model, and the AffNIST evaluation uses an entirely independent image set, so the model orderings are unaffected; but the Section 5.1 accuracies are max-over-epochs statistics and are optimistically biased in absolute terms. Second, the "mild random affine augmentation" below draws one transform per training image at dataset construction and replays it every epoch — a fixed re-rendered training set, not fresh per-epoch augmentation. This too is identical across models, so relative comparisons stand, but absolute robustness numbers should not be compared against standard per-epoch-augmentation results in the literature.

The mild affine training recipe:

| Factor | Training Range |
|---|---:|
| Rotation | `[-30, 30]` degrees |
| Translation | up to `15%` |
| Scale | `[0.85, 1.15]` |
| X shear | `[-15, 15]` degrees |
| Y shear | `0` |

Experiments are run with four seeds: `123`, `321`, `777`, and `2024`. In the first-draft runs the seed controlled data order and augmentation draws but not weight initialization (models were constructed before the seed was applied), so those runs were independent but not exactly reproducible from the stated seed. This is fixed in the current code: seeding now precedes model construction, and the confound-ablation runs of Section 5.6 are reproducible from their seeds. Because the fix changes only reproducibility, not the statistical independence of the four runs, we retain the first-draft results for the unchanged models rather than retraining them.

All reported uncertainty is the population standard deviation over the four seeds (divisor `n`, not `n-1`). With `n = 4` we treat means, per-seed sign consistency, and margins well outside seed spread as the meaningful evidence, and we flag comparisons that fall inside seed noise.

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

On the affine-augmented MNIST test split, the residual CNN is strongest, followed by the parameter-matched CNN. `ComplexCapsuleB` beats both the original and the corrected real capsule controls, but not the CNN baselines.

| Model | Mean Best Test Accuracy | Std |
|---|---:|---:|
| ResidualCNNBaseline | 98.905% | 0.068 |
| BaselineCNN | 97.883% | 0.040 |
| ComplexCapsuleB | 97.157% | 0.090 |
| RealCapsuleControlV2 (corrected control) | 96.668% | 0.105 |
| RealCapsuleLarge (original control) | 96.280% | 0.135 |
| VisionTransformerBaseline | 94.133% | 0.562 |

This result already narrows the claim. Learned phase improves the capsule model relative to the real capsule controls, but a simple parameter-matched CNN is stronger, and a smaller residual CNN is stronger still. Note also that the corrected control recovers about 0.4 points of the original control's deficit, a first sign that the original comparison overstated the phase effect.

### 5.2 Synthetic Random Affine Evaluation

On in-distribution random affine scenarios, `ComplexCapsuleB` again beats both capsule controls, but loses to both CNN baselines.

| Model | Mean Accuracy |
|---|---:|
| ResidualCNNBaseline | 98.124% |
| BaselineCNN | 92.409% |
| ComplexCapsuleB | 91.191% |
| RealCapsuleControlV2 (corrected control) | 88.968% |
| VisionTransformerBaseline | 88.853% |
| RealCapsuleLarge (original control) | 88.488% |

The capsule-family margin is real but smaller than first reported:

`ComplexCapsuleB - RealCapsuleControlV2 = +2.223` percentage points against the corrected control (versus `+2.703` against the original confounded control), and remains positive in every seed (per-seed margins `+2.49`, `+2.77`, `+1.72`, `+1.90`).

But the model-family conclusion is different:

`BaselineCNN - ComplexCapsuleB = +1.218` percentage points on the same sampled random affine setup, and the residual CNN widens the conventional-baseline advantage dramatically.

### 5.3 Held-Out Synthetic Affine Evaluation

Held-out affine shifts are harder. They move outside the mild training ranges, combining stronger rotations, scale changes, translations, and shear. (The held-out translation range, up to `25%`, also exceeds the `15%` training range — the shift is stronger than rotation/scale/shear alone would suggest.)

| Model | Mean Accuracy |
|---|---:|
| ResidualCNNBaseline | 78.358% |
| BaselineCNN | 50.518% |
| VisionTransformerBaseline | 46.411% |
| ComplexCapsuleB | 46.276% |
| RealCapsuleControlV2 (corrected control) | 43.715% |
| RealCapsuleLarge (original control) | 41.210% |

The ViT and `ComplexCapsuleB` means differ by `0.135` points with seed standard deviations several times larger; we treat those two models as tied on this evaluation rather than ordered.

Held-out breakdown:

| Scenario | CNN | ResNet | ViT | RealLarge | ComplexB |
|---|---:|---:|---:|---:|---:|
| `heldout_affine_left_zoom` | 47.075% | 79.329% | 41.663% | 37.334% | 42.721% |
| `heldout_affine_right_zoom` | 53.960% | 77.386% | 51.158% | 45.087% | 49.832% |

The held-out result gave the most generous synthetic interpretation for learned phase in the first draft: `ComplexCapsuleB` improved over `RealCapsuleLarge` by roughly 5 points. Against the corrected control the margin is `+2.561` points — about half the original — though still positive in every seed (Section 5.6). Either way, that advantage is below CNN and far below the residual CNN.

### 5.4 AffNIST Evaluation

AffNIST reverses even the capsule-family advantage. On independent transformed digits, `ComplexCapsuleB` loses to every real capsule control, original and corrected alike.

Resized `28x28` AffNIST:

| Model | Mean Accuracy | Std |
|---|---:|---:|
| ResidualCNNBaseline | 94.733% | 0.960 |
| BaselineCNN | 71.347% | 1.017 |
| VisionTransformerBaseline | 70.503% | 3.006 |
| RealCapsuleControlV2 (corrected control) | 62.497% | 1.141 |
| RealCapsuleLarge (original control) | 62.323% | 0.854 |
| ComplexCapsuleB | 51.635% | 2.308 |

Native `40x40` transfer:

| Model | Mean Accuracy | Std |
|---|---:|---:|
| ResidualCNNBaseline | 97.459% | 1.559 |
| BaselineCNN | 74.695% | 0.570 |
| VisionTransformerBaseline | 74.409% | 3.459 |
| RealCapsuleControlV2 (corrected control) | 54.370% | 1.528 |
| RealCapsuleLarge (original control) | 51.956% | 0.744 |
| ComplexCapsuleB | 37.309% | 1.855 |

This is the decisive negative result, and correcting the control strengthens it. If learned phase were producing robust reusable pose representations, AffNIST should be a favorable test. Instead, the complex capsule is worst among the main image models; the corrected control matches the original at `28x28` and beats it by `2.4` points at native `40x40`, widening the complex model's deficit to roughly `10.9` points resized and `17.1` points native. Whatever the confounds gave the complex model on synthetic data, they were not what made it lose on AffNIST.

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

This matters for interpretation. Learned phase appears to encode some pose-like signal, but the current model does not use that signal well enough to become a competitive robust classifier. Two caveats on the probe ranking itself: the Ridge regularization strength is fixed (`alpha = 10`) across feature blocks of different dimensionality rather than tuned per block, so the quantitative ordering should be read as approximate; and the headline mean R2 is dominated by translation recovery — per-factor results show `primary_phase_stats` recovering translation well (R2 up to `0.86`) but rotation only weakly (R2 about `0.16`), so "phase carries pose information" is, in these numbers, mostly "phase carries translation information."

### 5.6 Confound Ablation: How Much of the Capsule-Family Gap Was the Control's Fault?

The two confounds of Section 3.3 were fixed and re-run under the identical training protocol (four seeds, 20 epochs, mild affine augmentation, best checkpoints, five random samples per scenario). `ComplexCapsuleB` numbers are the original run; the three controls are freshly trained.

Four-seed means with population standard deviation:

| Model | Train Best | Random Affine | Held-Out Affine |
|---|---:|---:|---:|
| ComplexCapsuleB | 97.157±0.090 | 91.191±0.404 | 46.277±0.844 |
| RealCapsuleLarge (confounded) | 96.280±0.135 | 88.488±0.403 | 41.210±0.677 |
| RealCapsuleLargeL1 (readout fix only) | 96.370±0.187 | 88.708±0.329 | 42.566±0.555 |
| RealCapsuleControlV2Norm (capacity fix only) | 96.330±0.350 | 88.498±1.020 | 42.956±1.846 |
| RealCapsuleControlV2 (both fixes) | 96.668±0.105 | 88.968±0.142 | 43.715±0.316 |

Complex-minus-control margins (all sign-consistent in 4 of 4 seeds):

| Control | Random Affine | Held-Out Affine |
|---|---:|---:|
| RealCapsuleLarge (confounded) | +2.703 | +5.066 |
| RealCapsuleLargeL1 (readout fix) | +2.483 | +3.711 |
| RealCapsuleControlV2Norm (capacity fix) | +2.693 | +3.321 |
| RealCapsuleControlV2 (both fixes) | +2.223 | +2.561 |

AffNIST for the ablation controls:

| Control | Resized 28x28 | Native 40x40 |
|---|---:|---:|
| RealCapsuleLarge (confounded) | 62.323±0.854 | 51.956±0.744 |
| RealCapsuleLargeL1 (readout fix) | 62.375±0.528 | 50.364±2.614 |
| RealCapsuleControlV2Norm (capacity fix) | 61.359±1.053 | 49.904±1.077 |
| RealCapsuleControlV2 (both fixes) | 62.497±1.141 | 54.370±1.528 |

Three observations:

1. **The confounds were real and material.** On held-out affine data — the first draft's most favorable synthetic result — fixing both confounds cuts the capsule-family margin roughly in half, from `+5.07` to `+2.56` points. On random affine distributions the reduction is smaller (`+2.70` to `+2.22`).
2. **The two fixes contribute roughly additively on held-out data.** The readout fix alone improves the control by about `1.4` points and the capacity fix alone by about `1.7`; together they yield `2.5` of control improvement.
3. **A genuine learned-phase effect survives the corrected control.** The remaining margins are positive in every seed on both synthetic evaluations. The effect is smaller than first reported, but it is not an artifact of the control design. On AffNIST, however, the corrected control only extends the complex model's deficit.

### 6.1 What Worked

The central capsule-family result survives its strongest test: learned phase improves the complex capsule relative to a real capsule control matched in parameter count, representational width, and readout form, on synthetic affine distributions, in every seed. The first draft claimed this was "not a trivial parameter-count artifact" because both models had `2,767,568` parameters — an argument the original control could not actually support, since its parameter match concealed a capacity mismatch and a readout handicap. The corrected claim is narrower and stronger: after removing both confounds (Section 5.6), a `+2.2` to `+2.6` point capsule-family margin remains, sign-consistent across seeds. Parameter count alone is not a sufficient matching criterion; the surviving margin rests on the corrected control, not the count.

The phase probe also supports a mechanistic interpretation. `primary_phase_stats` produces stronger affine-factor probes than real capsule digit capsules, suggesting that learned phase stores transformation information — predominantly translation — in a compact form.

### 6.2 What Failed

The stronger robustness claim fails.

First, the parameter-matched CNN is stronger than `ComplexCapsuleB` on sampled synthetic affine tests. Second, the ViT CLS representation is stronger than learned phase under linear probing. Third, the residual CNN dominates every synthetic and AffNIST evaluation while using fewer parameters. Fourth, AffNIST reverses the capsule-family result: the complex capsule loses to every real capsule control, including the corrected one, which beats it by an even wider margin at native size.

These failures rule out the simple claim that learned complex phase is a broadly superior route to affine robustness.

### 6.3 Why AffNIST Matters

Synthetic affine MNIST is useful because it gives controlled factors and enables probes. But it is still generated by the same transform machinery used during evaluation. AffNIST is an independent transformed digit benchmark. It is therefore an important check against tuning to our synthetic generator.

The AffNIST result is not a small degradation. On resized AffNIST, `ComplexCapsuleB` is roughly 10.9 points below the corrected `RealCapsuleControlV2`, 19.7 points below `BaselineCNN`, and 43.1 points below `ResidualCNNBaseline`. On native `40x40` transfer, the complex capsule falls further to `37.309%`, some 17.1 points below the corrected control. The synthetic-versus-AffNIST contrast is also consistent with a selection effect the protocol cannot fully exclude: best checkpoints are chosen on the MNIST test distribution that the synthetic evaluations re-transform, while AffNIST is untouched by selection — and it is precisely on AffNIST that the capsule-family advantage disappears.

### 6.4 Negative Results as Useful Results

The experiment still contributes useful evidence. It identifies a real learned-phase effect, then shows that the effect is insufficient under stronger tests. This is valuable because capsule-style models are often motivated by qualitative intuitions about pose. Those intuitions need hard controls:

- real-valued capsule controls matched in capacity and readout, not merely parameter count,
- single-fix ablations that attribute any gap to specific control defects,
- parameter-matched CNN and transformer baselines,
- stronger conventional architecture baselines,
- independent datasets,
- representation probes.

Without these controls, the synthetic capsule-family improvement could easily be mistaken for a broader robustness result — and, as Section 5.6 shows, even a well-intentioned "exactly parameter-matched" control can silently inflate the effect being measured. Matching parameter counts is easy to verify and easy to get wrong: two summed linear banks pass the count check while adding no capacity, and a readout mismatch is invisible in any parameter table. We recommend that capsule-family comparisons verify controls functionally — for example, by checking that the control cannot be losslessly compressed into a smaller model — rather than arithmetically.

## 7. Limitations

The study has several limitations.

First, the main training data is MNIST-derived. MNIST is useful for controlled pose and affine tests, but it is saturated and not representative of natural images.

Second, the native `40x40` AffNIST result is a transfer protocol from `28x28` training. It is not canonical native AffNIST training. A stronger AffNIST claim requires training all image models directly on native `40x40` data. The native protocol's adaptive pooling of primary capsule maps plausibly hurts capsule routing more than pooling hurts a CNN head; it applies identically to real and complex capsules, so the capsule-family comparison is fair, but the native numbers should be read as capsule-unfavorable in absolute terms.

Third, the residual CNN is not parameter-matched. This is intentional: it is included as a stronger conventional baseline, not as a fair capacity control. Its fewer-parameter dominance makes the result more damaging to the capsule robustness claim, but it does not isolate architecture from optimization and inductive-bias differences.

Fourth, linear probes are not proof that a representation is causally used for classification. They show that affine information is recoverable, not that the model exploits it robustly. The probe ranking additionally uses a fixed Ridge alpha across feature blocks and is dominated by translation factors (Section 5.5).

Fifth, the complex capsule implementation is only one design point. Other complex routing objectives, explicit equivariance constraints, different phase regularizers, or native spatial capsule architectures may behave differently.

Sixth, checkpoint selection uses the MNIST test split rather than a held-out validation split (Section 4.1). Orderings are unaffected — selection applies identically to all models and AffNIST is independent — but Section 5.1's absolute accuracies are optimistically biased, and a validation-split protocol would be cleaner.

Seventh, the first-draft training runs (all models except the three ablation controls) were not exactly reproducible from their stated seeds because weight initialization preceded seeding; the code is fixed, but a full re-run of the unchanged models under seeded initialization has not been performed. The four runs per model remain statistically independent, which is what the seed-consistency evidence relies on.

Eighth, with four seeds and no formal significance testing, we rely on per-seed sign consistency and margins large relative to seed spread. The load-bearing capsule-family margins pass that bar (4/4 seeds, means 3 to 8 times the seed standard deviation); comparisons flagged as within noise (ViT versus `ComplexCapsuleB` held-out) should not be cited as orderings.

## 8. Future Work

The most immediate follow-up is canonical native AffNIST training:

- train all image models directly on `40x40` inputs,
- include `ResidualCNNBaseline` as the main conventional baseline,
- keep `BaselineCNN`, `VisionTransformerBaseline`, `RealCapsuleControlV2`, and `ComplexCapsuleB` (the confounded `RealCapsuleLarge` can be dropped),
- report both accuracy and representation probes where transform metadata is available.

The next controlled representation benchmarks should include datasets with explicit latent factors:

- dSprites for rotation, position, and scale,
- SmallNORB for viewpoint and lighting,
- possibly CIFAR-10-C or natural-image corruption benchmarks only after the controlled pose results are better understood.

Architecturally, future complex capsule work should test whether phase can be made more usable by the classifier. Possible directions include phase regularization, explicit equivariance losses, routing objectives that preserve phase information, or hybrid CNN/capsule models that use strong convolutional stems before capsule routing.

## 9. Conclusion

Learned complex phase is not a magic source of transformation invariance. It improves a capsule model relative to a real capsule control matched in parameter count, capacity, and readout form on synthetic affine distributions — by `+2.2` to `+2.6` points after two control confounds that had inflated the margin to `+2.7`/`+5.1` were removed — and its phase features carry recoverable affine information, mostly about translation. But the broader robustness claim fails under stronger baselines and independent evaluation. A parameter-matched CNN is stronger on sampled affine tests, ViT and residual CNN representations probe better for affine factors, and AffNIST reverses the capsule-family advantage against the corrected control by an even wider margin than against the original.

The correct conclusion is therefore narrow and diagnostic, twice over. About the model: learned phase is a real but modest capsule-family mechanism worth studying, and the present architecture does not compete with strong conventional image models for affine robustness. About the method: a parameter-count match is not a control; the count concealed a degenerate architecture and a handicapped readout, and only single-fix ablations made the true size of the phase effect visible.

## Reproducibility Notes

Latest verification:

```text
55 passed, 1 skipped
```

Primary source files:

- `ai_unity/complex_capsules.py` (includes `RealCapsuleNetControlV2`, `RealCapsuleNetControlV2Norm`, `RealCapsuleNetLargeL1`)
- `ai_unity/training.py`
- `ai_unity/evaluate_complex_affines.py`
- `ai_unity/evaluate_affine_probes.py`
- `ai_unity/evaluate_affnist.py`
- `ai_unity/data.py`

Primary result logs:

- `UNITY_NOTEBOOK.md`, section 28
- `FINDINGS_SUMMARY.md`, 2026-06-28 validation update
- `CONTROL_RERUN_FINDINGS.md`, 2026-07-14 confound ablation
- `results/complex_mnist_affine_aug_mild_controls_20ep_seed{123,321,777,2024}/` and matching `_best_affine_eval_5samples/` and `affnist_{resized28,native40}_controls_seed*_full_eval/` directories

Seeding: runs from 2026-07-14 onward (the three ablation controls) seed before model construction and are exactly reproducible from `--seed`; earlier runs seeded data order and augmentation draws but not weight initialization. Reported standard deviations are population standard deviations over four seeds.

## Draft Figure and Table Plan

Table 1: Model parameter counts and architecture summary (including the three ablation controls).

Table 2: Four-seed training accuracy on affine-augmented MNIST.

Table 3: Synthetic random and held-out affine accuracy.

Table 4: AffNIST resized `28x28` and native `40x40` transfer accuracy.

Table 5: Linear probe performance by feature block.

Table 6: Confound ablation — capsule-family margins against the original control, each single-fix control, and the corrected control (Section 5.6).

Figure 1: Schematic of real capsule versus learned-phase complex capsule, annotated with the two control confounds (degenerate summed banks; readout bound mismatch).

Figure 2: Accuracy drop from synthetic in-distribution affine to held-out affine.

Figure 3: AffNIST ranking by model family.

Figure 4: Probe R2 comparison for residual CNN, ViT, complex phase, and real capsules.

Figure 5: Waterfall of the held-out capsule-family margin: `+5.07` (confounded control) minus readout fix, minus capacity fix, to `+2.56` (corrected control).

## References To Fill

TODO: Add full BibTeX entries and dataset citations for the final submission.
