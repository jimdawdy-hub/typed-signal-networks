# Confound-Fixed Control Rerun

**Date:** 2026-07-14
**Branch:** `fix/capsule-control-confounds`

## Why

An external review of the paper draft identified two confounds in the headline
"exactly parameter-matched" capsule-family comparison:

1. **Capacity degeneracy.** `RealCapsuleNetLarge` reaches 2,767,568 parameters
   with two summed duplicate conv/W banks. The sum of two linear maps is one
   linear map: the control was verified (to 7e-10) to be functionally identical
   to the 1,394,320-parameter `RealCapsuleNet`. Parameter counts matched;
   capacity did not.
2. **Readout asymmetry.** The real control's class score was the L2 norm of the
   squashed class capsule (bounded by 1), while `ComplexCapsuleNetB` uses the
   sum of per-dimension magnitudes (bounded by sqrt(8) ≈ 2.83). Under shared
   softmax cross-entropy this caps the real control's confidence and gradient
   signal purely by readout choice.

A third defect fixed in passing: model weights were initialized *before*
`seed_everything`, so runs were not reproducible from `--seed`. `run_complex`
now seeds before model construction.

## New models

| Key | Class | What it isolates |
|---|---|---|
| `real-large-l1` | `RealCapsuleNetLargeL1` | Readout fix only (degenerate arch, L1 readout) |
| `control-v2-norm` | `RealCapsuleNetControlV2Norm` | Capacity fix only (16-dim primary caps, L2 readout) |
| `control-v2` | `RealCapsuleNetControlV2` | **Both fixes** — the corrected control |

`RealCapsuleNetControlV2` spends the budget the same way the complex model
does: one genuinely wider primary bank (16 real dims, mirroring the complex
model's 8 complex = 16 real dims) and the identical L1-of-magnitudes readout
with the same sqrt(8) bound. All three new models have exactly **2,767,568**
trainable parameters, equal to `ComplexCapsuleNetB` and `RealCapsuleNetLarge`.

## Protocol

Identical to the paper's cnn-run: 20 epochs, Adam 1e-3, mild affine
augmentation, AMP, seeds 123/321/777/2024, best-checkpoint selection, then
`evaluate_complex_affines` with 5 random samples per scenario and
`evaluate_affnist` at resized 28x28 and native 40x40.
`ComplexCapsuleB` numbers below are the paper's published run (checkpoints
reused, not retrained).

## Results (4-seed mean ± population std)

### Synthetic affine

| Model | Train best | Random affine | Held-out affine |
|---|---:|---:|---:|
| ComplexCapsuleB (published) | 97.157±0.090 | 91.191±0.404 | 46.277±0.844 |
| RealCapsuleLarge (published, confounded) | 96.280±0.135 | 88.488±0.403 | 41.210±0.677 |
| RealCapsuleLargeL1 (readout fix) | 96.370±0.187 | 88.708±0.329 | 42.566±0.555 |
| RealCapsuleControlV2Norm (capacity fix) | 96.330±0.350 | 88.498±1.020 | 42.956±1.846 |
| **RealCapsuleControlV2 (both fixes)** | **96.668±0.105** | **88.968±0.142** | **43.715±0.316** |

Margins, ComplexCapsuleB minus control (all sign-consistent 4/4 seeds):

| Control | Random affine | Held-out |
|---|---:|---:|
| RealCapsuleLarge (paper) | +2.703 | +5.066 |
| RealCapsuleControlV2 (fixed) | **+2.223** | **+2.561** |

### AffNIST (independent test)

| Model | Resized 28x28 | Native 40x40 |
|---|---:|---:|
| RealCapsuleLarge (published) | 62.323±0.854 | 51.956±0.744 |
| RealCapsuleLargeL1 | 62.375±0.528 | 50.364±2.614 |
| RealCapsuleControlV2Norm | 61.359±1.053 | 49.904±1.077 |
| **RealCapsuleControlV2** | **62.497±1.141** | **54.370±1.528** |
| ComplexCapsuleB (published) | 51.635±2.308 | 37.309±1.855 |

## Conclusions

1. **The confounds were real and inflated the capsule-family gap.** Fixing
   both cuts the held-out synthetic margin roughly in half (+5.07 → +2.56 pp)
   and trims the random-affine margin (+2.70 → +2.22 pp). The two fixes
   contribute roughly additively on held-out data (readout ≈ +1.4 pp,
   capacity ≈ +1.7 pp of control improvement).
2. **A genuine learned-phase advantage survives on synthetic affine data.**
   ComplexCapsuleB beats the corrected control in every seed on both random
   and held-out affine distributions. The paper's capsule-family claim stands
   in weakened form and should be restated with the new margins.
3. **The AffNIST negative result is confirmed and strengthened.** The
   corrected control scores essentially the same as the confounded one at
   28x28 (62.5%) and *better* at native 40x40 (54.4%), while ComplexCapsuleB
   remains far behind (51.6% / 37.3%). On independent data the complex capsule
   loses to a clean equal-parameter, equal-readout real control by ~11 pp
   (resized) and ~17 pp (native).

## Paper changes required

- Replace `RealCapsuleLarge` with `RealCapsuleControlV2` as the primary
  control in Sections 3.1, 5.x; report the old control and the two ablations
  in an appendix.
- Rewrite Section 6.1: "not a trivial parameter-count artifact" must
  acknowledge the original control's capacity degeneracy and readout
  handicap; the surviving margin is +2.2/+2.6 pp, not +2.7/+5.1 pp.
- Reproducibility notes: state that pre-fix runs were not seed-reproducible
  (init unseeded); post-fix runs are.
