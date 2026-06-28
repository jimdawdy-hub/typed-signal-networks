# Unified Neural Architecture: Test Plan v1
## Complex-Valued Capsules + MoE + Ternary Weights

**Created:** June 6, 2026  
**Researcher:** Jim Dawdy  
**AI Assistant:** Herbie (Hermes Agent)

---

## Overview

Test whether combining three novel neural network ideas produces something greater than the sum of its parts:

1. **Capsule Networks** — signals carry identity (pose/direction), not just magnitude
2. **Complex-Valued Networks** — signals carry phase as an independent axis beyond magnitude
3. **Mixture of Experts** — signal identity determines which computation path processes it

The thesis: these are **orthogonal improvements** that address different limitations of standard nets, and they compose naturally into a unified framework where signals are self-routing geometric objects.

---

## Test Sequence

### Phase 1: Complex-Valued Capsules vs Baseline ⬅️ CURRENT

**Goal:** Determine if making capsule networks complex-valued improves accuracy and/or learns meaningful phase representations.

**Models to compare:**

| Model | Description |
|---|---|
| **Baseline MLP** | Standard 784→256→256→10, real-valued, no capsules |
| **Real Capsule** | Capsule net with dynamic routing, real-valued vectors |
| **Complex Capsule (Option B)** | Capsule net where each component has magnitude + phase. Phase = spatial angle. Routing uses complex dot product (includes phase alignment). |
| **Complex Capsule (Option A)** | Capsule net where each component is a full complex number. Less interpretable but more general. |

**Option B (Phase as spatial angle):**
```
Capsule component = r∠θ  (magnitude r, phase angle θ)
Routing agreement = phase-weighted similarity
Phase should encode: rotation, spatial relationships between features
```

**Option A (Components are complex):**
```
Capsule component = a + bi  (real part + imaginary part)
Less interpretable, but more expressive
Phase emerges from training, not from explicit geometric meaning
```

**Dataset:** MNIST (28×28 handwritten digits)

**What to measure:**
- Test accuracy (primary metric)
- Training convergence speed
- Phase distribution (Option B/A): are phases learning meaningful patterns?
- Routing agreement patterns: do complex capsules route differently than real?

**Success criteria:**
- Complex capsules match or exceed real capsule accuracy
- Phase values show non-trivial distribution (not collapsed to 0)
- Qualitative: phase patterns correlate with digit identity or transformation

**Estimated time:** 2-3 days

---

### Phase 2: Complex-Valued MoE vs Baseline

**Goal:** Determine if complex-valued expert networks outperform float experts in a MoE framework.

**Models to compare:**

| Model | Description |
|---|---|
| **Float MoE** | 4 experts with float weights, learned router |
| **Complex MoE** | 4 experts with complex-valued weights, learned router |
| **Baseline MLP** | Same as Phase 1 (shared baseline) |

**Dataset:** MNIST

**What to measure:**
- Test accuracy
- Expert specialization (which experts handle which digits)
- Convergence speed
- Whether complex experts develop different phase patterns per expert

**Estimated time:** 1-2 days

---

### Phase 3: Full Combination (Complex Capsule + MoE)

**Goal:** Combine complex capsule identity with MoE routing. The capsule's complex identity IS the routing signal (no separate router).

**Models to compare:**

| Model | Description |
|---|---|
| **Baseline MLP** | Shared baseline |
| **Real Capsule** | Phase 1 result |
| **Complex Capsule (best of Option A/B)** | Phase 1 result |
| **Float MoE** | Phase 2 result |
| **Complex MoE** | Phase 2 result |
| **Complex Capsule + MoE** | THE FULL COMBO — complex capsule identity routes to complex experts |

**Key design:** No separate router network. The capsule's complex-valued identity vector (magnitude + phase pattern) determines which expert processes it. This is the unification — the signal carries identity + geometry + routing authority.

**Additional test:** Rotated MNIST
- Apply random rotations (0-360°) to test images
- Hypothesis: complex capsules (especially Option B) should handle rotation better because phase encodes spatial orientation
- This is where phase should really prove its value

**Estimated time:** 3-5 days

---

### Phase 4: Ternary Integration (If Phases 1-3 Succeed)

**Goal:** Add ternary weight quantization to the best-performing combination.

**Hypothesis:** If capsule-based MoE routing compensates for ternary precision loss (better than our earlier ternary-MoE test), then the full four-way combination is viable.

**This phase is conditional** — only pursue if Phases 1-3 show clear benefits from the complex+capsule+MoE combination.

---

## Datasets

| Dataset | Size | Difficulty | Why |
|---|---|---|---|
| MNIST | 60K train, 10K test | Easy | Fast iteration, baseline comparison |
| Rotated MNIST | Same, with random rotations | Medium | Tests phase encoding of spatial relationships |
| CIFAR-10 | 50K train, 10K test | Hard | Real-world complexity (if MNIST shows promise) |

---

## Key Questions to Answer

1. **Does complex-valued routing produce different routing patterns than real-valued routing?** (If no → complex capsules add nothing)
2. **Do phase values encode meaningful spatial information?** (If yes → this is a real contribution)
3. **Does capsule identity work as an MoE routing signal?** (If yes → eliminates the need for a separate router)
4. **Does the full combination outperform each piece individually?** (If yes → the unification is justified)
5. **Does phase help with rotated/transformed inputs?** (If yes → strong evidence for the geometric encoding hypothesis)

---

## Project Structure

```
~/workspace/
├── AI_RESEARCH_UNIFY6-26-26.md    # Full conversation + all code + references
├── UNITY_PLANV1.md                # This file — the test plan
├── UNITY_LOG.md                   # Running log of all work done
├── ternary-moe/                   # Previous work (ternary + MoE)
│   ├── model.py
│   ├── train.py
│   └── data/
└── complex-capsules/              # Phase 1 work (to be created)
    ├── model.py                   # All model variants
    ├── train.py                   # Training & comparison
    ├── results/                   # Saved results
    └── data/
```

---

## Reference Materials

**See:** `AI_RESEARCH_UNIFY6-26-26.md` for:
- Full explanation of all concepts (capsules, complex values, MoE, ternary)
- All 18 referenced papers with citations
- All 6 referenced GitHub repos
- The ternary-MoE code and results

---

## Decision Points

After each phase, assess:

| After Phase 1 | Decision |
|---|---|
| Complex capsules beat real capsules | → Proceed to Phase 2 with confidence |
| Complex capsules match real capsules | → Proceed cautiously; phase may not add value on MNIST |
| Complex capsules lose to real capsules | → Check if phase collapsed to 0; try Phase 3 on rotated MNIST before giving up |

| After Phase 2 | Decision |
|---|---|
| Complex MoE beats float MoE | → Strong signal to combine in Phase 3 |
| Complex MoE matches float MoE | → Phase 3 still worth trying (the combination might matter more than either alone) |

| After Phase 3 | Decision |
|---|---|
| Full combo beats all individual pieces | → Major result. Proceed to Phase 4 (ternary) and write up findings |
| Full combo matches best individual piece | → The ideas don't synergize; reconsider the thesis |
| Full combo loses | → Debug routing; check for phase collapse; try rotated MNIST |

---

*This plan is a living document. Update as results come in.*
