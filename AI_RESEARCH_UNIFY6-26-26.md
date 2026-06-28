# AI Research: Unifying Novel Neural Network Architectures
## Ternary Weights + Mixture of Experts + Capsule Networks + Complex-Valued Signals

**Date:** June 6, 2026  
**Researcher:** Jim Dawdy  
**AI Assistant:** Herbie (Hermes Agent)  
**Machine:** 2015 MacBook Pro (CPU-only)

---

## Table of Contents

1. [ML Math Concepts Explained](#1-ml-math-concepts-explained)
2. [Brain vs. Neural Network Comparison](#2-brain-vs-neural-network-comparison)
3. [Floating Point Range in ML](#3-floating-point-range-in-ml)
4. [Beyond Scalar Signals: The "Red 5" Problem](#4-beyond-scalar-signals-the-red-5-problem)
5. [Giving Signals Additional State](#5-giving-signals-additional-state)
6. [The Unified Architecture Idea](#6-the-unified-architecture-idea)
7. [Compatibility Analysis](#7-compatibility-analysis)
8. [Proof of Concept: Ternary + MoE](#8-proof-of-concept-ternary--moe)
9. [Results & Analysis](#9-results--analysis)
10. [Next Steps: The Full Vision](#10-next-steps-the-full-vision)
11. [References: Papers](#11-references-papers)
12. [References: Code Repositories](#12-references-code-repositories)
13. [Source Code](#13-source-code)

---

## 1. ML Math Concepts Explained

### Activation Function

**What it is:** A gate that decides whether a neuron "fires" or not.

**Analogy:** A light switch with a dimmer. The activation function looks at how much electricity is coming in and decides: "Not enough? → Light stays off." "Enough? → Light turns on (maybe brighter as input grows)."

**Why it matters:** Without it, a neural network is just doing simple addition over and over — it can never learn curves, shapes, or anything interesting. The activation function adds the ability to learn **non-linear** patterns.

**Common ones:**
- **ReLU** — "If input is negative, output zero. Otherwise, pass it through." (The bouncer at a club — negative vibes not allowed in.)
- **Sigmoid** — Squishes any number into a value between 0 and 1. (Like a probability — "how confident am I?")
- **Tanh** — Like sigmoid but squishes to between -1 and +1. (Centered, so easier to work with.)

### Weights

**What they are:** Numbers that the model adjusts during training to get better.

**Analogy:** A mixing board in a recording studio. Each slider (weight) controls how much influence one input has. During training, the model slides each knob up and down until the output sounds right.

### Bias

**What it is:** An extra number added to the output, independent of the input.

**Analogy:** If weights are the sliders on a mixing board, bias is the **master volume knob**. It shifts the whole output up or down, giving the model flexibility even when all inputs are zero.

### Loss Function

**What it is:** A score that measures how wrong the model's prediction is.

**Analogy:** Playing darts. The loss function measures the distance between where your dart landed and the bullseye. The bigger the number, the worse you did. Training = trying to get that distance to zero.

**Common ones:**
- **MSE (Mean Squared Error)** — Average of (how far off)². Punishes big misses *way* more than small ones.
- **Cross-Entropy** — Used for classification. Measures how surprised you are by the wrong answer.

### Gradient Descent

**What it is:** The algorithm that figures out how to adjust weights to reduce the loss.

**Analogy:** Blindfolded on a hilly landscape, needing to find the lowest valley. You feel the slope under your feet and take a step **downhill**. Each step moves weights in the direction that reduces error.

**Learning rate** = how big your steps are. Too big → you overshoot the valley. Too small → you'll be there till the end of time.

### Backpropagation

**What it is:** The math that figures out "who's to blame" for the error, layer by layer, from output back to input.

**Analogy:** A relay race where the team lost. The coach starts at the finish line and works backward — "the last runner messed up because the second-to-last runner handed off badly, which happened because the first runner started too slow." Each runner (layer) gets told how much they contributed to the loss, so they can adjust.

### Epoch

**What it is:** One complete pass through the entire training dataset.

**Analogy:** Reading a textbook cover to cover once. You'll learn *something*, but you probably need to re-read it a few times (multiple epochs) to really get it.

### Overfitting

**What it is:** When the model memorizes the training data instead of learning general patterns.

**Analogy:** A student who memorizes every practice exam answer but can't solve a new problem. They ace the practice test but bomb the real one.

**Fix:** Dropout (randomly ignore some neurons during training), regularization (punish complexity), or just get more data.

### Underfitting

**What it is:** The model is too simple to capture the pattern.

**Analogy:** Trying to draw a portrait with only straight lines. The model just can't represent what's going on.

### Embedding

**What it is:** Converting something discrete (like a word) into a list of numbers (a vector) that captures its meaning.

**Analogy:** Instead of labeling a dog as "item #4,521," you describe it as [furry: 0.9, size: 0.6, friendly: 0.8]. Now the model knows that "dog" and "puppy" are *close* in this number-space, while "dog" and "airplane" are far apart.

### Softmax

**What it is:** Takes a bunch of raw numbers and turns them into probabilities that add up to 1.

**Analogy:** Three contestants with raw scores of [10, 5, 2]. Softmax turns that into something like [0.85, 0.12, 0.03] — "there's an 85% chance the answer is #1."

### Tensor

**What it is:** A multi-dimensional array of numbers.

- A single number = 0D tensor (scalar)
- A list of numbers = 1D tensor (vector)
- A spreadsheet = 2D tensor (matrix)
- A stack of spreadsheets = 3D tensor

---

## 2. Brain vs. Neural Network Comparison

### Where the Analogy Holds

**Neurons Actually Do a Weighted Sum → Fire or Don't**

A biological neuron:
1. Receives signals from other neurons through **dendrites** (inputs)
2. Each connection has a **synaptic strength** — some signals matter more than others (≈ weights)
3. The neuron sums up all incoming signals
4. If the total crosses a **threshold** → it fires an electrical spike (≈ activation function)

**Hebbian Learning ≈ Gradient Descent (In Spirit)**

"Neurons that fire together wire together." (Hebb's rule, 1949) — conceptually similar to adjusting connection strengths based on experience.

### Where the Analogy Breaks Down

**We Have No Idea How the Brain Does Backpropagation.**

- **The weight transport problem** — Backprop requires neuron A to know exactly what weights downstream neuron B used. Biological neurons don't have a way to "read" other neurons' connection strengths.
- **The update symmetry problem** — Backprop uses the *same* weights for forward and backward passes. Biology doesn't.
- **No separate "training phase"** — The brain learns continuously while running. ML typically trains, then freezes.

**The Scale Difference Is Staggering:**

| | Biological Neuron | Typical ML Node |
|---|---|---|
| Incoming connections | ~10,000 synapses (some Purkinje cells have 200,000+) | Hundreds to ~12,000 |
| Signal types | 200+ neurotransmitters | 1 floating point value |
| Internal processing | Active dendrites do local computation | One multiply + one add + one activation function |
| Timing | Precise spike timing matters (millisecond-level) | No concept of time in feedforward nets |
| Plasticity | Connections change strength continuously | Only changes during training, then frozen |

**If you tried to model a single biological neuron as an artificial neural network**, research suggests you'd need something like a **5–8 layer deep network** to capture its computational behavior. (Reference: Beniaguev et al., 2021)

```
1 biological neuron ≈ 1 small deep network
1 artificial neuron ≈ 1 multiply + 1 add + 1 threshold
```

The brain runs on **~20 watts** while training GPT-4 took **megawatts**.

---

## 3. Floating Point Range in ML

### Theoretical Range

| Precision | Range | Bits |
|---|---|---|
| Float32 | ~±3.4 × 10³⁸ | 32 |
| Float16 | ~±65,504 | 16 |
| BFloat16 | ~±3.4 × 10³⁸ (same range, less precision) | 16 |
| FP8 (E4M3) | ~±57,344 | 8 |

### Practical Range

In practice, values inside a neural network are **surprisingly constrained**:
- Activations with ReLU: **0 to maybe 5–10**
- Sigmoid: **0 to 1**
- Weights after training: usually **-2 to +2**

The *operational* range during inference is usually something like **-10 to +10**.

### The Real Precision Problem

A float32 has ~7 decimal digits of precision:
```
1.0000001 → distinguishable ✓
1.00000001 → might get rounded to 1.0 ✗
```

When multiplying billions of numbers together (as in a transformer), tiny rounding errors **accumulate**. That's why quantization (int8, int4) is a whole subfield.

### Brain Comparison

The brain doesn't use discrete numbers. A neuron's signal is an **analog electrochemical waveform** — essentially infinite precision in theory, though noisy in practice.

---

## 4. Beyond Scalar Signals: The "Red 5" Problem

### The Core Insight

A single "5" is just 5. It carries no other meaning. The trick is: **never represent anything as a single number. Always use a vector.**

```
Red   5 → [5, 0.9, 0.1, 0.0, 0.8]
Blue  5 → [5, 0.1, 0.9, 0.0, 0.8]
```

### But That's Not Enough

The deeper question: **Can a signal carry a property that isn't reducible to magnitude?**

Can the signal itself have *color* independent of its *strength*? Not "put the color in a second number," but genuinely, intrinsically, have two independent qualities at once?

Current neural networks fundamentally cannot do this. Every signal is a scalar. The hardware, the math, the algorithms — all of it assumes signals are quantities, not qualities.

---

## 5. Giving Signals Additional State

### Attempts to Give Signals "Extra State"

#### 1. Ternary / Ternary Networks
```
Binary:  {0, 1}
Ternary: {-1, 0, +1}
```
Three states. But it's still just magnitude — went from 2 magnitudes to 3.

**Reference:** Li et al., 2016 — Ternary Weight Networks

#### 2. Complex-Valued Neural Networks
Each signal is a complex number: `a + bi`
- **Magnitude** (how strong): `|z| = √(a² + b²)`
- **Phase** (where it sits in a cycle): `θ = arctan(b/a)`

These are genuinely **independent axes**. Phase is a qualitatively different property.

**Reference:** Trabelsi et al., 2018 — Deep Complex Networks ([arXiv:1705.09792](https://arxiv.org/abs/1705.09792))

#### 3. Quaternion / Octonion Networks
```
Real:      1 axis (magnitude)
Complex:   2 axes (magnitude + phase)
Quaternion: 4 axes (magnitude + 3 rotations)
Octonion:  8 axes (magnitude + 7 rotations)
```

#### 4. Spiking Neural Networks
The signal is **not a number at all**. It's an **event in time**.
```
Traditional NN signal: 0.73 (just a number)
Spiking NN signal: *spike at t=3.2ms* (a timestamp)
```
When a neuron fires carries meaning. The pattern of spikes over time carries meaning.

**Problem:** We don't have good training algorithms for spiking networks. Backprop doesn't work well because spikes are discrete events, not differentiable functions.

**Reference:** snnTorch ([GitHub](https://github.com/jeshraghian/snntorch))

#### 5. Hypernetworks / Conditioning
A signal doesn't carry a *value* but carries **instructions for how to interpret itself**. The signal says "process the *next* signal differently."

```
Normal:      output = f(weights × input)
Hypernetwork: weights = g(conditioning_signal); output = f(weights × input)
```

### The Fundamental Constraint

All of our hardware computes on scalar values. GPUs, TPUs, CPUs — they multiply and add numbers. Period. Even if you invent a richer signal type, you ultimately have to **serialize it back into numbers** to run on silicon.

The closest thing to the vision: **Spiking neural networks on neuromorphic hardware** (like Intel's Loihi chip), where signals genuinely have multiple independent properties processed with different physical mechanisms.

---

## 6. The Unified Architecture Idea

### The Breakthrough Insight

Each "exotic" architecture fills a different gap in the same underlying problem — **signals are too simple in current nets.**

| Property | Current Nets | Unified System |
|---|---|---|
| Signal identity | None — just a scalar | Capsule vector direction |
| Signal phase | None | Complex-valued activation |
| Weight efficiency | Full float32 | Ternary {-1, 0, +1} |
| Computation routing | Same path for all inputs | MoE routing by signal identity |
| Temporal meaning | None | Spike timing |

These aren't competing ideas — they're **orthogonal dimensions of the same problem**.

### A Sketch of the Unified Architecture

```
Input: complex-valued vector [a₁+b₁i, a₂+b₂i, ..., aₙ+bₙi]
       ↑
       magnitude = "what features are present"
       phase = "how features relate to each other"

    ↓

Router: looks at the vector's IDENTITY (capsule pose / phase pattern)
        → assigns to Expert k

    ↓

Expert k: computes with TERNARY weights {-1, 0, +1}
          multiplication becomes: flip sign, zero out, or pass through
          → massively cheaper computation

    ↓

Output: complex-valued capsule vector (identity preserved through layers)

    ↓

Dynamic routing: output vectors "agree" with downstream capsules
                 (Hinton's routing-by-agreement)
                 → meaning propagates through identity, not just magnitude
```

### The Key Unifying Insight

In current nets:
```
signal = number
computation = multiply + add
learning = adjust weights to reduce loss
```

In the unified system:
```
signal = vector with identity + phase + timing
computation = route-by-identity → transform-with-ternary-weights → preserve-identity
learning = adjust routing AND weights AND phase relationships
```

**The signal becomes a first-class object with properties, not just a quantity.**

---

## 7. Compatibility Analysis

### Pairwise Compatibility

| Pair | Compatible? | Why / Why Not |
|---|---|---|
| Complex-valued + Capsules | ✅ **Perfect fit** | Capsule vectors already carry identity. Complex values add phase. Same philosophy. Both use standard backprop. |
| Complex-valued + MoE | ✅ Works | Phase pattern could inform routing. No fundamental conflict. |
| Complex-valued + Ternary | ⚠️ Tricky | Ternary quantizes weights to {-1,0,+1}. Complex ternary would need to quantize both real and imaginary parts. Doable but messy. |
| Capsules + MoE | ✅ **Perfect fit** | Capsule identity IS the routing signal. No separate router needed. |
| Capsules + Ternary | ✅ Works | Ternary weights inside capsule layers. Routing mechanism is weight-independent. |
| MoE + Ternary | ✅ Works | Ternary experts. Each expert is cheap. Routing decides which cheap expert to activate. |
| Anything + Spiking | ⚠️ **Hard** | Spiking breaks backprop. Surrogate gradients needed for everything. Incompatible with "reasonable ease." |

**Recommendation:** Drop spiking. The other four play nicely together.

### The Three Best Combinations

#### 🏆 Combination 1 (Strongest, Most Novel)
**Complex-Valued Capsules + MoE Routing by Identity**

Capsule vectors with complex values, where capsule identity determines routing to experts. No separate router — the capsule's identity vector IS the routing signal.

**Estimated difficulty:** Moderate. ~1-2 weeks of coding.  
**Has anyone done this?** No. This would be novel.

#### 🥈 Combination 2 (Easiest to Implement)
**Ternary Weights + MoE**

Multiple expert networks with weights constrained to {-1, 0, +1}. Router selects which expert processes each input. Orthogonal benefits — ternary reduces weight precision, MoE reduces which weights are active.

**Estimated difficulty:** Easy. ~2-3 days.  
**Has anyone done this?** Minimally. A few efficiency papers, but not tested for "does routing compensate for quantization loss."

#### 🥉 Combination 3 (Most Novel Research Contribution)
**Complex-Valued Capsules**

Capsule network where every activation is complex-valued. Phase naturally encodes relative position — two signals in phase = aligned, out of phase = rotated.

**Estimated difficulty:** Easy-moderate. ~3-5 days.  
**Has anyone done this?** No.

### Recommended Progression
```
Ternary + MoE (easy, fast)
    → Add complex-valued experts (moderate)
        → Add capsule routing by identity (the full vision)
```

Each step is a working system you can test. Three progressively more ambitious experiments, each producing publishable results.

---

## 8. Proof of Concept: Ternary + MoE

### What Was Built

Four model variants on MNIST to test whether MoE routing compensates for the accuracy loss of ternary weight quantization:

1. **FloatSingle** — Standard network, float weights, single path (baseline)
2. **FloatMoE** — Standard network, float weights, mixture of experts
3. **TernarySingle** — Ternary weights {-1,0,+1}, single path
4. **TernaryMoE** — Ternary weights + mixture of experts (the combination)

### Key Components

#### TernaryLinear Layer
- Keeps full-precision "shadow" weights (what gets updated by backprop)
- Forward pass: ternarize to {-1, 0, +1} based on threshold (0.7 × mean absolute value)
- Backward pass: gradients flow through as if weights were float (straight-through estimator)
- **Gradual ternarization warmup:** starts with float weights, blends toward ternary over 500 training steps

#### MoE Layer
- Router: small linear network that outputs probability distribution over experts
- Top-k selection: only activates best expert(s)
- Load balancing loss: encourages router to use all experts roughly equally
- Expert usage analysis: tracks which experts handle which digit classes

### Hypothesis

TernaryMoE should close the gap between TernarySingle and FloatSingle, because routing lets each expert specialize with its own ternary pattern rather than one ternary network trying to do everything.

---

## 9. Results & Analysis

### 5-Epoch Results on MNIST

```
Model                                 Params    Train     Test     Time
----------------------------------------------------------------------
FloatSingle (baseline)               269,322   98.6%   97.8%    91.2s
FloatMoE (4 experts, top-1)          410,220   96.3%   95.7%   116.2s
TernarySingle                        269,322   94.5%   93.4%   114.5s
TernaryMoE (4 experts, top-1)        410,220   89.6%   89.6%   140.2s
```

### Key Comparison

```
FloatSingle accuracy:       97.79%
TernarySingle accuracy:     93.40%
Gap from ternary:           +4.39%

TernaryMoE accuracy:        89.61%
MoE improvement on ternary: -3.79%
Remaining gap to float:     +8.18%
```

### What This Tells Us

**MoE did NOT help ternary in this configuration.** TernaryMoE actually performed worse than TernarySingle.

Possible explanations:
1. **The MoE router itself uses float weights** — the routing decision is high-precision while the experts are low-precision, creating a mismatch
2. **Expert collapse** — despite load balancing loss, experts may not be specializing effectively
3. **Too few epochs** — MoE typically needs more training to develop specialization
4. **Model too small** — at 410K params on MNIST, there's not enough capacity for meaningful specialization
5. **The combination needs more sophisticated training** — perhaps curriculum learning where you train float MoE first, then ternarize the experts

### Expert Usage Analysis

Both FloatMoE and TernaryMoE showed some expert specialization, with different experts dominating different digit classes. But the specialization was not clean enough to compensate for ternary precision loss.

### Weight Sparsity

Ternary models achieved ~36% sparsity (fraction of zero weights), meaning ~36% of multiplications can be skipped entirely.

---

## 10. Next Steps: The Full Vision

### Immediate Improvements to Try
1. **More epochs** (20+) — MoE needs longer to develop specialization
2. **Larger model** — More capacity for experts to differentiate
3. **Top-k=2** — Let each input consult two experts instead of one
4. **More experts** (8 or 16) — More specialization paths
5. **CIFAR-10** — Harder dataset that may reveal MoE benefits
6. **Pre-train float MoE, then ternarize** — Curriculum approach

### The Research Roadmap
```
Phase 1: Ternary + MoE (current) ← you are here
    → Optimize: more experts, longer training, curriculum learning
    
Phase 2: Add complex-valued experts
    → Replace float experts with complex-valued experts
    → Phase pattern informs routing
    
Phase 3: Add capsule routing by identity
    → Capsule pose IS the routing signal
    → No separate router network
    → The full unification
```

### If It Works
This would be a publishable contribution: "A unified framework for typed, efficient, identity-preserving neural computation." Nobody has combined these ideas.

---

## 11. References: Papers

### Foundational
1. **McCulloch & Pitts, 1943** — "A Logical Calculus of Ideas Immanent in Nervous Activity" — Original neuron model as logic gate
2. **Hebb, 1949** — "The Organization of Behavior" — "Neurons that fire together wire together"
3. **Rosenblatt, 1958** — The Perceptron — First artificial neural network

### Neural Network Fundamentals
4. **Cybenko, 1989** — "Approximation by Superpositions of a Sigmoidal Function" — Universal approximation theorem
5. **Hornik et al., 1989** — "Multilayer Feedforward Networks Are Universal Approximators"

### Brain vs. ML
6. **Beniaguev, Segev & London, 2021** — "Single Cortical Neurons as Deep Artificial Neural Networks" — Showed a single biological neuron ≈ 5-8 layer DNN ([bioRxiv](https://www.biorxiv.org/content/10.1101/2021.03.29.437573v1))

### Word Embeddings
7. **Mikolov et al., 2013** — "Efficient Estimation of Word Representations in Vector Space" — word2vec

### Ternary Weight Networks
8. **Li, Zhang & Liu, 2016** — "Ternary Weight Networks" — Constraining weights to {-1, 0, +1} with threshold-based ternarization

### Complex-Valued Neural Networks
9. **Trabelsi et al., 2018** — "Deep Complex Networks" — Complex-valued neural network architectures ([arXiv:1705.09792](https://arxiv.org/abs/1705.09792))

### Capsule Networks
10. **Sabour, Frosst & Hinton, 2017** — "Dynamic Routing Between Capsules" — NIPS 2017. Capsule vectors with routing-by-agreement

### Mixture of Experts
11. **Shazeer et al., 2017** — "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" — Google's Switch Transformer precursor
12. **Fedus, Zoph & Shazeer, 2022** — "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity"

### Spiking Neural Networks
13. **Maass, 1997** — "Networks of Spiking Neurons: The Third Generation of Neural Network Models"
14. **Eshraghian et al., 2023** — "Training Spiking Neural Networks Using Lessons from Deep Learning" — snnTorch framework ([arXiv:2109.12894](https://arxiv.org/abs/2109.12894))

### Gating Mechanisms
15. **Hochreiter & Schmidhuber, 1997** — "Long Short-Term Memory" — LSTM gates
16. **Cho et al., 2014** — "Learning Phrase Representations using RNN Encoder-Decoder" — GRU gates

### Geometric / Equivariant Networks
17. **Cohen & Welling, 2016** — "Group Equivariant Convolutional Networks"
18. **Bronstein et al., 2021** — "Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges"

---

## 12. References: Code Repositories

### Capsule Networks
- [abdulfatir/capsule-networks-pytorch](https://github.com/abdulfatir/capsule-networks-pytorch) — PyTorch implementation of Dynamic Routing Between Capsules (MNIST, FashionMNIST, CIFAR10)
- [gram-ai/capsule-networks](https://github.com/gram-ai/capsule-networks) — Well-documented capsule network implementation

### Complex-Valued Neural Networks
- [mehdihosseinimoghadam/Complex-Neural-Networks](https://github.com/mehdihosseinimoghadam/Complex-Neural-Networks) — Implementation of complex valued neural networks in PyTorch

### Ternary Weight Networks
- [buaabai/Ternary-Weights-Network](https://github.com/buaabai/Ternary-Weights-Network) — PyTorch implementation of Ternary-Weights-Network for MNIST (LeNet-5)

### Spiking Neural Networks
- [jeshraghian/snntorch](https://github.com/jeshraghian/snntorch) — snnTorch: deep learning with spiking neural networks in PyTorch
- [BindsNET/bindsnet](https://github.com/BindsNET/bindsnet) — BindsNET: simulation of spiking neural networks

### Mixture of Experts
- Numerous implementations exist; the core concept is simple enough to implement from scratch (~15 lines of code for the basic mechanism)

---

## 13. Source Code

### File: `model.py`

```python
"""
Ternary + MoE: Proof of Concept
================================
Four model variants on MNIST to test whether MoE routing compensates
for the accuracy loss of ternary weight quantization.

Variants:
    1. FloatSingle   — standard network, float weights, single path
    2. FloatMoE      — standard network, float weights, mixture of experts
    3. TernarySingle — ternary weights {-1,0,+1}, single path
    4. TernaryMoE    — ternary weights + mixture of experts (the combo)

The hypothesis: TernaryMoE should close the gap between TernarySingle
and FloatSingle, because routing lets each expert specialize with its
own ternary pattern rather than one ternary network trying to do everything.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# COMPONENT 1: Ternary Weight Layer
# =============================================================================

class TernaryLinear(nn.Module):
    """
    Linear layer with ternary weights {-1, 0, +1}.
    
    Training trick: gradual ternarization.
    - Start with full float weights (train normally)
    - Ramp up ternarization over training steps
    - By end of warmup, weights are fully ternary
    
    This prevents the "dead at initialization" problem where
    aggressive quantization kills gradients before training starts.
    """
    
    def __init__(self, in_features, out_features, warmup_steps=500):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.warmup_steps = warmup_steps
        self.step_count = 0
        # Full-precision shadow weights (what gets updated by backprop)
        self.weight_float = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        # Initialize with Kaiming
        nn.init.kaiming_uniform_(self.weight_float)
    
    def ternarize(self, w):
        """Convert float weights to ternary {-1, 0, +1}."""
        threshold = 0.7 * w.abs().mean()
        w_ternary = torch.where(
            w.abs() > threshold,
            torch.sign(w),
            torch.zeros_like(w)
        )
        return w_ternary
    
    def forward(self, x):
        # Gradual ternarization: blend float and ternary during warmup
        if self.training:
            self.step_count += 1
        
        alpha = min(1.0, self.step_count / max(self.warmup_steps, 1))
        
        if self.training and alpha < 1.0:
            # Blend: mostly float at start, mostly ternary at end
            w_ternary = self.ternarize(self.weight_float)
            w_mixed = (1 - alpha) * self.weight_float + alpha * w_ternary
            return F.linear(x, w_mixed, self.bias)
        else:
            # Full ternary (inference or after warmup)
            w_ternary = self.ternarize(self.weight_float)
            return F.linear(x, w_ternary, self.bias)
    
    def sparsity(self):
        """How many weights are zero (not used)."""
        w = self.ternarize(self.weight_float)
        return (w == 0).float().mean().item()


# =============================================================================
# COMPONENT 2: Standard Float Linear Layer (for fair comparison)
# =============================================================================

class FloatLinear(nn.Module):
    """Standard linear layer — same interface as TernaryLinear for easy swapping."""
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        return self.linear(x)
    
    def sparsity(self):
        return 0.0  # no sparsity in float weights


# =============================================================================
# COMPONENT 3: Expert Block (used inside MoE)
# =============================================================================

class Expert(nn.Module):
    """A single expert: Linear → ReLU → Linear."""
    
    def __init__(self, in_features, hidden_features, out_features, use_ternary=False):
        super().__init__()
        LinearClass = TernaryLinear if use_ternary else FloatLinear
        self.net = nn.Sequential(
            LinearClass(in_features, hidden_features),
            nn.ReLU(),
            LinearClass(hidden_features, out_features),
        )
    
    def forward(self, x):
        return self.net(x)


# =============================================================================
# COMPONENT 4: Mixture of Experts Layer
# =============================================================================

class MoELayer(nn.Module):
    """
    Mixture of Experts layer.
    
    Router: small linear network that outputs a probability distribution
    over experts. Top-k experts are activated (k=1 or k=2 typically).
    
    Load balancing loss: encourages the router to use all experts roughly
    equally, preventing "expert collapse" where one expert does everything.
    """
    
    def __init__(self, in_features, hidden_features, out_features,
                 num_experts=4, top_k=1, use_ternary=False):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # The experts
        self.experts = nn.ModuleList([
            Expert(in_features, hidden_features, out_features, use_ternary)
            for _ in range(num_experts)
        ])
        
        # The router: decides which expert to use
        self.router = nn.Linear(in_features, num_experts)
    
    def forward(self, x):
        # x shape: (batch, in_features)
        batch_size = x.shape[0]
        
        # Router scores: which experts are relevant for this input?
        router_logits = self.router(x)                    # (batch, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)   # (batch, num_experts)
        
        # Top-k selection: only activate the best expert(s)
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        # Renormalize so top-k probs sum to 1
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Compute expert outputs (all experts, then select)
        expert_outputs = torch.stack([
            expert(x) for expert in self.experts
        ], dim=1)  # (batch, num_experts, out_features)
        
        # Gather the top-k expert outputs
        top_k_indices_expanded = top_k_indices.unsqueeze(-1).expand(
            -1, -1, expert_outputs.shape[-1]
        )
        selected_outputs = torch.gather(expert_outputs, 1, top_k_indices_expanded)
        
        # Weighted sum of selected expert outputs
        output = (selected_outputs * top_k_probs.unsqueeze(-1)).sum(dim=1)
        
        # Load balancing loss: penalize uneven expert usage
        avg_probs = router_probs.mean(dim=0)  # (num_experts,)
        ideal = torch.ones_like(avg_probs) / self.num_experts
        load_balance_loss = F.mse_loss(avg_probs, ideal)
        
        return output, load_balance_loss, router_probs
    
    def expert_usage(self, x):
        """Analyze which experts get used for a batch of inputs."""
        with torch.no_grad():
            router_logits = self.router(x)
            router_probs = F.softmax(router_logits, dim=-1)
            _, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
            
            usage = torch.zeros(self.num_experts)
            for i in range(self.num_experts):
                usage[i] = (top_k_indices == i).float().sum().item()
            return usage / x.shape[0]


# =============================================================================
# MODEL 1: FloatSingle — Baseline (standard network)
# =============================================================================

class FloatSingle(nn.Module):
    """Standard MLP. The baseline to beat."""
    
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10):
        super().__init__()
        self.name = "FloatSingle (baseline)"
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x):
        return self.net(x), torch.tensor(0.0)  # no aux loss


# =============================================================================
# MODEL 2: FloatMoE — Mixture of Experts with float weights
# =============================================================================

class FloatMoE(nn.Module):
    """MoE with float-weighted experts."""
    
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10,
                 num_experts=4, top_k=1):
        super().__init__()
        self.name = f"FloatMoE ({num_experts} experts, top-{top_k})"
        self.moe = MoELayer(
            input_dim, hidden_dim, output_dim,
            num_experts=num_experts, top_k=top_k, use_ternary=False
        )
    
    def forward(self, x):
        output, lb_loss, routing = self.moe(x)
        return output, lb_loss


# =============================================================================
# MODEL 3: TernarySingle — Ternary weights, single path
# =============================================================================

class TernarySingle(nn.Module):
    """Network with all ternary weights. The efficiency play."""
    
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10):
        super().__init__()
        self.name = "TernarySingle"
        self.net = nn.Sequential(
            TernaryLinear(input_dim, hidden_dim),
            nn.ReLU(),
            TernaryLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            TernaryLinear(hidden_dim, output_dim),
        )
    
    def forward(self, x):
        return self.net(x), torch.tensor(0.0)
    
    def sparsity(self):
        sparsities = []
        for module in self.net.modules():
            if isinstance(module, TernaryLinear):
                sparsities.append(module.sparsity())
        return sum(sparsities) / len(sparsities) if sparsities else 0.0


# =============================================================================
# MODEL 4: TernaryMoE — THE COMBO: Ternary experts + MoE routing
# =============================================================================

class TernaryMoE(nn.Module):
    """
    Ternary-weighted experts with MoE routing.
    
    This is the novel combination:
    - Each expert uses ternary weights (efficient)
    - Router decides which expert handles each input (specialization)
    - Hypothesis: routing compensates for ternary precision loss
    """
    
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10,
                 num_experts=4, top_k=1):
        super().__init__()
        self.name = f"TernaryMoE ({num_experts} experts, top-{top_k})"
        self.moe = MoELayer(
            input_dim, hidden_dim, output_dim,
            num_experts=num_experts, top_k=top_k, use_ternary=True
        )
    
    def forward(self, x):
        output, lb_loss, routing = self.moe(x)
        return output, lb_loss
    
    def sparsity(self):
        sparsities = []
        for expert in self.moe.experts:
            for module in expert.modules():
                if isinstance(module, TernaryLinear):
                    sparsities.append(module.sparsity())
        return sum(sparsities) / len(sparsities) if sparsities else 0.0


# =============================================================================
# Parameter counter
# =============================================================================

def count_params(model):
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model):
    """Print a concise model summary."""
    params = count_params(model)
    print(f"  {model.name}: {params:,} parameters")
    if hasattr(model, 'sparsity'):
        print(f"    Weight sparsity: {model.sparsity():.1%} (fraction of zero weights)")
    return params
```

### File: `train.py`

```python
"""
Ternary + MoE: Training & Comparison
======================================
Trains all four model variants on MNIST and produces a comparison table.

Usage:
    source .venv/bin/activate
    python train.py
    
    # Quick test (fewer epochs):
    python train.py --epochs 3
    
    # Custom settings:
    python train.py --epochs 10 --num-experts 8 --top-k 2
"""

import argparse
import time
import sys
import ssl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Fix SSL certificate verification on macOS
ssl._create_default_https_context = ssl._create_unverified_context

from model import (
    FloatSingle, FloatMoE, TernarySingle, TernaryMoE,
    count_params, model_summary, TernaryLinear
)


# =============================================================================
# Data
# =============================================================================

def get_mnist_loaders(batch_size=128):
    """Download MNIST and create data loaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),  # flatten 28x28 -> 784
    ])
    
    train_data = datasets.MNIST(
        './data', train=True, download=True, transform=transform
    )
    test_data = datasets.MNIST(
        './data', train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


# =============================================================================
# Training Loop
# =============================================================================

def train_one_epoch(model, train_loader, optimizer, device, lb_weight=0.01):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_main_loss = 0
    total_lb_loss = 0
    correct = 0
    total = 0
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits, lb_loss = model(batch_x)
        
        # Main loss: cross-entropy classification
        main_loss = F.cross_entropy(logits, batch_y)
        
        # Combined loss
        loss = main_loss + lb_weight * lb_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        total_main_loss += main_loss.item()
        total_lb_loss += lb_loss.item()
        
        preds = logits.argmax(dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_y.shape[0]
    
    return {
        'loss': total_loss / len(train_loader),
        'main_loss': total_main_loss / len(train_loader),
        'lb_loss': total_lb_loss / len(train_loader),
        'accuracy': correct / total,
    }


@torch.no_grad()
def evaluate(model, test_loader, device):
    """Evaluate on test set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        logits, _ = model(batch_x)
        loss = F.cross_entropy(logits, batch_y)
        
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_y.shape[0]
    
    return {
        'loss': total_loss / len(test_loader),
        'accuracy': correct / total,
    }


# =============================================================================
# Expert Usage Analysis
# =============================================================================

def analyze_expert_usage(model, test_loader, device):
    """For MoE models: which experts handle which digits?"""
    if not hasattr(model, 'moe'):
        return None
    
    model.eval()
    num_experts = model.moe.num_experts
    usage = torch.zeros(10, num_experts)  # usage[digit][expert] = count
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            
            router_logits = model.moe.router(batch_x)
            router_probs = torch.softmax(router_logits, dim=-1)
            _, top_k_indices = torch.topk(router_probs, model.moe.top_k, dim=-1)
            
            for i in range(batch_y.shape[0]):
                digit = batch_y[i].item()
                for k in range(model.moe.top_k):
                    expert_idx = top_k_indices[i, k].item()
                    usage[digit, expert_idx] += 1
    
    # Normalize: each row (digit) sums to 100%
    row_sums = usage.sum(dim=1, keepdim=True).clamp(min=1)
    usage_pct = usage / row_sums * 100
    return usage_pct


# =============================================================================
# Main Training Script
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Ternary + MoE comparison on MNIST")
    parser.add_argument('--epochs', type=int, default=5, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num-experts', type=int, default=4, help='Number of MoE experts')
    parser.add_argument('--top-k', type=int, default=1, help='Top-k experts to activate')
    parser.add_argument('--lb-weight', type=float, default=0.01, help='Load-balance loss weight')
    args = parser.parse_args()
    
    device = torch.device('cpu')
    print(f"Device: {device}")
    print(f"Settings: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    print(f"MoE: {args.num_experts} experts, top-{args.top_k}")
    print("=" * 70)
    
    # Data
    print("\nLoading MNIST...")
    train_loader, test_loader = get_mnist_loaders(args.batch_size)
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Test:  {len(test_loader.dataset)} samples")
    
    # Build all four models
    print("\nBuilding models...")
    models = [
        FloatSingle(),
        FloatMoE(num_experts=args.num_experts, top_k=args.top_k),
        TernarySingle(),
        TernaryMoE(num_experts=args.num_experts, top_k=args.top_k),
    ]
    
    for m in models:
        model_summary(m)
    
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    
    results = {}
    
    for model in models:
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        print(f"\n--- {model.name} ---")
        history = []
        train_start = time.time()
        
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, device, args.lb_weight
            )
            test_metrics = evaluate(model, test_loader, device)
            epoch_time = time.time() - epoch_start
            
            history.append({
                'epoch': epoch,
                'train_acc': train_metrics['accuracy'],
                'test_acc': test_metrics['accuracy'],
                'train_loss': train_metrics['main_loss'],
                'test_loss': test_metrics['loss'],
                'lb_loss': train_metrics['lb_loss'],
                'time': epoch_time,
            })
            
            sparsity_str = ""
            if hasattr(model, 'sparsity'):
                sparsity_str = f"  sparsity={model.sparsity():.1%}"
            
            print(f"  Epoch {epoch}/{args.epochs} "
                  f"| train={train_metrics['accuracy']:.1%} "
                  f"| test={test_metrics['accuracy']:.1%} "
                  f"| loss={train_metrics['main_loss']:.4f} "
                  f"| {epoch_time:.1f}s"
                  f"{sparsity_str}")
        
        total_time = time.time() - train_start
        usage = analyze_expert_usage(model, test_loader, device)
        
        results[model.name] = {
            'model': model,
            'history': history,
            'final_test_acc': history[-1]['test_acc'],
            'final_train_acc': history[-1]['train_acc'],
            'total_time': total_time,
            'params': count_params(model),
            'expert_usage': usage,
        }
    
    # Results table
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    
    print(f"\n{'Model':<35} {'Params':>8} {'Train':>8} {'Test':>8} {'Time':>8}")
    print("-" * 70)
    
    for name, r in results.items():
        print(f"{name:<35} {r['params']:>8,} "
              f"{r['final_train_acc']:>7.1%} "
              f"{r['final_test_acc']:>7.1%} "
              f"{r['total_time']:>7.1f}s")
    
    # Key comparison
    print("\n" + "=" * 70)
    print("KEY COMPARISON: Does MoE help ternary?")
    print("=" * 70)
    
    float_single_acc = results["FloatSingle (baseline)"]["final_test_acc"]
    float_moe_acc = results[f"FloatMoE ({args.num_experts} experts, top-{args.top_k})"]["final_test_acc"]
    ternary_single_acc = results["TernarySingle"]["final_test_acc"]
    ternary_moe_acc = results[f"TernaryMoE ({args.num_experts} experts, top-{args.top_k})"]["final_test_acc"]
    
    ternary_gap = float_single_acc - ternary_single_acc
    moe_helped = ternary_moe_acc - ternary_single_acc
    remaining_gap = float_single_acc - ternary_moe_acc
    
    print(f"\n  FloatSingle accuracy:       {float_single_acc:.2%}")
    print(f"  TernarySingle accuracy:     {ternary_single_acc:.2%}")
    print(f"  Gap from ternary:           {ternary_gap:+.2%}")
    print()
    print(f"  TernaryMoE accuracy:        {ternary_moe_acc:.2%}")
    print(f"  MoE improvement on ternary: {moe_helped:+.2%}")
    print(f"  Remaining gap to float:     {remaining_gap:+.2%}")
    print()
    
    if moe_helped > 0.005:
        print("  ✓ MoE ROUTING HELPED CLOSE THE TERNARY GAP.")
        print(f"    Routing recovered {moe_helped/ternary_gap:.0%} of the accuracy lost to quantization.")
    elif moe_helped > 0:
        print("  ~ MoE helped slightly, but the effect is small.")
        print("    More experts or different routing might help.")
    else:
        print("  ✗ MoE did NOT help ternary in this configuration.")
        print("    Try: more experts, different top-k, longer training, or larger model.")
    
    # Expert usage analysis
    print("\n" + "=" * 70)
    print("EXPERT USAGE BY DIGIT CLASS")
    print("=" * 70)
    
    for name, r in results.items():
        if r['expert_usage'] is not None:
            usage = r['expert_usage']
            print(f"\n  {name}:")
            print(f"  {'Digit':<8}", end="")
            for e in range(usage.shape[1]):
                print(f"{'E'+str(e):>8}", end="")
            print()
            print(f"  {'-'*(8 + 8*usage.shape[1])}")
            
            for digit in range(10):
                print(f"  {digit:<8}", end="")
                for e in range(usage.shape[1]):
                    pct = usage[digit, e].item()
                    print(f"{pct:>6.1%} ", end="")
                print()
            
            print(f"\n  Expert specialization (max % for any digit):")
            for e in range(usage.shape[1]):
                dominant_digit = usage[:, e].argmax().item()
                dominance = usage[dominant_digit, e].item()
                print(f"    Expert {e}: dominated by digit {dominant_digit} ({dominance:.1%})")
    
    # Sparsity analysis
    print("\n" + "=" * 70)
    print("WEIGHT SPARSITY (ternary models)")
    print("=" * 70)
    
    for name, r in results.items():
        model = r['model']
        if hasattr(model, 'sparsity'):
            print(f"\n  {name}: {model.sparsity():.1%} of weights are zero")
            print(f"    Effective weights: {{-1, 0, +1}} instead of full float32")
            print(f"    Multiply becomes: sign flip or skip (much cheaper)")
    
    # Save results
    output_file = "results.txt"
    with open(output_file, 'w') as f:
        f.write("Ternary + MoE: MNIST Comparison Results\n")
        f.write("=" * 50 + "\n\n")
        for name, r in results.items():
            f.write(f"{name}\n")
            f.write(f"  Parameters: {r['params']:,}\n")
            f.write(f"  Train accuracy: {r['final_train_acc']:.2%}\n")
            f.write(f"  Test accuracy: {r['final_test_acc']:.2%}\n")
            f.write(f"  Training time: {r['total_time']:.1f}s\n\n")
        f.write(f"\nKey finding:\n")
        f.write(f"  Ternary gap: {ternary_gap:+.2%}\n")
        f.write(f"  MoE helped: {moe_helped:+.2%}\n")
        f.write(f"  Remaining gap: {remaining_gap:+.2%}\n")
    
    print(f"\nResults saved to {output_file}")
    print("\nDone! 🏄")


if __name__ == "__main__":
    main()
```

---

## Appendix: Project Structure

```
~/workspace/ternary-moe/
├── .venv/                  # Python 3.12 virtual environment
├── data/                   # MNIST dataset (auto-downloaded)
├── model.py                # All model definitions
├── train.py                # Training & comparison script
├── results.txt             # Saved results after training
└── AI_RESEARCH_UNIFY6-26-26.md  # This document
```

### Setup
```bash
cd ~/workspace/ternary-moe
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install "numpy<2"
```

### Run
```bash
python train.py --epochs 5
python train.py --epochs 10 --num-experts 8 --top-k 2
```

---

*Document generated by Herbie (Hermes Agent) on June 6, 2026.*
codex resume 019f079b-32b6-7cc2-8f4e-d1af080c975c

