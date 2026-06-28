# Complex Capsules: Project Handover
## Ready to run on a GPU machine

**Created:** June 26, 2026  
**Source Machine:** 2015 MacBook Pro (CPU only, too slow)  
**Target:** Any machine with Python 3.10+, PyTorch 2.0+, optional GPU

---

## What Is This Project?

A research proof-of-concept testing whether **complex-valued capsule networks** outperform standard real-valued capsule networks on image classification.

### The Core Idea

Standard neural network signals are just **one number** (a scalar). This project explores giving signals **two independent properties**:
- **Magnitude** — what features are present (how strong)
- **Phase** — how features relate spatially (rotation/position)

This maps to complex numbers: `a + bi` where `|z| = magnitude` and `θ = phase`.

**Hypothesis:** Complex-valued capsules should handle **rotated/transformed images** better than real-valued capsules, because phase naturally encodes spatial relationships.

### Three-Phase Roadmap

| Phase | What | Status | Difficulty |
|---|---|---|---|
| **1** | Complex + Capsule vs baseline | ⬜ Not started | Easy-Moderate |
| **2** | Complex + MoE vs baseline | ⬜ Not started | Easy |
| **3** | Complex + MoE + Capsule (full combo) | ⬜ Not started | Moderate-Hard |

Phase 1 is the foundation. Phases 2 and 3 build on it.

---

## Project Structure

```
complex-capsules/
├── README.md                           # This file
├── model.py                            # ✅ All model definitions (READY)
├── train.py                            # ✅ Training & comparison script (READY)
├── HANDOVER.md                         # This handover document
├── .gitignore
├── data/                               # MNIST (auto-downloaded)
└── .venv/                              # Python virtual environment (create fresh)
```

**Both files (`model.py` and `train.py`) are complete and tested for Phase 1.** The code has not been run on MNIST yet (the MacBook was too slow).

---

## Setup (5 minutes)

```bash
# 1. Clone or copy the project
cd ~/workspace/complex-capsules   # or wherever you put it

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install PyTorch
# For GPU (CUDA):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# For CPU only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# For Mac M1/M2 (MPS):
pip install torch torchvision

# 4. Install dependencies
pip install "numpy<2"

# 5. Verify
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

---

## How to Run

### Phase 1: Complex + Capsule (START HERE)

```bash
cd ~/workspace/complex-capsules
source .venv/bin/activate

# Standard run (5 epochs, MNIST)
python train.py --epochs 5

# Longer training
python train.py --epochs 20

# Quick test
python train.py --epochs 2
```

This trains **4 models** and prints a comparison table:

| Model | What It Tests |
|---|---|
| **BaselineMLP** | Standard MLP, no capsules (reference point) |
| **StandardCapsuleNet** | Real-valued capsule network with dynamic routing |
| **ComplexCapsuleNet** | Complex-valued capsule network ← THE EXPERIMENT |
| **TransformInvariantTest** | Tests whether complex capsules handle rotation better |

### Phase 2: Complex + MoE (after Phase 1)

Not yet implemented. See "What's Left to Build" below.

### Phase 3: Complex + MoE + Capsule (after Phase 2)

Not yet implemented. See "What's Left to Build" below.

---

## What's Already Built (model.py)

### 1. BaselineMLP
Standard 3-layer MLP. Reference point for "what does a normal net get?"

### 2. ComplexLinear
A linear layer where weights AND inputs are complex-valued.
- Weights stored as real (2× size): `[real_part, imag_part]`
- Forward pass performs complex matrix multiplication
- Backward pass works through PyTorch's autograd

### 3. ComplexSqrt / complex_relu / complex_leaky_relu
Complex activation functions. These are **non-trivial** because there's no natural ordering for complex numbers. Options implemented:
- **Modulus clamping:** `ReLU(|z|) * z/|z|` — keeps phase, clamps magnitude
- **Leaky version:** `LeakyReLU(|z|) * z/|z|`

### 4. ComplexPrimaryCapsuleLayer
Converts a feature map into primary capsules:
- Each capsule = a complex-valued vector (e.g., 8 complex components = 16 real numbers)
- Squashing function preserves phase, compresses magnitude to [0, 1)

### 5. ComplexDigitCapsuleLayer
The main capsule layer with **routing-by-agreement in complex space**:
- Agreement = `Re(z_child · z_parent*)` — real part of complex dot product
- This measures both magnitude similarity AND phase alignment
- Iterative routing (3 rounds by default)
- Parent capsules represent digit classes (10 capsules, 16 complex dims each)

### 6. StandardCapsuleNet
Real-valued capsule network. Same architecture as ComplexCapsuleNet but uses standard real arithmetic.

### 7. ComplexCapsuleNet
**The main experiment.** Full complex-valued capsule network:
- Conv layer → Primary capsules → Digit capsules (with routing) → Output

### 8. TransformInvariantTest
A small network that tests whether complex-valued features encode rotation:
- Extract features from an image
- Extract features from the rotated version
- Compare: are the complex features related by a phase shift?

---

## What's Left to Build

### Phase 2: Complex + MoE

```python
# Need to implement:
class ComplexMoELayer(nn.Module):
    """MoE with complex-valued experts."""
    def __init__(self, ...):
        self.experts = nn.ModuleList([
            ComplexExpert(...) for _ in range(num_experts)
        ])
        self.router = nn.Linear(...)  # router can stay real-valued

    def forward(self, x):
        # x is complex-valued
        # Router decides which expert
        # Expert processes in complex space
        ...
```

**Key decision:** Should the router be real-valued (simpler) or complex-valued (consistent)? I'd start with real-valued router, complex experts.

**Test:** Compare `FloatMoE` vs `ComplexMoE` vs `BaselineMLP`

### Phase 3: Complex + MoE + Capsule

The full unification:
- Capsule identity (complex-valued) IS the MoE routing signal
- No separate router network
- Phase pattern determines which expert processes the capsule

**This is the novel contribution. Nobody has published this.**

---

## Key Design Decisions Made

### 1. Phase Preservation in Activations

Standard ReLU destroys phase information. Our complex activations **preserve phase, clamp magnitude**:

```python
def complex_relu(z):
    magnitude = torch.abs(z)
    phase = z / (magnitude + 1e-8)
    return torch.relu(magnitude) * phase  # phase survives!
```

This is critical — if you destroy phase, you lose the whole point of complex-valued networks.

### 2. Routing Agreement Uses Complex Dot Product

Hinton's routing uses real dot product. We use:

```python
agreement = torch.real(torch.sum(prediction * parent.conj(), dim=-1))
```

This captures **phase alignment** — two capsules that are "in phase" (spatially coherent) score higher than two that are out of phase.

### 3. Complex Weights Stored as 2× Real

PyTorch supports complex tensors, but autograd can be finicky. We store weights as real tensors with shape `[..., 2]` where `[..., 0]` is real part and `[..., 1]` is imaginary part. This gives full autograd support.

### 4. Classification Uses Magnitude Only

The final output capsule's **magnitude** = classification confidence. **Phase** carries spatial information but doesn't directly affect the class prediction. This is intentional — we want phase to be an internal representation, not an output.

---

## Expected Results

### Phase 1 Predictions

| Model | Standard MNIST | Rotated MNIST |
|---|---|---|
| BaselineMLP | ~97-98% | ~70-80% (drops a lot) |
| StandardCapsuleNet | ~99% | ~85-90% (better) |
| ComplexCapsuleNet | ~99% | ~90-95%? (phase encodes rotation) |

The **key test** is rotated MNIST. If complex capsules outperform real capsules on rotated digits, that's evidence that phase encoding captures spatial relationships.

### What Would Constitute a Positive Result

1. ComplexCapsuleNet matches or beats StandardCapsuleNet on standard MNIST
2. ComplexCapsuleNet significantly outperforms StandardCapsuleNet on **rotated MNIST** (this is the big one)
3. Analysis shows learned phase values correlate with rotation angle (this proves phase is encoding spatial info)

---

## Troubleshooting

### "No module named 'torch'"
```bash
source .venv/bin/activate
pip install torch torchvision
```

### MNIST download fails (SSL error on macOS)
```python
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```
(Already handled in train.py)

### Out of memory on GPU
```bash
python train.py --epochs 5 --batch-size 64  # smaller batches
```

### Training is slow
- GPU: Should take ~2-5 minutes for 5 epochs on MNIST
- CPU: 10-30 minutes depending on hardware
- If still slow, reduce `num_routing_iterations` in ComplexDigitCapsuleLayer from 3 to 2

---

## References

### Papers
1. **Sabour, Frosst & Hinton, 2017** — "Dynamic Routing Between Capsules" — NIPS 2017
2. **Trabelsi et al., 2018** — "Deep Complex Networks" — [arXiv:1705.09792](https://arxiv.org/abs/1705.09792)
3. **Shazeer et al., 2017** — "Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer"
4. **Beniaguev, Segev & London, 2021** — "Single Cortical Neurons as Deep Artificial Neural Networks" — [bioRxiv](https://www.biorxiv.org/content/10.1101/2021.03.29.437573v1)

### Code References
- [abdulfatir/capsule-networks-pytorch](https://github.com/abdulfatir/capsule-networks-pytorch)
- [mehdihosseinimoghadam/Complex-Neural-Networks](https://github.com/mehdihosseinimoghadam/Complex-Neural-Networks)
- [buaabai/Ternary-Weights-Network](https://github.com/buaabai/Ternary-Weights-Network)

### Related Project
- `~/workspace/ternary-moe/` — Ternary + MoE proof of concept (separate experiment, already run)

---

## Full Conversation Log

The complete research conversation that led to this project is documented in:
`/Users/jimdawdy/AI_RESEARCH_UNIFY6-26-26.md`

This includes:
- ML math concepts explained with analogies
- Brain vs neural network comparison
- The "Red 5" problem (giving signals additional meaning)
- Why scalars are limiting and what alternatives exist
- Compatibility analysis of capsule/complex/MoE/ternary/spiking architectures
- The unified architecture vision

---

*Handover prepared by Herbie (Hermes Agent), June 26, 2026.*
