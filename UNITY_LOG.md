# Unified Neural Architecture: Work Log

**Started:** June 6, 2026  
**Researcher:** Jim Dawdy  
**AI Assistant:** Herbie (Hermes Agent)

---

## Phase 1: Complex-Valued Capsules vs Baseline

### 2026-06-06 — Setup & Implementation

**Step 1: Project setup**
- Created `~/workspace/complex-capsules/` directory
- Set up Python 3.12 virtual environment
- Installed PyTorch (CPU-only) and torchvision

**Step 2: Implemented four model variants**

Models built:
1. **Baseline MLP** — Standard 784→256→256→10
2. **Real Capsule** — Capsule net with dynamic routing (Hinton-style)
3. **Complex Capsule Option B** — Phase as spatial angle (r∠θ)
4. **Complex Capsule Option A** — Components are full complex (a+bi)

**Step 3: Training & comparison**
- [ ] Run all four models on MNIST
- [ ] Record accuracy, convergence, phase distributions
- [ ] Compare Option A vs Option B

**Problems encountered:**
(To be filled as work progresses)

**Recommendations:**
(To be filled as results come in)

---

## Phase 2: Complex-Valued MoE vs Baseline

(Not started)

---

## Phase 3: Full Combination

(Not started)

---

## Phase 4: Ternary Integration

(Not started)

---

*This log is updated as work progresses.*
