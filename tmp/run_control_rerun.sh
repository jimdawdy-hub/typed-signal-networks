#!/usr/bin/env bash
# Rerun of the capsule-family comparison with confound-fixed real controls.
# Protocol identical to the paper's cnn-run (UNITY_NOTEBOOK.md "Full Experiment"):
# 20 epochs, mild affine augmentation, AMP, best-checkpoint selection,
# then best-checkpoint affine eval with 5 random samples per scenario.
set -euo pipefail
cd "/home/jim/Coding Projects/AI_Unity"
PY=.venv/bin/python
MODELS="real-large-l1 control-v2 control-v2-norm"
LOG=tmp/control_rerun_progress.log
echo "PID $$ started $(date -Is)" > "$LOG"

for SEED in 123 321 777 2024; do
  RUN="complex_mnist_affine_aug_mild_controls_20ep_seed${SEED}"
  echo "=== TRAIN seed ${SEED} start $(date -Is) ===" >> "$LOG"
  $PY complex-capsules/train.py \
    --epochs 20 \
    --device auto \
    --seed "$SEED" \
    --models $MODELS \
    --affine-augment random-affine-mild \
    --amp \
    --output-dir "results/${RUN}" \
    --checkpoint-dir "checkpoints/${RUN}" >> "$LOG" 2>&1
  echo "=== EVAL seed ${SEED} start $(date -Is) ===" >> "$LOG"
  $PY -m ai_unity.evaluate_complex_affines \
    --comparison-json "results/${RUN}/complex_capsules_comparison.json" \
    --checkpoint-key best_checkpoint \
    --output-dir "results/${RUN}_best_affine_eval_5samples" \
    --data-dir complex-capsules/data \
    --device auto \
    --seed "$SEED" \
    --models $MODELS \
    --random-samples 5 >> "$LOG" 2>&1
  echo "=== seed ${SEED} done $(date -Is) ===" >> "$LOG"
done
echo "ALL DONE $(date -Is)" >> "$LOG"
