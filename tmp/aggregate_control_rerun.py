#!/usr/bin/env python3
"""Aggregate the confound-fixed control rerun against the paper's published capsule numbers.

Published (paper cnn-run) sources:
  training acc : results/complex_mnist_affine_aug_mild_cnn_20ep_seed{S}/complex_capsules_comparison.json
  random affine: results/complex_mnist_affine_aug_mild_cnn_20ep_seed{S}_best_affine_eval_5samples/
  held-out     : results/complex_mnist_affine_aug_mild_cnn_20ep_seed{S}_heldout_affine_phase_eval_5samples/

New (this rerun) sources:
  results/complex_mnist_affine_aug_mild_controls_20ep_seed{S}/ and
  results/complex_mnist_affine_aug_mild_controls_20ep_seed{S}_best_affine_eval_5samples/
"""
import json
import statistics as st
from pathlib import Path

SEEDS = [123, 321, 777, 2024]
ROOT = Path("/home/jim/Coding Projects/AI_Unity")

PUBLISHED = {
    "ComplexCapsuleB (phase=angle)": "cnn",
    "RealCapsuleLarge": "cnn",
}
NEW = ["RealCapsuleLargeL1", "RealCapsuleControlV2", "RealCapsuleControlV2Norm"]


def best_train_acc(run_dir: Path, model: str) -> float:
    d = json.load(open(run_dir / "complex_capsules_comparison.json"))
    hist = d[model]["history"]
    return max(h["test_acc"] for h in hist) * 100


def scenario_means(eval_dir: Path, model: str, prefix: str) -> float:
    d = json.load(open(eval_dir / "complex_affine_eval.json"))
    vals = [v[model]["accuracy"] for k, v in d.items()
            if k.startswith(prefix) and isinstance(v, dict) and model in v]
    assert vals, f"no {prefix}* scenarios for {model} in {eval_dir}"
    return 100 * sum(vals) / len(vals)


def agg(fn):
    vals = [fn(s) for s in SEEDS]
    return st.mean(vals), st.pstdev(vals), vals


def main():
    rows = {}
    # published capsule numbers (paper run set)
    for model in PUBLISHED:
        run = lambda s: ROOT / f"results/complex_mnist_affine_aug_mild_cnn_20ep_seed{s}"
        ev = lambda s: ROOT / f"results/complex_mnist_affine_aug_mild_cnn_20ep_seed{s}_best_affine_eval_5samples"
        ho = lambda s: ROOT / f"results/complex_mnist_affine_aug_mild_cnn_20ep_seed{s}_heldout_affine_phase_eval_5samples"
        rows[model] = {
            "train": agg(lambda s: best_train_acc(run(s), model)),
            "random": agg(lambda s: scenario_means(ev(s), model, "random_")),
            "heldout": agg(lambda s: scenario_means(ho(s), model, "heldout_")),
        }
    # new controls
    for model in NEW:
        run = lambda s: ROOT / f"results/complex_mnist_affine_aug_mild_controls_20ep_seed{s}"
        ev = lambda s: ROOT / f"results/complex_mnist_affine_aug_mild_controls_20ep_seed{s}_best_affine_eval_5samples"
        rows[model] = {
            "train": agg(lambda s: best_train_acc(run(s), model)),
            "random": agg(lambda s: scenario_means(ev(s), model, "random_")),
            "heldout": agg(lambda s: scenario_means(ev(s), model, "heldout_")),
        }

    print(f"{'Model':32s} {'TrainBest':>16s} {'RandomAffine':>16s} {'HeldOut':>16s}")
    for m, r in rows.items():
        cells = []
        for k in ("train", "random", "heldout"):
            mean, std, _ = r[k]
            cells.append(f"{mean:7.3f}±{std:5.3f}")
        print(f"{m:32s} {cells[0]:>16s} {cells[1]:>16s} {cells[2]:>16s}")

    print("\nPer-seed margins, ComplexCapsuleB minus each control (random affine / held-out):")
    cb = rows["ComplexCapsuleB (phase=angle)"]
    for m in ["RealCapsuleLarge"] + NEW:
        r = rows[m]
        dr = [a - b for a, b in zip(cb["random"][2], r["random"][2])]
        dh = [a - b for a, b in zip(cb["heldout"][2], r["heldout"][2])]
        sign_r = sum(1 for x in dr if x > 0)
        sign_h = sum(1 for x in dh if x > 0)
        print(f"  vs {m:28s} random: mean {st.mean(dr):+6.3f} pp  per-seed {[f'{x:+.2f}' for x in dr]} ({sign_r}/4 positive)")
        print(f"  {'':31s} heldout: mean {st.mean(dh):+6.3f} pp  per-seed {[f'{x:+.2f}' for x in dh]} ({sign_h}/4 positive)")

    out = {m: {k: {"mean": v[0], "pstd": v[1], "per_seed": dict(zip(map(str, SEEDS), v[2]))}
               for k, v in r.items()} for m, r in rows.items()}
    (ROOT / "tmp/control_rerun_aggregate.json").write_text(json.dumps(out, indent=2))
    print("\nWrote tmp/control_rerun_aggregate.json")


if __name__ == "__main__":
    main()
