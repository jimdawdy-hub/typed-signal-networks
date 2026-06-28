from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import Subset

from ai_unity.data import get_vision_loaders
from ai_unity.evaluate_complex_rotations import (
    MODEL_CHOICES,
    MODEL_FACTORIES,
    evaluate_loaders,
    load_model,
    load_payload,
)
from ai_unity.utils import ensure_dir, resolve_device, seed_everything, write_json


FIXED_AFFINES = {
    "clean": {"degrees": 0.0, "translate": (0, 0), "scale": 1.0, "shear": (0.0, 0.0)},
    "translate4": {"degrees": 0.0, "translate": (4, 4), "scale": 1.0, "shear": (0.0, 0.0)},
    "translate8": {"degrees": 0.0, "translate": (8, 8), "scale": 1.0, "shear": (0.0, 0.0)},
    "scale085": {"degrees": 0.0, "translate": (0, 0), "scale": 0.85, "shear": (0.0, 0.0)},
    "scale115": {"degrees": 0.0, "translate": (0, 0), "scale": 1.15, "shear": (0.0, 0.0)},
    "shear15": {"degrees": 0.0, "translate": (0, 0), "scale": 1.0, "shear": (15.0, 0.0)},
    "affine_moderate": {"degrees": 30.0, "translate": (4, 4), "scale": 0.9, "shear": (10.0, 0.0)},
    "affine_strong": {"degrees": 45.0, "translate": (6, 6), "scale": 0.8, "shear": (18.0, 8.0)},
}

RANDOM_AFFINES = {
    "random_translate": {
        "degrees": (0.0, 0.0),
        "translate": (0.25, 0.25),
        "scale": (1.0, 1.0),
        "shear_x": (0.0, 0.0),
        "shear_y": (0.0, 0.0),
    },
    "random_translate_scale": {
        "degrees": (0.0, 0.0),
        "translate": (0.20, 0.20),
        "scale": (0.80, 1.20),
        "shear_x": (0.0, 0.0),
        "shear_y": (0.0, 0.0),
    },
    "random_affine_mild": {
        "degrees": (-30.0, 30.0),
        "translate": (0.15, 0.15),
        "scale": (0.85, 1.15),
        "shear_x": (-15.0, 15.0),
        "shear_y": (0.0, 0.0),
    },
    "random_affine_strong": {
        "degrees": (-45.0, 45.0),
        "translate": (0.25, 0.25),
        "scale": (0.75, 1.25),
        "shear_x": (-25.0, 25.0),
        "shear_y": (-10.0, 10.0),
    },
    "heldout_affine_left_zoom": {
        "degrees": (-60.0, -30.0),
        "translate": (0.25, 0.25),
        "scale": (0.70, 0.85),
        "shear_x": (-30.0, -15.0),
        "shear_y": (-15.0, 0.0),
    },
    "heldout_affine_right_zoom": {
        "degrees": (30.0, 60.0),
        "translate": (0.25, 0.25),
        "scale": (1.15, 1.35),
        "shear_x": (15.0, 30.0),
        "shear_y": (0.0, 15.0),
    },
}

PHASE_FACTORS = (
    "angle_degrees",
    "translate_x_frac",
    "translate_y_frac",
    "scale",
    "shear_x_degrees",
    "shear_y_degrees",
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate trained complex capsule checkpoints on synthetic affine MNIST.")
    p.add_argument("--comparison-json", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=Path("complex-capsules/data"))
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument(
        "--checkpoint-key",
        choices=["checkpoint", "best_checkpoint", "latest_checkpoint"],
        default="checkpoint",
        help="Comparison JSON checkpoint field to load.",
    )
    p.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_CHOICES),
        default=None,
        help="Optional model subset to evaluate. Defaults to every supported model in the comparison JSON.",
    )
    p.add_argument(
        "--fixed-scenarios",
        nargs="+",
        choices=sorted(FIXED_AFFINES),
        default=list(FIXED_AFFINES),
    )
    p.add_argument(
        "--random-scenarios",
        nargs="+",
        choices=sorted(RANDOM_AFFINES),
        default=list(RANDOM_AFFINES),
    )
    p.add_argument("--random-samples", type=int, default=1)
    p.add_argument(
        "--phase-diagnostics",
        action="store_true",
        help="Write phase/affine-factor correlation diagnostics for complex models on random affine scenarios.",
    )
    p.add_argument("--limit-test-batches", type=int, default=None)
    return p


def build_test_loaders(args, *, affine=None, random_affine=None, affine_seed=None) -> dict:
    flat_loader = get_vision_loaders(
        "mnist",
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        flatten=True,
        num_workers=args.num_workers,
        seed=args.seed,
        affine=affine,
        random_affine=random_affine,
        rotation_seed=affine_seed,
    )[1]
    img_loader = get_vision_loaders(
        "mnist",
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        flatten=False,
        num_workers=args.num_workers,
        seed=args.seed,
        affine=affine,
        random_affine=random_affine,
        rotation_seed=affine_seed,
    )[1]
    return {"flat": flat_loader, "img": img_loader}


def _affine_params(dataset, idx: int) -> dict[str, float]:
    if isinstance(dataset, Subset):
        return _affine_params(dataset.dataset, int(dataset.indices[idx]))
    if hasattr(dataset, "affine_params"):
        return dataset.affine_params(idx)
    raise TypeError(f"Dataset {type(dataset).__name__} does not expose affine_params().")


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float() - x.float().mean()
    y = y.float() - y.float().mean()
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if denom.item() == 0.0:
        return 0.0
    return float((x * y).sum().div(denom).item())


@torch.no_grad()
def phase_factor_diagnostics(models: dict, loader, device: torch.device, limit_batches: int | None) -> dict:
    diagnostics = {}
    dataset = loader.dataset
    for name, (model, data_key) in models.items():
        if data_key != "img" or "Complex" not in name:
            continue

        phase_rows = []
        factor_rows: list[dict[str, float]] = []
        offset = 0
        model.eval()
        for batch_idx, (x, y) in enumerate(loader):
            if limit_batches is not None and batch_idx >= limit_batches:
                break
            batch_size = y.shape[0]
            x = x.to(device, non_blocking=True)
            y_device = y.to(device, non_blocking=True)
            _, info = model(x)
            if "digit_phase" not in info:
                offset += batch_size
                continue
            phases = info["digit_phase"]
            row_index = torch.arange(batch_size, device=device)
            phase_rows.append(phases[row_index, y_device].detach().cpu())
            factor_rows.extend(_affine_params(dataset, offset + row) for row in range(batch_size))
            offset += batch_size

        if not phase_rows:
            continue

        phase = torch.cat(phase_rows, dim=0)
        factor_tensors = {
            factor: torch.tensor([row[factor] for row in factor_rows], dtype=torch.float32)
            for factor in PHASE_FACTORS
        }
        features = torch.cat([phase.sin(), phase.cos()], dim=1)
        phase_dim = phase.shape[1]
        feature_names = [f"sin_dim_{idx}" for idx in range(phase_dim)] + [f"cos_dim_{idx}" for idx in range(phase_dim)]

        factor_results = {}
        for factor, values in factor_tensors.items():
            correlations = [_pearson(features[:, col], values) for col in range(features.shape[1])]
            abs_correlations = [abs(value) for value in correlations]
            best_idx = max(range(len(abs_correlations)), key=abs_correlations.__getitem__)
            factor_results[factor] = {
                "best_feature": feature_names[best_idx],
                "best_corr": correlations[best_idx],
                "best_abs_corr": abs_correlations[best_idx],
                "mean_abs_corr": float(torch.tensor(abs_correlations).mean().item()),
            }

        diagnostics[name] = {
            "examples": int(phase.shape[0]),
            "phase_dims": int(phase_dim),
            "phase_std": float(phase.std(unbiased=False).item()),
            "factor_correlations": factor_results,
        }
    return diagnostics


def main() -> None:
    args = parser().parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device, args.gpu_index)
    output_dir = ensure_dir(args.output_dir)
    comparison = load_payload(args.comparison_json)

    selected_models = None
    if args.models is not None:
        selected_models = {MODEL_CHOICES[model] for model in args.models}

    models = {}
    for name, record in comparison.items():
        if name not in MODEL_FACTORIES:
            continue
        if selected_models is not None and name not in selected_models:
            continue
        checkpoint_value = record.get(args.checkpoint_key)
        if checkpoint_value is None:
            raise ValueError(f"{name} does not have checkpoint field {args.checkpoint_key!r}.")
        models[name] = load_model(name, Path(checkpoint_value), device)

    results = {}
    for scenario in args.fixed_scenarios:
        loaders = build_test_loaders(args, affine=FIXED_AFFINES[scenario])
        results[scenario] = evaluate_loaders(
            f"Affine scenario {scenario}",
            models,
            loaders,
            device,
            args.limit_test_batches,
        )

    phase_diagnostics = {}
    for scenario in args.random_scenarios:
        for sample_idx in range(args.random_samples):
            seed = args.seed + 2003 * (sample_idx + 1)
            key = f"{scenario}_sample{sample_idx + 1}"
            loaders = build_test_loaders(args, random_affine=RANDOM_AFFINES[scenario], affine_seed=seed)
            results[key] = evaluate_loaders(
                f"Random affine scenario {scenario} sample {sample_idx + 1}",
                models,
                loaders,
                device,
                args.limit_test_batches,
            )
            if args.phase_diagnostics:
                diagnostics = phase_factor_diagnostics(models, loaders["img"], device, args.limit_test_batches)
                if diagnostics:
                    phase_diagnostics[key] = diagnostics
                    for name, item in diagnostics.items():
                        best = max(
                            item["factor_correlations"].items(),
                            key=lambda row: row[1]["best_abs_corr"],
                        )
                        print(
                            f"  {name} phase diagnostic: strongest={best[0]} "
                            f"abs_corr={best[1]['best_abs_corr']:.3f} feature={best[1]['best_feature']}"
                        )

    write_json(output_dir / "complex_affine_eval.json", results)
    if phase_diagnostics:
        write_json(output_dir / "complex_affine_phase_diagnostics.json", phase_diagnostics)


if __name__ == "__main__":
    main()
