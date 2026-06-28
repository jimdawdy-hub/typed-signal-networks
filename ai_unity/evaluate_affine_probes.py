from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ai_unity.complex_capsules import BaselineCNN, ResidualCNNBaseline, VisionTransformerBaseline
from ai_unity.data import get_vision_loaders
from ai_unity.evaluate_complex_affines import PHASE_FACTORS, RANDOM_AFFINES, _affine_params, _pearson
from ai_unity.evaluate_complex_rotations import MODEL_CHOICES, MODEL_FACTORIES, load_model, load_payload
from ai_unity.utils import ensure_dir, output_tuple, resolve_device, seed_everything, write_json


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Probe affine-factor information in CNN and capsule representations.")
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
        default="best_checkpoint",
    )
    p.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_CHOICES),
        default=["cnn", "vit", "real-large", "complex-b"],
    )
    p.add_argument(
        "--random-scenarios",
        nargs="+",
        choices=sorted(RANDOM_AFFINES),
        default=["heldout_affine_left_zoom", "heldout_affine_right_zoom"],
    )
    p.add_argument("--probe-train-examples", type=int, default=2000)
    p.add_argument("--probe-test-examples", type=int, default=2000)
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--min-class-examples", type=int, default=20)
    return p


def build_probe_loader(args, scenario: str):
    return get_vision_loaders(
        "mnist",
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        flatten=False,
        num_workers=args.num_workers,
        seed=args.seed,
        random_affine=RANDOM_AFFINES[scenario],
        rotation_seed=args.seed + 7919,
    )[1]


def cnn_penultimate(model: BaselineCNN, x: torch.Tensor) -> torch.Tensor:
    h = model.features(x)
    h = h.view(h.shape[0], -1)
    return model.classifier[:4](h)


def feature_blocks(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    blocks = {}
    if isinstance(model, BaselineCNN):
        blocks["cnn_penultimate"] = cnn_penultimate(model, x).detach().cpu()
    if isinstance(model, ResidualCNNBaseline):
        blocks["resnet_features"] = model.encode(x).detach().cpu()
    if isinstance(model, VisionTransformerBaseline):
        blocks["vit_cls"] = model.encode(x).detach().cpu()

    logits, _, info = output_tuple(model(x), device)
    if "digit_caps" in info:
        caps = info["digit_caps"]
        if torch.is_complex(caps):
            caps_features = torch.cat([caps.real.flatten(1), caps.imag.flatten(1), caps.abs().flatten(1)], dim=1)
        else:
            caps_features = caps.flatten(1)
        blocks["digit_caps"] = caps_features.detach().cpu()

    if "digit_phase" in info:
        phase = info["digit_phase"]
        blocks["digit_phase_all"] = torch.cat([phase.sin().flatten(1), phase.cos().flatten(1)], dim=1).detach().cpu()
        row_index = torch.arange(y.shape[0], device=device)
        true_phase = phase[row_index, y]
        blocks["digit_phase_true"] = torch.cat([true_phase.sin(), true_phase.cos()], dim=1).detach().cpu()

    if "primary_phase" in info:
        primary = info["primary_phase"]
        sin_primary = primary.sin()
        cos_primary = primary.cos()
        blocks["primary_phase_stats"] = torch.cat(
            [
                sin_primary.mean(dim=1),
                sin_primary.std(dim=1, unbiased=False),
                cos_primary.mean(dim=1),
                cos_primary.std(dim=1, unbiased=False),
            ],
            dim=1,
        ).detach().cpu()

    return blocks


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    max_examples: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    model.eval()
    feature_rows: dict[str, list[torch.Tensor]] = {}
    labels = []
    factor_rows: list[dict[str, float]] = []
    offset = 0
    collected = 0

    for x, y in loader:
        if collected >= max_examples:
            break
        take = min(x.shape[0], max_examples - collected)
        x = x[:take].to(device, non_blocking=True)
        y = y[:take].to(device, non_blocking=True)
        blocks = feature_blocks(model, x, y, device)
        for name, values in blocks.items():
            feature_rows.setdefault(name, []).append(values)
        labels.extend(y.detach().cpu().numpy().tolist())
        factor_rows.extend(_affine_params(loader.dataset, offset + row) for row in range(take))
        offset += take
        collected += take

    features = {name: torch.cat(values, dim=0).numpy() for name, values in feature_rows.items()}
    factor_arrays = {
        factor: np.asarray([row[factor] for row in factor_rows], dtype=np.float32)
        for factor in PHASE_FACTORS
    }
    return features, np.asarray(labels, dtype=np.int64), factor_arrays


def probe_feature_block(
    x: np.ndarray,
    factors: dict[str, np.ndarray],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    alpha: float,
) -> dict[str, dict[str, float]]:
    result = {}
    for factor, y in factors.items():
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(x[train_idx], y[train_idx])
        pred = model.predict(x[test_idx])
        result[factor] = {
            "r2": float(r2_score(y[test_idx], pred)),
            "mae": float(mean_absolute_error(y[test_idx], pred)),
            "target_std": float(np.std(y[test_idx])),
        }
    return result


def class_conditioned_phase_correlations(
    digit_phase_true: np.ndarray,
    labels: np.ndarray,
    factors: dict[str, np.ndarray],
    min_class_examples: int,
) -> dict[str, Any]:
    result = {}
    features = torch.from_numpy(digit_phase_true)
    for digit in range(10):
        mask_np = labels == digit
        if int(mask_np.sum()) < min_class_examples:
            continue
        mask = torch.from_numpy(mask_np)
        digit_result = {}
        for factor, values_np in factors.items():
            values = torch.from_numpy(values_np)[mask]
            correlations = [_pearson(features[mask, col], values) for col in range(features.shape[1])]
            abs_correlations = [abs(value) for value in correlations]
            best_idx = max(range(len(abs_correlations)), key=abs_correlations.__getitem__)
            digit_result[factor] = {
                "best_feature": f"phase_true_sincos_{best_idx}",
                "best_corr": correlations[best_idx],
                "best_abs_corr": abs_correlations[best_idx],
                "mean_abs_corr": float(np.mean(abs_correlations)),
                "examples": int(mask_np.sum()),
            }
        result[str(digit)] = digit_result
    return result


def main() -> None:
    args = parser().parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device, args.gpu_index)
    output_dir = ensure_dir(args.output_dir)
    comparison = load_payload(args.comparison_json)
    selected_models = {MODEL_CHOICES[model] for model in args.models}
    max_examples = args.probe_train_examples + args.probe_test_examples

    models = {}
    for name, record in comparison.items():
        if name not in MODEL_FACTORIES or name not in selected_models:
            continue
        checkpoint_value = record.get(args.checkpoint_key)
        if checkpoint_value is None:
            raise ValueError(f"{name} does not have checkpoint field {args.checkpoint_key!r}.")
        models[name] = load_model(name, Path(checkpoint_value), device)[0]

    results = {}
    rng = np.random.default_rng(args.seed)
    for scenario in args.random_scenarios:
        print(f"\nProbe scenario {scenario}")
        loader = build_probe_loader(args, scenario)
        scenario_result = {}
        for model_name, model in models.items():
            print(f"  Extracting {model_name}")
            features, labels, factors = extract_features(model, loader, device, max_examples=max_examples)
            if len(labels) < max_examples:
                raise ValueError(f"Requested {max_examples} examples but only extracted {len(labels)}.")
            permutation = rng.permutation(len(labels))
            train_idx = permutation[: args.probe_train_examples]
            test_idx = permutation[args.probe_train_examples : max_examples]

            model_result: dict[str, Any] = {
                "examples": int(len(labels)),
                "train_examples": int(len(train_idx)),
                "test_examples": int(len(test_idx)),
                "feature_blocks": {},
            }
            for block_name, x in features.items():
                print(f"    Probing {block_name} ({x.shape[1]} dims)")
                model_result["feature_blocks"][block_name] = {
                    "dims": int(x.shape[1]),
                    "factors": probe_feature_block(x, factors, train_idx, test_idx, args.alpha),
                }

            if "digit_phase_true" in features:
                model_result["class_conditioned_phase_correlations"] = class_conditioned_phase_correlations(
                    features["digit_phase_true"],
                    labels,
                    factors,
                    args.min_class_examples,
                )
            scenario_result[model_name] = model_result
        results[scenario] = scenario_result

    write_json(output_dir / "affine_probe_eval.json", results)


if __name__ == "__main__":
    main()
