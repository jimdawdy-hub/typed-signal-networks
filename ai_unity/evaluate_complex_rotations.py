from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ai_unity.complex_capsules import (
    BaselineCNN,
    BaselineMLP,
    ComplexCapsuleNetA,
    ComplexCapsuleNetB,
    ComplexCapsuleNetSpatialB,
    RealCapsuleNet,
    RealCapsuleNetLarge,
    ResidualCNNBaseline,
    VisionTransformerBaseline,
)
from ai_unity.data import get_vision_loaders
from ai_unity.training import evaluate
from ai_unity.utils import ensure_dir, resolve_device, seed_everything, write_json


MODEL_FACTORIES = {
    "BaselineMLP": (BaselineMLP, "flat"),
    "BaselineCNN": (BaselineCNN, "img"),
    "ResidualCNNBaseline": (ResidualCNNBaseline, "img"),
    "VisionTransformerBaseline": (VisionTransformerBaseline, "img"),
    "RealCapsule": (RealCapsuleNet, "img"),
    "RealCapsuleLarge": (RealCapsuleNetLarge, "img"),
    "ComplexCapsuleB (phase=angle)": (ComplexCapsuleNetB, "img"),
    "ComplexCapsuleSpatialB (phase=spatial_angle)": (ComplexCapsuleNetSpatialB, "img"),
    "ComplexCapsuleA (full complex)": (ComplexCapsuleNetA, "img"),
}

MODEL_CHOICES = {
    "baseline": "BaselineMLP",
    "cnn": "BaselineCNN",
    "resnet": "ResidualCNNBaseline",
    "vit": "VisionTransformerBaseline",
    "real": "RealCapsule",
    "real-large": "RealCapsuleLarge",
    "complex-b": "ComplexCapsuleB (phase=angle)",
    "spatial-b": "ComplexCapsuleSpatialB (phase=spatial_angle)",
    "complex-a": "ComplexCapsuleA (full complex)",
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate trained complex capsule checkpoints on rotated MNIST tests.")
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
    p.add_argument("--rotations", type=float, nargs="+", default=[0, 15, 30, 45, 60, 75, 90])
    p.add_argument(
        "--random-rotations",
        type=float,
        nargs=2,
        action="append",
        default=[],
        metavar=("MIN_DEGREES", "MAX_DEGREES"),
        help="Evaluate randomly rotated test sets. Repeat for multiple ranges, e.g. --random-rotations -30 30.",
    )
    p.add_argument(
        "--random-samples",
        type=int,
        default=1,
        help="Number of independently sampled test sets per random rotation range.",
    )
    p.add_argument("--limit-test-batches", type=int, default=None)
    return p


def load_payload(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a comparison object.")
    return payload


def load_model(name: str, checkpoint: Path, device: torch.device):
    factory, data_key = MODEL_FACTORIES[name]
    model = factory().to(device)
    payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, data_key


def rotation_key(degrees: float) -> str:
    return str(int(degrees)) if float(degrees).is_integer() else str(degrees)


def range_key(low: float, high: float, sample_idx: int) -> str:
    def fmt(value: float) -> str:
        text = str(int(value)) if float(value).is_integer() else str(value)
        return text.replace("-", "neg").replace(".", "p")

    return f"random_{fmt(low)}_{fmt(high)}_sample{sample_idx + 1}"


def evaluate_loaders(
    label: str,
    models: dict,
    loaders: dict,
    device: torch.device,
    limit_test_batches: int | None,
) -> dict:
    print(f"\n{label}")
    result = {}
    for name, (model, data_key) in models.items():
        metrics = evaluate(model, loaders[data_key], device, limit_batches=limit_test_batches)
        result[name] = {
            "accuracy": metrics["accuracy"],
            "loss": metrics["loss"],
            "phase_mean": metrics.get("phase_mean"),
            "phase_std": metrics.get("phase_std"),
            "phase_collapsed": metrics.get("phase_collapsed"),
        }
        phase = ""
        if "phase_std" in metrics:
            phase = f" phase_std={metrics['phase_std']:.3f}"
        print(f"  {name}: acc={metrics['accuracy']:.2%} loss={metrics['loss']:.4f}{phase}")
    return result


def build_test_loaders(args, *, rotate_degrees=None, random_rotate_degrees=None, rotation_seed=None) -> dict:
    flat_loader = get_vision_loaders(
        "mnist",
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        flatten=True,
        num_workers=args.num_workers,
        seed=args.seed,
        rotate_degrees=rotate_degrees,
        random_rotate_degrees=random_rotate_degrees,
        rotation_seed=rotation_seed,
    )[1]
    img_loader = get_vision_loaders(
        "mnist",
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        flatten=False,
        num_workers=args.num_workers,
        seed=args.seed,
        rotate_degrees=rotate_degrees,
        random_rotate_degrees=random_rotate_degrees,
        rotation_seed=rotation_seed,
    )[1]
    return {"flat": flat_loader, "img": img_loader}


def main() -> None:
    args = parser().parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device, args.gpu_index)
    output_dir = ensure_dir(args.output_dir)
    comparison = load_payload(args.comparison_json)

    models = {}
    selected_models = None
    if args.models is not None:
        selected_models = {MODEL_CHOICES[model] for model in args.models}
    for name, record in comparison.items():
        if name not in MODEL_FACTORIES:
            continue
        if selected_models is not None and name not in selected_models:
            continue
        checkpoint_value = record.get(args.checkpoint_key)
        if checkpoint_value is None:
            raise ValueError(f"{name} does not have checkpoint field {args.checkpoint_key!r}.")
        checkpoint = Path(checkpoint_value)
        models[name] = load_model(name, checkpoint, device)

    results = {}
    for degrees in args.rotations:
        loaders = build_test_loaders(args, rotate_degrees=degrees)
        results[rotation_key(degrees)] = evaluate_loaders(
            f"Rotation {degrees:g} degrees",
            models,
            loaders,
            device,
            args.limit_test_batches,
        )

    for low, high in args.random_rotations:
        if low > high:
            raise ValueError(f"Invalid random rotation range: {low} > {high}")
        for sample_idx in range(args.random_samples):
            seed = args.seed + 1009 * (sample_idx + 1)
            loaders = build_test_loaders(
                args,
                random_rotate_degrees=(low, high),
                rotation_seed=seed,
            )
            key = range_key(low, high, sample_idx)
            results[key] = evaluate_loaders(
                f"Random rotation [{low:g}, {high:g}] sample {sample_idx + 1}",
                models,
                loaders,
                device,
                args.limit_test_batches,
            )

    write_json(output_dir / "complex_rotation_eval.json", results)


if __name__ == "__main__":
    main()
