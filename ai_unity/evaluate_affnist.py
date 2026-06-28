from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from ai_unity.data import AffNISTDataset
from ai_unity.evaluate_complex_rotations import MODEL_CHOICES, MODEL_FACTORIES, evaluate_loaders, load_model, load_payload
from ai_unity.utils import ensure_dir, resolve_device, seed_everything, write_json


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate trained MNIST checkpoints on resized AffNIST.")
    p.add_argument("--comparison-json", type=Path, nargs="+", required=True)
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
    p.add_argument("--resize", type=int, default=28)
    p.add_argument("--test-subset", type=int, default=None)
    p.add_argument("--limit-test-batches", type=int, default=None)
    p.add_argument("--no-download", action="store_true")
    return p


def build_affnist_loaders(args, data_keys: set[str]) -> dict:
    loaders = {}
    if "img" in data_keys:
        img_transform = transforms.Compose(
            [
                transforms.Resize((args.resize, args.resize)),
                transforms.ToTensor(),
            ]
        )
        img_data = AffNISTDataset(args.data_dir, split="test", download=not args.no_download, transform=img_transform)
        if args.test_subset is not None:
            img_data = Subset(img_data, range(min(args.test_subset, len(img_data))))
        loaders["img"] = DataLoader(
            img_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=False,
        )
    if "flat" in data_keys:
        flat_transform = transforms.Compose(
            [
                transforms.Resize((args.resize, args.resize)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.view(-1)),
            ]
        )
        flat_data = AffNISTDataset(args.data_dir, split="test", download=not args.no_download, transform=flat_transform)
        if args.test_subset is not None:
            flat_data = Subset(flat_data, range(min(args.test_subset, len(flat_data))))
        loaders["flat"] = DataLoader(
            flat_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=False,
        )
    return loaders


def merged_comparison(paths: list[Path]) -> dict:
    comparison = {}
    for path in paths:
        for name, record in load_payload(path).items():
            if name in comparison:
                raise ValueError(f"Model {name!r} appears in more than one comparison JSON.")
            comparison[name] = record
    return comparison


def main() -> None:
    args = parser().parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device, args.gpu_index)
    output_dir = ensure_dir(args.output_dir)
    comparison = merged_comparison(args.comparison_json)
    selected_models = {MODEL_CHOICES[model] for model in args.models}

    models = {}
    for name, record in comparison.items():
        if name not in MODEL_FACTORIES or name not in selected_models:
            continue
        checkpoint_value = record.get(args.checkpoint_key)
        if checkpoint_value is None:
            raise ValueError(f"{name} does not have checkpoint field {args.checkpoint_key!r}.")
        models[name] = load_model(name, Path(checkpoint_value), device)

    loaders = build_affnist_loaders(args, {data_key for _, data_key in models.values()})
    results = {
        "metadata": {
            "dataset": "AffNIST transformed test set",
            "source": "https://www.cs.toronto.edu/~tijmen/affNIST/32x/transformed/test.mat.zip",
            "original_size": 40,
            "resize": args.resize,
            "test_subset": args.test_subset,
            "comparison_json": [str(path) for path in args.comparison_json],
        },
        "affnist": evaluate_loaders(
            f"AffNIST resized {args.resize}x{args.resize}",
            models,
            loaders,
            device,
            args.limit_test_batches,
        ),
    }
    write_json(output_dir / "affnist_eval.json", results)


if __name__ == "__main__":
    main()
