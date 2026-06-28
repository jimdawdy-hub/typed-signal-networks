from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch import nn

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
from ai_unity.data import get_mnist_pair_loaders, get_vision_loaders
from ai_unity.ternary_moe import FloatMoE, FloatSingle, TernaryMoE, TernarySingle
from ai_unity.utils import count_params, ensure_dir, output_tuple, resolve_device, seed_everything, write_json

COMPLEX_MODEL_CHOICES = (
    "all",
    "baseline",
    "cnn",
    "resnet",
    "vit",
    "real",
    "real-large",
    "complex-b",
    "spatial-b",
    "complex-a",
)

COMPLEX_AFFINE_AUGMENT_CHOICES = (
    "none",
    "random-affine-mild",
    "random-affine-strong",
)

COMPLEX_AFFINE_AUGMENTS = {
    "random-affine-mild": {
        "degrees": (-30.0, 30.0),
        "translate": (0.15, 0.15),
        "scale": (0.85, 1.15),
        "shear_x": (-15.0, 15.0),
        "shear_y": (0.0, 0.0),
    },
    "random-affine-strong": {
        "degrees": (-45.0, 45.0),
        "translate": (0.25, 0.25),
        "scale": (0.75, 1.25),
        "shear_x": (-25.0, 25.0),
        "shear_y": (-10.0, 10.0),
    },
}


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision on CUDA.")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile when available.")
    parser.add_argument("--resume", action="store_true", help="Resume each model from its latest checkpoint when present.")
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-test-batches", type=int, default=None)
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--test-subset", type=int, default=None)
    parser.add_argument("--rotate-degrees", type=float, default=None)
    return parser


def iter_limited(loader, limit_batches: int | None):
    for idx, batch in enumerate(loader):
        if limit_batches is not None and idx >= limit_batches:
            break
        yield batch


def train_one_epoch(
    model: nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lb_weight: float = 0.01,
    amp: bool = False,
    limit_batches: int | None = None,
) -> dict[str, float]:
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    total_loss = 0.0
    total_main_loss = 0.0
    total_aux_loss = 0.0
    correct = 0
    total = 0
    batches = 0

    for x, y in iter_limited(train_loader, limit_batches):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            logits, aux_loss, _ = output_tuple(model(x), device)
            main_loss = F.cross_entropy(logits, y)
            loss = main_loss + lb_weight * aux_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = y.shape[0]
        total_loss += loss.item() * batch_size
        total_main_loss += main_loss.item() * batch_size
        total_aux_loss += aux_loss.item() * batch_size
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += batch_size
        batches += 1

    return {
        "loss": total_loss / max(total, 1),
        "main_loss": total_main_loss / max(total, 1),
        "aux_loss": total_aux_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "batches": float(batches),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_loader,
    device: torch.device,
    limit_batches: int | None = None,
) -> dict:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    phases = []

    for x, y in iter_limited(test_loader, limit_batches):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits, _, info = output_tuple(model(x), device)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=1)

        batch_size = y.shape[0]
        total_loss += loss.item() * batch_size
        correct += (preds == y).sum().item()
        total += batch_size
        y_true.extend(y.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())
        if "digit_phase" in info:
            phases.append(info["digit_phase"].detach().cpu())

    result = {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(10))).tolist() if y_true else [],
    }
    if phases:
        phase_tensor = torch.cat(phases, dim=0)
        result.update(
            {
                "phase_mean": phase_tensor.mean().item(),
                "phase_std": phase_tensor.std(unbiased=False).item(),
                "phase_collapsed": bool(phase_tensor.std(unbiased=False).item() < 1e-3),
                "phase_per_class_std": phase_tensor.std(dim=(0, 2), unbiased=False).tolist(),
            }
        )
    return result


@torch.no_grad()
def analyze_expert_usage(model: nn.Module, loader, device: torch.device, limit_batches: int | None = None) -> dict | None:
    if not hasattr(model, "moe"):
        return None
    model.eval()
    usage_by_class = torch.zeros(10, model.moe.num_experts, device=device)
    usage_total = torch.zeros(model.moe.num_experts, device=device)
    entropy_values = []
    for x, y in iter_limited(loader, limit_batches):
        x, y = x.to(device), y.to(device)
        router_probs = torch.softmax(model.moe.router(x), dim=-1)
        _, top_k_indices = torch.topk(router_probs, model.moe.top_k, dim=-1)
        entropy_values.append((-(router_probs * router_probs.clamp_min(1e-8).log()).sum(dim=-1)).detach())
        usage_total += torch.bincount(top_k_indices.reshape(-1), minlength=model.moe.num_experts).float()
        for digit in range(10):
            mask = y == digit
            if mask.any():
                usage_by_class[digit] += torch.bincount(
                    top_k_indices[mask].reshape(-1), minlength=model.moe.num_experts
                ).float()
    usage_fraction = usage_total / usage_total.sum().clamp_min(1)
    per_class = usage_by_class / usage_by_class.sum(dim=1, keepdim=True).clamp_min(1)
    avg_probs = usage_fraction
    ideal = torch.ones_like(avg_probs) / model.moe.num_experts
    load_balance_loss = F.mse_loss(avg_probs, ideal)
    entropy = torch.cat(entropy_values).mean() if entropy_values else torch.tensor(0.0)
    return {
        "usage_fraction": usage_fraction.cpu().tolist(),
        "usage_by_class": per_class.cpu().tolist(),
        "entropy": entropy.item(),
        "collapse_rate": usage_fraction.max().item(),
        "load_balance_loss": load_balance_loss.item(),
    }


@torch.no_grad()
def analyze_phase_patterns(model: nn.Module, loader, device: torch.device, limit_batches: int | None = 5) -> dict | None:
    if "Complex" not in getattr(model, "name", ""):
        return None
    model.eval()
    by_class: dict[int, list[torch.Tensor]] = {digit: [] for digit in range(10)}
    for x, y in iter_limited(loader, limit_batches):
        x, y = x.to(device), y.to(device)
        _, _, info = output_tuple(model(x), device)
        if "digit_phase" not in info:
            continue
        phases = info["digit_phase"]
        for row in range(x.shape[0]):
            digit = int(y[row].item())
            by_class[digit].append(phases[row, digit].detach().cpu())

    result = {}
    for digit, values in by_class.items():
        if values:
            p = torch.stack(values)
            result[digit] = {
                "mean_phase_per_dim": p.mean(dim=0).tolist(),
                "std_phase_per_dim": p.std(dim=0, unbiased=False).tolist(),
                "overall_std": p.std(unbiased=False).item(),
                "collapsed": bool(p.std(unbiased=False).item() < 1e-3),
            }
    return result


def save_history_csv(path: Path, history: list[dict]) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in history for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def save_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    args,
    name: str,
    epoch: int,
    history: list[dict],
    best_test_acc: float,
    best_epoch: int | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "name": name,
            "epoch": epoch,
            "history": history,
            "best_test_acc": best_test_acc,
            "best_epoch": best_epoch,
        },
        path,
    )


def save_accuracy_plot(path: Path, results: dict) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    for name, record in results.items():
        epochs = [row["epoch"] for row in record["history"]]
        acc = [row["test_acc"] for row in record["history"]]
        plt.plot(epochs, acc, marker="o", label=name.split("(")[0].strip())
    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def maybe_compile(model: nn.Module, enabled: bool) -> nn.Module:
    if enabled and hasattr(torch, "compile"):
        return torch.compile(model)
    return model


def train_models(
    models: Iterable[tuple[nn.Module, str, object]],
    args,
    experiment_name: str,
) -> dict:
    seed_everything(args.seed)
    device = resolve_device(args.device, args.gpu_index)
    output_dir = ensure_dir(args.output_dir)
    checkpoint_dir = ensure_dir(args.checkpoint_dir)

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Seed: {args.seed}")
    print("=" * 70)

    results = {}
    for model, data_key, loaders in models:
        model = maybe_compile(model.to(device), args.compile)
        display_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        train_loader = loaders[data_key]["train"] if isinstance(loaders, dict) else loaders[0]
        test_loader = loaders[data_key]["test"] if isinstance(loaders, dict) else loaders[1]

        name = getattr(display_model, "name", display_model.__class__.__name__)
        safe_name = name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").replace("=", "-")
        history_path = output_dir / f"{experiment_name}_{safe_name}_history.csv"
        confusion_path = output_dir / f"{experiment_name}_{safe_name}_confusion.json"
        ckpt_path = checkpoint_dir / f"{experiment_name}_{safe_name}_seed{args.seed}.pt"
        latest_ckpt_path = checkpoint_dir / f"{experiment_name}_{safe_name}_seed{args.seed}_latest.pt"
        best_ckpt_path = checkpoint_dir / f"{experiment_name}_{safe_name}_seed{args.seed}_best.pt"
        print(f"\n--- {name} ---")
        has_complex_params = any(torch.is_complex(param) for param in display_model.parameters())
        model_amp = args.amp and not has_complex_params
        if args.amp and has_complex_params:
            print("  AMP disabled for complex-valued parameters.")
        history = []
        start_epoch = 1
        best_test_acc = float("-inf")
        best_epoch = None
        if getattr(args, "resume", False) and latest_ckpt_path.exists():
            payload = torch.load(latest_ckpt_path, map_location=device)
            display_model.load_state_dict(payload["model_state_dict"])
            optimizer.load_state_dict(payload["optimizer_state_dict"])
            history = list(payload.get("history", []))
            best_test_acc = float(payload.get("best_test_acc", float("-inf")))
            best_epoch = payload.get("best_epoch")
            start_epoch = int(payload.get("epoch", 0)) + 1
            print(f"  Resumed from epoch {start_epoch - 1}.")
        start = time.time()
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_start = time.time()
            train_m = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                lb_weight=getattr(args, "lb_weight", 0.01),
                amp=model_amp,
                limit_batches=args.limit_train_batches,
            )
            test_m = evaluate(model, test_loader, device, limit_batches=args.limit_test_batches)
            elapsed = time.time() - epoch_start
            row = {
                "epoch": epoch,
                "train_acc": train_m["accuracy"],
                "test_acc": test_m["accuracy"],
                "train_loss": train_m["main_loss"],
                "test_loss": test_m["loss"],
                "aux_loss": train_m["aux_loss"],
                "time": elapsed,
            }
            history.append(row)
            save_history_csv(history_path, history)
            if test_m["accuracy"] > best_test_acc:
                best_test_acc = test_m["accuracy"]
                best_epoch = epoch
                save_training_checkpoint(
                    best_ckpt_path,
                    display_model,
                    optimizer,
                    args,
                    name,
                    epoch,
                    history,
                    best_test_acc,
                    best_epoch,
                )
            save_training_checkpoint(
                latest_ckpt_path,
                display_model,
                optimizer,
                args,
                name,
                epoch,
                history,
                best_test_acc,
                best_epoch,
            )
            phase_str = f" phase_std={test_m['phase_std']:.3f}" if "phase_std" in test_m else ""
            sparse_str = f" sparsity={display_model.sparsity():.1%}" if hasattr(display_model, "sparsity") else ""
            print(
                f"  Epoch {epoch}/{args.epochs} | train={train_m['accuracy']:.1%} "
                f"| test={test_m['accuracy']:.1%} | loss={train_m['main_loss']:.4f} "
                f"| {elapsed:.1f}s{phase_str}{sparse_str}"
            )

        final_eval = evaluate(model, test_loader, device, limit_batches=args.limit_test_batches)
        routing = analyze_expert_usage(display_model, test_loader, device, limit_batches=args.limit_test_batches)
        phase = analyze_phase_patterns(display_model, test_loader, device)
        total_time = time.time() - start
        torch.save({"model_state_dict": display_model.state_dict(), "args": vars(args), "name": name}, ckpt_path)
        save_history_csv(history_path, history)
        write_json(confusion_path, final_eval["confusion_matrix"])

        results[name] = {
            "params": count_params(display_model),
            "history": history,
            "final_train_acc": history[-1]["train_acc"] if history else None,
            "final_test_acc": final_eval["accuracy"],
            "final_test_loss": final_eval["loss"],
            "total_time": total_time,
            "checkpoint": str(ckpt_path),
            "latest_checkpoint": str(latest_ckpt_path),
            "best_checkpoint": str(best_ckpt_path) if best_ckpt_path.exists() else None,
            "best_test_acc": best_test_acc if best_test_acc != float("-inf") else None,
            "best_epoch": best_epoch,
            "routing": routing,
            "phase_analysis": phase,
            "confusion_matrix": final_eval["confusion_matrix"],
        }

    write_json(output_dir / f"{experiment_name}_comparison.json", results)
    save_accuracy_plot(output_dir / f"{experiment_name}_accuracy.png", results)
    print(f"\nResults saved under {output_dir}")
    return results


def run_ternary(args) -> dict:
    train_loader, test_loader = get_vision_loaders(
        "mnist",
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        flatten=True,
        num_workers=args.num_workers,
        seed=args.seed,
        rotate_degrees=args.rotate_degrees,
        train_subset=args.train_subset,
        test_subset=args.test_subset,
    )
    loaders = (train_loader, test_loader)
    models = [
        (FloatSingle(), "flat", loaders),
        (FloatMoE(num_experts=args.num_experts, top_k=args.top_k, hidden_dim=args.hidden_dim), "flat", loaders),
        (TernarySingle(), "flat", loaders),
        (TernaryMoE(num_experts=args.num_experts, top_k=args.top_k, hidden_dim=args.hidden_dim), "flat", loaders),
    ]
    return train_models(models, args, "ternary_moe")


def run_complex(args) -> dict:
    random_affine = COMPLEX_AFFINE_AUGMENTS.get(getattr(args, "affine_augment", "none"))
    loaders = get_mnist_pair_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        rotate_degrees=args.rotate_degrees,
        random_affine=random_affine,
        rotation_seed=args.seed,
        train_subset=args.train_subset,
        test_subset=args.test_subset,
    )
    models_by_key = {
        "baseline": (BaselineMLP(), "flat", loaders),
        "cnn": (BaselineCNN(), "img", loaders),
        "resnet": (ResidualCNNBaseline(), "img", loaders),
        "vit": (VisionTransformerBaseline(), "img", loaders),
        "real": (RealCapsuleNet(), "img", loaders),
        "real-large": (RealCapsuleNetLarge(), "img", loaders),
        "complex-b": (ComplexCapsuleNetB(), "img", loaders),
        "spatial-b": (ComplexCapsuleNetSpatialB(), "img", loaders),
        "complex-a": (ComplexCapsuleNetA(), "img", loaders),
    }
    selected = getattr(args, "models", ["all"])
    if "all" in selected:
        model_keys = [key for key in models_by_key]
    else:
        model_keys = selected
    models = [models_by_key[key] for key in model_keys]
    return train_models(models, args, "complex_capsules")


def ternary_parser() -> argparse.ArgumentParser:
    parser = add_common_args(argparse.ArgumentParser(description="Ternary + MoE comparison on MNIST"))
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lb-weight", type=float, default=0.01)
    return parser


def complex_parser() -> argparse.ArgumentParser:
    parser = add_common_args(argparse.ArgumentParser(description="Complex capsule comparison on MNIST"))
    parser.add_argument(
        "--models",
        nargs="+",
        choices=COMPLEX_MODEL_CHOICES,
        default=["all"],
        help="Complex experiment model subset. Use 'all' for the full comparison.",
    )
    parser.add_argument(
        "--affine-augment",
        choices=COMPLEX_AFFINE_AUGMENT_CHOICES,
        default="none",
        help="Apply a random affine preset to MNIST train/test loaders during training.",
    )
    return parser
