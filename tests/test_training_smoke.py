from __future__ import annotations

import argparse
import csv

import numpy as np
import torch
from PIL import Image
from scipy.io import savemat
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from ai_unity.data import AffNISTDataset, IndexedRandomAffineDataset, IndexedRandomRotationDataset, get_vision_loaders
from ai_unity.evaluate_affnist import parser as affnist_eval_parser
from ai_unity.evaluate_complex_affines import parser as affine_eval_parser, phase_factor_diagnostics
from ai_unity.evaluate_affine_probes import parser as affine_probe_parser, probe_feature_block
from ai_unity.evaluate_complex_rotations import parser as rotation_eval_parser
from ai_unity.ternary_moe import FloatMoE
from ai_unity.training import complex_parser, train_models


def test_one_batch_training_writes_results(tmp_path):
    x = torch.randn(8, 784)
    y = torch.randint(0, 10, (8,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    args = argparse.Namespace(
        seed=5,
        device="cpu",
        gpu_index=0,
        output_dir=str(tmp_path / "results"),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        lr=1e-3,
        epochs=1,
        amp=False,
        compile=False,
        limit_train_batches=1,
        limit_test_batches=1,
        lb_weight=0.01,
    )

    results = train_models([(FloatMoE(num_experts=2, top_k=1), "flat", (loader, loader))], args, "smoke")
    assert "FloatMoE (2 experts, top-1)" in results
    assert (tmp_path / "results" / "smoke_comparison.json").exists()
    assert any((tmp_path / "checkpoints").glob("*.pt"))


def test_complex_parser_accepts_model_subset():
    args = complex_parser().parse_args(
        [
            "--models",
            "cnn",
            "resnet",
            "vit",
            "real-large",
            "complex-b",
            "--affine-augment",
            "random-affine-mild",
            "--resume",
        ]
    )

    assert args.models == ["cnn", "resnet", "vit", "real-large", "complex-b"]
    assert args.affine_augment == "random-affine-mild"
    assert args.resume is True


def test_rotation_eval_parser_accepts_random_rotation_ranges():
    args = rotation_eval_parser().parse_args(
        [
            "--comparison-json",
            "comparison.json",
            "--output-dir",
            "out",
            "--models",
            "cnn",
            "resnet",
            "vit",
            "real-large",
            "complex-b",
            "--checkpoint-key",
            "best_checkpoint",
            "--random-rotations",
            "-30",
            "30",
            "--random-rotations",
            "-90",
            "90",
            "--random-samples",
            "2",
        ]
    )

    assert args.rotations == [0, 15, 30, 45, 60, 75, 90]
    assert args.models == ["cnn", "resnet", "vit", "real-large", "complex-b"]
    assert args.checkpoint_key == "best_checkpoint"
    assert args.random_rotations == [[-30.0, 30.0], [-90.0, 90.0]]
    assert args.random_samples == 2


def test_indexed_random_rotation_dataset_is_seeded_by_index():
    image = Image.new("L", (8, 8), color=255)
    base = [(image, 0), (image, 1), (image, 2)]

    first = IndexedRandomRotationDataset(base, (-30, 30), seed=123)
    second = IndexedRandomRotationDataset(base, (-30, 30), seed=123)
    different = IndexedRandomRotationDataset(base, (-30, 30), seed=124)

    assert first.angles == second.angles
    assert first.angles != different.angles


def test_indexed_random_affine_dataset_is_seeded_by_index():
    image = Image.new("L", (8, 8), color=255)
    base = [(image, 0), (image, 1), (image, 2)]

    first = IndexedRandomAffineDataset(base, degrees=(-30, 30), translate=(0.2, 0.2), seed=123)
    second = IndexedRandomAffineDataset(base, degrees=(-30, 30), translate=(0.2, 0.2), seed=123)
    different = IndexedRandomAffineDataset(base, degrees=(-30, 30), translate=(0.2, 0.2), seed=124)

    assert first.angles == second.angles
    assert first.translate_x == second.translate_x
    assert first.angles != different.angles
    assert set(first.affine_params(0)) == {
        "angle_degrees",
        "translate_x_frac",
        "translate_y_frac",
        "scale",
        "shear_x_degrees",
        "shear_y_degrees",
    }


def test_affine_eval_parser_accepts_scenario_subsets():
    args = affine_eval_parser().parse_args(
        [
            "--comparison-json",
            "comparison.json",
            "--output-dir",
            "out",
            "--models",
            "cnn",
            "resnet",
            "vit",
            "real-large",
            "complex-b",
            "--checkpoint-key",
            "best_checkpoint",
            "--fixed-scenarios",
            "clean",
            "affine_moderate",
            "--random-scenarios",
            "heldout_affine_left_zoom",
            "--random-samples",
            "2",
            "--phase-diagnostics",
        ]
    )

    assert args.models == ["cnn", "resnet", "vit", "real-large", "complex-b"]
    assert args.checkpoint_key == "best_checkpoint"
    assert args.fixed_scenarios == ["clean", "affine_moderate"]
    assert args.random_scenarios == ["heldout_affine_left_zoom"]
    assert args.random_samples == 2
    assert args.phase_diagnostics is True


def test_affnist_eval_parser_accepts_multiple_comparison_jsons():
    args = affnist_eval_parser().parse_args(
        [
            "--comparison-json",
            "capsules.json",
            "vit.json",
            "--output-dir",
            "out",
            "--models",
            "cnn",
            "resnet",
            "vit",
            "complex-b",
            "--test-subset",
            "32",
        ]
    )

    assert [str(path) for path in args.comparison_json] == ["capsules.json", "vit.json"]
    assert args.models == ["cnn", "resnet", "vit", "complex-b"]
    assert args.test_subset == 32


def test_affnist_dataset_loads_local_mat(tmp_path):
    affnist_dir = tmp_path / "affnist"
    affnist_dir.mkdir()
    image = np.zeros((1600, 2), dtype=np.uint8)
    image[0, 0] = 255
    image[-1, 1] = 255
    savemat(affnist_dir / "test.mat", {"image": image, "label_int": np.array([3, 7])})

    dataset = AffNISTDataset(tmp_path, split="test", download=False, transform=transforms.ToTensor())
    x, y = dataset[0]

    assert len(dataset) == 2
    assert x.shape == (1, 40, 40)
    assert y == 3


def test_phase_factor_diagnostics_smoke():
    class FakeComplex(torch.nn.Module):
        name = "FakeComplex"

        def forward(self, x):
            batch_size = x.shape[0]
            phases = torch.zeros(batch_size, 10, 2, device=x.device)
            phases[:, :, 0] = torch.linspace(0.0, 1.0, batch_size, device=x.device).view(-1, 1)
            return torch.zeros(batch_size, 10, device=x.device), {"digit_phase": phases}

    image = Image.new("L", (8, 8), color=255)
    base = [(image, idx % 10) for idx in range(6)]
    dataset = IndexedRandomAffineDataset(
        base,
        degrees=(-45, 45),
        translate=(0.2, 0.2),
        seed=123,
        transform=transforms.ToTensor(),
    )
    loader = DataLoader(dataset, batch_size=3)

    result = phase_factor_diagnostics(
        {"FakeComplex": (FakeComplex(), "img")},
        loader,
        torch.device("cpu"),
        limit_batches=None,
    )

    assert result["FakeComplex"]["examples"] == 6
    assert "angle_degrees" in result["FakeComplex"]["factor_correlations"]


def test_affine_probe_parser_accepts_probe_options():
    args = affine_probe_parser().parse_args(
        [
            "--comparison-json",
            "comparison.json",
            "--output-dir",
            "out",
            "--models",
            "cnn",
            "resnet",
            "vit",
            "complex-b",
            "--random-scenarios",
            "heldout_affine_left_zoom",
            "--probe-train-examples",
            "12",
            "--probe-test-examples",
            "8",
            "--alpha",
            "3.5",
        ]
    )

    assert args.models == ["cnn", "resnet", "vit", "complex-b"]
    assert args.random_scenarios == ["heldout_affine_left_zoom"]
    assert args.probe_train_examples == 12
    assert args.probe_test_examples == 8
    assert args.alpha == 3.5


def test_probe_feature_block_reports_factor_metrics():
    x = np.arange(40, dtype=np.float32).reshape(20, 2)
    factors = {
        "angle_degrees": x[:, 0] * 2,
        "translate_x_frac": x[:, 1],
    }
    train_idx = np.arange(12)
    test_idx = np.arange(12, 20)

    result = probe_feature_block(x, factors, train_idx, test_idx, alpha=1.0)

    assert set(result) == {"angle_degrees", "translate_x_frac"}
    assert "r2" in result["angle_degrees"]
    assert "mae" in result["angle_degrees"]


def test_vision_loaders_reject_multiple_geometric_modes(tmp_path):
    try:
        get_vision_loaders(
            "mnist",
            data_dir=tmp_path,
            rotate_degrees=15,
            affine={"degrees": 0.0, "translate": (0, 0), "scale": 1.0, "shear": (0.0, 0.0)},
            train_subset=1,
            test_subset=1,
        )
    except ValueError as exc:
        assert "one geometric transform" in str(exc)
    else:
        raise AssertionError("Expected multiple geometric modes to be rejected.")


def test_train_models_flushes_best_latest_and_resumes(tmp_path):
    x = torch.randn(8, 784)
    y = torch.randint(0, 10, (8,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    args = argparse.Namespace(
        seed=7,
        device="cpu",
        gpu_index=0,
        output_dir=str(tmp_path / "results"),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        lr=1e-3,
        epochs=1,
        amp=False,
        compile=False,
        resume=False,
        limit_train_batches=1,
        limit_test_batches=1,
        lb_weight=0.01,
    )

    train_models([(FloatMoE(num_experts=2, top_k=1), "flat", (loader, loader))], args, "resume_smoke")

    history_path = tmp_path / "results" / "resume_smoke_FloatMoE_2_experts,_top-1_history.csv"
    latest_path = tmp_path / "checkpoints" / "resume_smoke_FloatMoE_2_experts,_top-1_seed7_latest.pt"
    best_path = tmp_path / "checkpoints" / "resume_smoke_FloatMoE_2_experts,_top-1_seed7_best.pt"
    assert history_path.exists()
    assert latest_path.exists()
    assert best_path.exists()

    args.epochs = 2
    args.resume = True
    results = train_models([(FloatMoE(num_experts=2, top_k=1), "flat", (loader, loader))], args, "resume_smoke")

    with open(history_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [row["epoch"] for row in rows] == ["1", "2"]
    record = results["FloatMoE (2 experts, top-1)"]
    assert record["latest_checkpoint"].endswith("_latest.pt")
    assert record["best_checkpoint"].endswith("_best.pt")
    assert record["best_epoch"] in {1, 2}
