from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

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
from ai_unity.utils import count_params
from ai_unity.ternary_moe import FloatMoE, FloatSingle, MoELayer, TernaryLinear, TernaryMoE, TernarySingle
from ai_unity.utils import output_tuple, seed_everything


def available_devices():
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda:0"))
    return devices


TERNARY_MODELS = [FloatSingle, FloatMoE, TernarySingle, TernaryMoE]
CAPSULE_MODELS = [
    BaselineMLP,
    BaselineCNN,
    ResidualCNNBaseline,
    VisionTransformerBaseline,
    RealCapsuleNet,
    RealCapsuleNetLarge,
    ComplexCapsuleNetA,
    ComplexCapsuleNetB,
    ComplexCapsuleNetSpatialB,
]

IMAGE_MODELS = [
    BaselineCNN,
    ResidualCNNBaseline,
    VisionTransformerBaseline,
    RealCapsuleNet,
    RealCapsuleNetLarge,
    ComplexCapsuleNetA,
    ComplexCapsuleNetB,
    ComplexCapsuleNetSpatialB,
]


@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("model_cls", TERNARY_MODELS)
def test_ternary_family_forward_backward_no_nans(model_cls, device):
    seed_everything(7)
    model = model_cls().to(device)
    x = torch.randn(4, 784, device=device)
    y = torch.randint(0, 10, (4,), device=device)

    logits, aux_loss, _ = output_tuple(model(x), device)
    assert logits.shape == (4, 10)
    assert aux_loss.device == device
    assert torch.isfinite(logits).all()

    loss = F.cross_entropy(logits, y) + aux_loss
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("model_cls", CAPSULE_MODELS)
def test_capsule_family_forward_backward_no_nans(model_cls, device):
    seed_everything(11)
    model = model_cls().to(device)
    x = torch.randn(2, 1, 28, 28, device=device)
    y = torch.randint(0, 10, (2,), device=device)

    logits, aux_loss, info = output_tuple(model(x), device)
    assert logits.shape == (2, 10)
    assert aux_loss.device == device
    assert torch.isfinite(logits).all()
    if "digit_phase" in info:
        assert torch.isfinite(info["digit_phase"]).all()
        assert info["digit_phase"].std().item() > 0.0

    loss = F.cross_entropy(logits, y)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("model_cls", IMAGE_MODELS)
def test_image_models_accept_native_affnist_shape(model_cls):
    seed_everything(17)
    model = model_cls()
    x = torch.randn(2, 1, 40, 40)

    logits, aux_loss, _ = output_tuple(model(x), torch.device("cpu"))

    assert logits.shape == (2, 10)
    assert aux_loss.device == torch.device("cpu")
    assert torch.isfinite(logits).all()


def test_ternary_layer_uses_only_minus_one_zero_plus_one_in_full_ternary_mode():
    layer = TernaryLinear(8, 4, warmup_steps=1)
    layer.eval()
    weights = layer.forward_weight()
    unique = set(weights.detach().cpu().unique().tolist())
    assert unique.issubset({-1.0, 0.0, 1.0})


@pytest.mark.parametrize("device", available_devices())
def test_moe_routing_diagnostics_are_finite(device):
    seed_everything(13)
    moe = MoELayer(8, 6, 3, num_experts=4, top_k=2).to(device)
    x = torch.randn(5, 8, device=device)
    y = torch.tensor([0, 1, 2, 3, 4], device=device)
    output, lb_loss, router_probs = moe(x)
    diagnostics = moe.routing_diagnostics(x, y)

    assert output.shape == (5, 3)
    assert router_probs.shape == (5, 4)
    assert torch.isfinite(lb_loss)
    assert abs(sum(diagnostics["usage_fraction"].tolist()) - 1.0) < 1e-5
    assert diagnostics["entropy"] >= 0
    assert 0 <= diagnostics["collapse_rate"] <= 1


@pytest.mark.parametrize("device", available_devices())
def test_complex_cuda_construction_path(device):
    if device.type != "cuda":
        pytest.skip("CUDA-specific complex construction check")
    z = torch.exp(torch.randn(8, device=device, dtype=torch.cfloat))
    assert z.dtype == torch.cfloat
    assert torch.isfinite(z).all()


def test_baseline_cnn_is_parameter_comparable_to_large_capsule():
    cnn_params = count_params(BaselineCNN())
    real_large_params = count_params(RealCapsuleNetLarge())

    assert abs(cnn_params - real_large_params) / real_large_params < 0.01


def test_vit_is_parameter_comparable_to_large_capsule():
    vit_params = count_params(VisionTransformerBaseline())
    real_large_params = count_params(RealCapsuleNetLarge())

    assert abs(vit_params - real_large_params) / real_large_params < 0.01
