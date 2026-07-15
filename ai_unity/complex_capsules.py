from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

NUM_CAP_TYPES = 8
PRIMARY_DIM = 8
NUM_CLASSES = 10
CLASS_DIM = 8
ROUTING_ITERS = 3
CONV_KERNEL = 9
CONV_STRIDE = 5


def _num_primary() -> int:
    spatial = ((20 - CONV_KERNEL) // CONV_STRIDE) + 1
    return NUM_CAP_TYPES * spatial * spatial


NUM_PRIMARY = _num_primary()
CAPSULE_SPATIAL = ((20 - CONV_KERNEL) // CONV_STRIDE) + 1


def real_squash(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    norm_sq = (x * x).sum(dim=dim, keepdim=True)
    norm = torch.sqrt(norm_sq + 1e-8)
    scale = norm_sq / (1.0 + norm_sq) / (norm + 1e-8)
    return scale * x


def complex_squash(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    mag_sq = (x * x.conj()).real.sum(dim=dim, keepdim=True)
    mag = torch.sqrt(mag_sq + 1e-8)
    scale = mag_sq / (1.0 + mag_sq) / (mag + 1e-8)
    return scale * x


class BaselineMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "BaselineMLP"
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return self.net(x.view(x.shape[0], -1)), {}


class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "BaselineCNN"
        hidden_dim = 382
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((7, 7)),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        h = self.features(x)
        return self.classifier(h.view(h.shape[0], -1)), {}


class VisionTransformerBaseline(nn.Module):
    """Small ViT-style baseline parameter-matched to the large capsule models."""

    def __init__(self):
        super().__init__()
        self.name = "VisionTransformerBaseline"
        embed_dim = 192
        num_patches = 49
        self.patch_embed = nn.Conv2d(1, embed_dim, kernel_size=4, stride=4)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=384,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=9, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 399),
            nn.GELU(),
            nn.Linear(399, 10),
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _pos_embed_for(self, grid_h: int, grid_w: int) -> torch.Tensor:
        if grid_h * grid_w + 1 == self.pos_embed.shape[1]:
            return self.pos_embed
        cls_pos = self.pos_embed[:, :1]
        patch_pos = self.pos_embed[:, 1:]
        source = int(patch_pos.shape[1] ** 0.5)
        patch_pos = patch_pos.transpose(1, 2).reshape(1, -1, source, source)
        patch_pos = F.interpolate(patch_pos, size=(grid_h, grid_w), mode="bicubic", align_corners=False)
        patch_pos = patch_pos.flatten(2).transpose(1, 2)
        return torch.cat([cls_pos, patch_pos], dim=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.patch_embed(x)
        grid_h, grid_w = h.shape[-2:]
        h = h.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        h = torch.cat([cls, h], dim=1) + self._pos_embed_for(grid_h, grid_w)
        h = self.encoder(h)
        return self.norm(h[:, 0])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        cls = self.encode(x)
        return self.head(cls), {"transformer_cls": cls}


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.net(x) + self.skip(x))


class ResidualCNNBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "ResidualCNNBaseline"
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
            ResidualBlock(128, 192, stride=2),
            ResidualBlock(192, 192),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(192, 10)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(self.stem(x))
        return self.pool(h).flatten(1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        h = self.encode(x)
        return self.head(h), {"residual_features": h}


class RealCapsuleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "RealCapsule"
        self.conv1 = nn.Conv2d(1, 256, kernel_size=9, stride=1)
        self.primary_conv = nn.Conv2d(256, NUM_CAP_TYPES * PRIMARY_DIM, kernel_size=CONV_KERNEL, stride=CONV_STRIDE)
        self.W = nn.Parameter(torch.randn(NUM_CLASSES, NUM_PRIMARY, CLASS_DIM, PRIMARY_DIM) * 0.01)
        self.bias = nn.Parameter(torch.zeros(NUM_CLASSES, 1, CLASS_DIM))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        B = x.shape[0]
        h = F.relu(self.conv1(x))
        h = self.primary_conv(h)
        h = F.adaptive_avg_pool2d(h, (CAPSULE_SPATIAL, CAPSULE_SPATIAL))
        h = h.view(B, NUM_CAP_TYPES, PRIMARY_DIM, CAPSULE_SPATIAL, CAPSULE_SPATIAL)
        h = h.permute(0, 1, 3, 4, 2).reshape(B, NUM_PRIMARY, PRIMARY_DIM)
        u = real_squash(h, dim=-1)
        u_hat = torch.einsum("cnij,bnj->bcni", self.W, u)

        b = torch.zeros(B, NUM_CLASSES, NUM_PRIMARY, device=x.device)
        for _ in range(ROUTING_ITERS):
            c = F.softmax(b, dim=1)
            s = (c.unsqueeze(-1) * u_hat).sum(dim=2) + self.bias.squeeze(1)
            v = real_squash(s, dim=-1)
            agreement = (u_hat * v.unsqueeze(2)).sum(dim=-1)
            b = b + agreement

        probs = torch.sqrt((v**2).sum(dim=-1) + 1e-8)
        return probs, {"digit_caps": v}


class RealCapsuleNetLarge(nn.Module):
    """Parameter-matched real capsule control with two real transformation banks."""

    def __init__(self):
        super().__init__()
        self.name = "RealCapsuleLarge"
        self.conv1 = nn.Conv2d(1, 256, kernel_size=9, stride=1)
        self.primary_conv = nn.Conv2d(256, NUM_CAP_TYPES * PRIMARY_DIM, kernel_size=CONV_KERNEL, stride=CONV_STRIDE)
        self.primary_conv_extra = nn.Conv2d(
            256, NUM_CAP_TYPES * PRIMARY_DIM, kernel_size=CONV_KERNEL, stride=CONV_STRIDE
        )
        self.W_a = nn.Parameter(torch.randn(NUM_CLASSES, NUM_PRIMARY, CLASS_DIM, PRIMARY_DIM) * 0.01)
        self.W_b = nn.Parameter(torch.randn(NUM_CLASSES, NUM_PRIMARY, CLASS_DIM, PRIMARY_DIM) * 0.01)
        self.bias = nn.Parameter(torch.zeros(NUM_CLASSES, 1, CLASS_DIM))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        B = x.shape[0]
        h = F.relu(self.conv1(x))
        h = self.primary_conv(h) + self.primary_conv_extra(h)
        h = F.adaptive_avg_pool2d(h, (CAPSULE_SPATIAL, CAPSULE_SPATIAL))
        h = h.view(B, NUM_CAP_TYPES, PRIMARY_DIM, CAPSULE_SPATIAL, CAPSULE_SPATIAL)
        h = h.permute(0, 1, 3, 4, 2).reshape(B, NUM_PRIMARY, PRIMARY_DIM)
        u = real_squash(h, dim=-1)
        u_hat = torch.einsum("cnij,bnj->bcni", self.W_a, u) + torch.einsum("cnij,bnj->bcni", self.W_b, u)

        b = torch.zeros(B, NUM_CLASSES, NUM_PRIMARY, device=x.device)
        for _ in range(ROUTING_ITERS):
            c = F.softmax(b, dim=1)
            s = (c.unsqueeze(-1) * u_hat).sum(dim=2) + self.bias.squeeze(1)
            v = real_squash(s, dim=-1)
            agreement = (u_hat * v.unsqueeze(2)).sum(dim=-1)
            b = b + agreement

        probs = self._readout(v)
        return probs, {"digit_caps": v}

    def _readout(self, v: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((v**2).sum(dim=-1) + 1e-8)


class RealCapsuleNetLargeL1(RealCapsuleNetLarge):
    """Readout-only ablation of RealCapsuleNetLarge.

    Same (capacity-degenerate) two-bank architecture, but the class readout
    matches ComplexCapsuleNetB: sum of per-dimension magnitudes, bounded by
    sqrt(CLASS_DIM), instead of the L2 norm bounded by 1. Isolates the
    readout-dynamic-range confound from the capacity confound.
    """

    def __init__(self):
        super().__init__()
        self.name = "RealCapsuleLargeL1"

    def _readout(self, v: torch.Tensor) -> torch.Tensor:
        return v.abs().sum(dim=-1)


PRIMARY_DIM_V2 = 16


class RealCapsuleNetControlV2(nn.Module):
    """Non-degenerate parameter-matched real capsule control (both fixes).

    Replaces RealCapsuleNetLarge's two summed duplicate banks (whose sum
    collapses to a single linear map, adding no capacity) with one genuinely
    wider primary bank: 16-dimensional primary capsules. This mirrors how
    ComplexCapsuleNetB spends its budget — a complex 8-dim primary capsule is
    16 real dimensions — so the control matches the complex model in both
    parameter count and representational width.

    The class readout also matches ComplexCapsuleNetB exactly in functional
    form and bound: sum of per-dimension magnitudes of the squashed class
    capsule, bounded by sqrt(CLASS_DIM) = sqrt(8), removing the dynamic-range
    handicap of the L2-norm readout under cross-entropy.

    Exactly 2,767,568 trainable parameters, equal to ComplexCapsuleNetB and
    RealCapsuleNetLarge.
    """

    def __init__(self):
        super().__init__()
        self.name = "RealCapsuleControlV2"
        self.conv1 = nn.Conv2d(1, 256, kernel_size=9, stride=1)
        self.primary_conv = nn.Conv2d(
            256, NUM_CAP_TYPES * PRIMARY_DIM_V2, kernel_size=CONV_KERNEL, stride=CONV_STRIDE
        )
        self.W = nn.Parameter(torch.randn(NUM_CLASSES, NUM_PRIMARY, CLASS_DIM, PRIMARY_DIM_V2) * 0.01)
        self.bias = nn.Parameter(torch.zeros(NUM_CLASSES, 1, CLASS_DIM))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        B = x.shape[0]
        h = F.relu(self.conv1(x))
        h = self.primary_conv(h)
        h = F.adaptive_avg_pool2d(h, (CAPSULE_SPATIAL, CAPSULE_SPATIAL))
        h = h.view(B, NUM_CAP_TYPES, PRIMARY_DIM_V2, CAPSULE_SPATIAL, CAPSULE_SPATIAL)
        h = h.permute(0, 1, 3, 4, 2).reshape(B, NUM_PRIMARY, PRIMARY_DIM_V2)
        u = real_squash(h, dim=-1)
        u_hat = torch.einsum("cnij,bnj->bcni", self.W, u)

        b = torch.zeros(B, NUM_CLASSES, NUM_PRIMARY, device=x.device)
        for _ in range(ROUTING_ITERS):
            c = F.softmax(b, dim=1)
            s = (c.unsqueeze(-1) * u_hat).sum(dim=2) + self.bias.squeeze(1)
            v = real_squash(s, dim=-1)
            agreement = (u_hat * v.unsqueeze(2)).sum(dim=-1)
            b = b + agreement

        probs = self._readout(v)
        return probs, {"digit_caps": v}

    def _readout(self, v: torch.Tensor) -> torch.Tensor:
        return v.abs().sum(dim=-1)


class RealCapsuleNetControlV2Norm(RealCapsuleNetControlV2):
    """Capacity-only ablation: V2's non-degenerate wide architecture with the
    original L2-norm readout of RealCapsuleNetLarge. Isolates the capacity fix
    from the readout fix."""

    def __init__(self):
        super().__init__()
        self.name = "RealCapsuleControlV2Norm"

    def _readout(self, v: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((v**2).sum(dim=-1) + 1e-8)


class ComplexCapsuleNetB(nn.Module):
    """Complex capsule where phase is learned from spatial-angle features."""

    def __init__(self):
        super().__init__()
        self.name = "ComplexCapsuleB (phase=angle)"
        self.conv1 = nn.Conv2d(1, 256, kernel_size=9, stride=1)
        self.primary_conv_mag = nn.Conv2d(256, NUM_CAP_TYPES * PRIMARY_DIM, kernel_size=CONV_KERNEL, stride=CONV_STRIDE)
        self.primary_conv_phase = nn.Conv2d(256, NUM_CAP_TYPES * PRIMARY_DIM, kernel_size=CONV_KERNEL, stride=CONV_STRIDE)
        self.W_real = nn.Parameter(torch.randn(NUM_CLASSES, NUM_PRIMARY, CLASS_DIM, PRIMARY_DIM) * 0.01)
        self.W_imag = nn.Parameter(torch.randn(NUM_CLASSES, NUM_PRIMARY, CLASS_DIM, PRIMARY_DIM) * 0.01)
        self.bias = nn.Parameter(torch.zeros(NUM_CLASSES, 1, CLASS_DIM, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        B = x.shape[0]
        h = F.relu(self.conv1(x))
        mag = F.softplus(self.primary_conv_mag(h))
        phase = self.primary_conv_phase(h)
        mag = F.adaptive_avg_pool2d(mag, (CAPSULE_SPATIAL, CAPSULE_SPATIAL))
        phase = F.adaptive_avg_pool2d(phase, (CAPSULE_SPATIAL, CAPSULE_SPATIAL))

        def reshape_caps(t: torch.Tensor) -> torch.Tensor:
            t = t.view(B, NUM_CAP_TYPES, PRIMARY_DIM, CAPSULE_SPATIAL, CAPSULE_SPATIAL)
            return t.permute(0, 1, 3, 4, 2).reshape(B, NUM_PRIMARY, PRIMARY_DIM)

        mag = reshape_caps(mag)
        phase = reshape_caps(phase)
        u = torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))
        u = complex_squash(u, dim=-1)

        W = torch.complex(self.W_real, self.W_imag)
        u_hat = torch.einsum("cnij,bnj->bcni", W, u)
        b = torch.zeros(B, NUM_CLASSES, NUM_PRIMARY, device=x.device)
        for _ in range(ROUTING_ITERS):
            c = F.softmax(b, dim=1)
            s = (c.unsqueeze(-1).to(torch.cfloat) * u_hat).sum(dim=2) + self.bias.squeeze(1)
            v = complex_squash(s, dim=-1)
            agreement = (u_hat * v.unsqueeze(2).conj()).real.sum(dim=-1)
            b = b + agreement

        probs = torch.abs(v).sum(dim=-1)
        return probs, {"digit_caps": v, "primary_phase": phase, "digit_phase": v.angle()}


class ComplexCapsuleNetSpatialB(nn.Module):
    """Complex capsule where primary phase is anchored to capsule spatial angle."""

    def __init__(self):
        super().__init__()
        self.name = "ComplexCapsuleSpatialB (phase=spatial_angle)"
        self.conv1 = nn.Conv2d(1, 256, kernel_size=9, stride=1)
        self.primary_conv_mag = nn.Conv2d(256, NUM_CAP_TYPES * PRIMARY_DIM, kernel_size=CONV_KERNEL, stride=CONV_STRIDE)
        self.phase_bias = nn.Parameter(torch.zeros(NUM_CAP_TYPES, PRIMARY_DIM, 1, 1))
        self.W_real = nn.Parameter(torch.randn(NUM_CLASSES, NUM_PRIMARY, CLASS_DIM, PRIMARY_DIM) * 0.01)
        self.W_imag = nn.Parameter(torch.randn(NUM_CLASSES, NUM_PRIMARY, CLASS_DIM, PRIMARY_DIM) * 0.01)
        self.bias = nn.Parameter(torch.zeros(NUM_CLASSES, 1, CLASS_DIM, dtype=torch.cfloat))

    def spatial_phase(self, spatial: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        coords = torch.linspace(-1.0, 1.0, spatial, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        return torch.atan2(yy, xx).view(1, 1, spatial, spatial)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        B = x.shape[0]
        h = F.relu(self.conv1(x))
        mag = F.softplus(self.primary_conv_mag(h))
        mag = F.adaptive_avg_pool2d(mag, (CAPSULE_SPATIAL, CAPSULE_SPATIAL))
        mag = mag.view(B, NUM_CAP_TYPES, PRIMARY_DIM, CAPSULE_SPATIAL, CAPSULE_SPATIAL)

        phase = self.spatial_phase(CAPSULE_SPATIAL, x.device, mag.dtype).unsqueeze(2) + self.phase_bias.unsqueeze(0)
        phase = phase.expand(B, -1, -1, -1, -1)

        mag = mag.permute(0, 1, 3, 4, 2).reshape(B, NUM_PRIMARY, PRIMARY_DIM)
        phase = phase.permute(0, 1, 3, 4, 2).reshape(B, NUM_PRIMARY, PRIMARY_DIM)
        u = torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))
        u = complex_squash(u, dim=-1)

        W = torch.complex(self.W_real, self.W_imag)
        u_hat = torch.einsum("cnij,bnj->bcni", W, u)
        b = torch.zeros(B, NUM_CLASSES, NUM_PRIMARY, device=x.device)
        for _ in range(ROUTING_ITERS):
            c = F.softmax(b, dim=1)
            s = (c.unsqueeze(-1).to(torch.cfloat) * u_hat).sum(dim=2) + self.bias.squeeze(1)
            v = complex_squash(s, dim=-1)
            agreement = (u_hat * v.unsqueeze(2).conj()).real.sum(dim=-1)
            b = b + agreement

        probs = torch.abs(v).sum(dim=-1)
        return probs, {"digit_caps": v, "primary_phase": phase, "digit_phase": v.angle()}


class ComplexCapsuleNetA(nn.Module):
    """Full complex capsule where phase emerges from real and imaginary channels."""

    def __init__(self):
        super().__init__()
        self.name = "ComplexCapsuleA (full complex)"
        self.conv1 = nn.Conv2d(1, 256, kernel_size=9, stride=1)
        self.primary_conv_real = nn.Conv2d(256, NUM_CAP_TYPES * PRIMARY_DIM, kernel_size=CONV_KERNEL, stride=CONV_STRIDE)
        self.primary_conv_imag = nn.Conv2d(256, NUM_CAP_TYPES * PRIMARY_DIM, kernel_size=CONV_KERNEL, stride=CONV_STRIDE)
        self.W_real = nn.Parameter(torch.randn(NUM_CLASSES, NUM_PRIMARY, CLASS_DIM, PRIMARY_DIM) * 0.01)
        self.W_imag = nn.Parameter(torch.randn(NUM_CLASSES, NUM_PRIMARY, CLASS_DIM, PRIMARY_DIM) * 0.01)
        self.bias = nn.Parameter(torch.zeros(NUM_CLASSES, 1, CLASS_DIM, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        B = x.shape[0]
        h = F.relu(self.conv1(x))
        real = self.primary_conv_real(h)
        imag = self.primary_conv_imag(h)
        real = F.adaptive_avg_pool2d(real, (CAPSULE_SPATIAL, CAPSULE_SPATIAL))
        imag = F.adaptive_avg_pool2d(imag, (CAPSULE_SPATIAL, CAPSULE_SPATIAL))

        def reshape_caps(t: torch.Tensor) -> torch.Tensor:
            t = t.view(B, NUM_CAP_TYPES, PRIMARY_DIM, CAPSULE_SPATIAL, CAPSULE_SPATIAL)
            return t.permute(0, 1, 3, 4, 2).reshape(B, NUM_PRIMARY, PRIMARY_DIM)

        u = torch.complex(reshape_caps(real), reshape_caps(imag))
        u = complex_squash(u, dim=-1)

        W = torch.complex(self.W_real, self.W_imag)
        u_hat = torch.einsum("cnij,bnj->bcni", W, u)
        b = torch.zeros(B, NUM_CLASSES, NUM_PRIMARY, device=x.device)
        for _ in range(ROUTING_ITERS):
            c = F.softmax(b, dim=1)
            s = (c.unsqueeze(-1).to(torch.cfloat) * u_hat).sum(dim=2) + self.bias.squeeze(1)
            v = complex_squash(s, dim=-1)
            agreement = (u_hat * v.unsqueeze(2).conj()).real.sum(dim=-1)
            b = b + agreement

        probs = torch.abs(v).sum(dim=-1)
        return probs, {"digit_caps": v, "digit_phase": v.angle()}


count_params = __import__("ai_unity.utils", fromlist=["count_params"]).count_params
model_summary = __import__("ai_unity.utils", fromlist=["model_summary"]).model_summary
