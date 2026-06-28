from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class TernaryLinear(nn.Module):
    """Linear layer whose forward-pass weights become {-1, 0, +1}."""

    def __init__(self, in_features: int, out_features: int, warmup_steps: int = 500):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.warmup_steps = warmup_steps
        self.step_count = 0
        self.weight_float = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_uniform_(self.weight_float)

    def ternarize(self, w: torch.Tensor) -> torch.Tensor:
        threshold = 0.7 * w.abs().mean()
        return torch.where(w.abs() > threshold, torch.sign(w), torch.zeros_like(w))

    def forward_weight(self) -> torch.Tensor:
        if self.training:
            self.step_count += 1
        alpha = min(1.0, self.step_count / max(self.warmup_steps, 1))
        w_ternary = self.ternarize(self.weight_float)
        if self.training and alpha < 1.0:
            return (1 - alpha) * self.weight_float + alpha * w_ternary
        return w_ternary

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.forward_weight(), self.bias)

    def sparsity(self) -> float:
        w = self.ternarize(self.weight_float)
        return (w == 0).float().mean().item()


class FloatLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def sparsity(self) -> float:
        return 0.0


class Expert(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int, use_ternary: bool = False):
        super().__init__()
        LinearClass = TernaryLinear if use_ternary else FloatLinear
        self.net = nn.Sequential(
            LinearClass(in_features, hidden_features),
            nn.ReLU(),
            LinearClass(hidden_features, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoELayer(nn.Module):
    """Mixture of experts (router chooses top-k specialist subnetworks)."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_experts: int = 4,
        top_k: int = 1,
        use_ternary: bool = False,
    ):
        super().__init__()
        if top_k < 1 or top_k > num_experts:
            raise ValueError("top_k must be between 1 and num_experts.")
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList(
            [Expert(in_features, hidden_features, out_features, use_ternary) for _ in range(num_experts)]
        )
        self.router = nn.Linear(in_features, num_experts)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        router_logits = self.router(x)
        router_probs = F.softmax(router_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        gather_index = top_k_indices.unsqueeze(-1).expand(-1, -1, expert_outputs.shape[-1])
        selected_outputs = torch.gather(expert_outputs, 1, gather_index)
        output = (selected_outputs * top_k_probs.unsqueeze(-1)).sum(dim=1)

        avg_probs = router_probs.mean(dim=0)
        ideal = torch.ones_like(avg_probs) / self.num_experts
        load_balance_loss = F.mse_loss(avg_probs, ideal)
        return output, load_balance_loss, router_probs

    @torch.no_grad()
    def routing_diagnostics(self, x: torch.Tensor, y: torch.Tensor | None = None) -> dict[str, torch.Tensor | float]:
        router_probs = F.softmax(self.router(x), dim=-1)
        _, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        usage = torch.bincount(top_k_indices.reshape(-1), minlength=self.num_experts).float()
        usage_fraction = usage / usage.sum().clamp_min(1)
        entropy = -(router_probs * router_probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
        collapse_rate = usage_fraction.max()
        diagnostics: dict[str, torch.Tensor | float] = {
            "usage_fraction": usage_fraction.cpu(),
            "entropy": float(entropy.detach().cpu()),
            "collapse_rate": float(collapse_rate.detach().cpu()),
        }
        if y is not None:
            per_class = torch.zeros(10, self.num_experts, device=x.device)
            for digit in range(10):
                mask = y == digit
                if mask.any():
                    selected = top_k_indices[mask].reshape(-1)
                    per_class[digit] = torch.bincount(selected, minlength=self.num_experts).float()
            diagnostics["usage_by_class"] = (per_class / per_class.sum(dim=1, keepdim=True).clamp_min(1)).cpu()
        return diagnostics


class FloatSingle(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, output_dim: int = 10):
        super().__init__()
        self.name = "FloatSingle (baseline)"
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.net(x), torch.zeros((), device=x.device)


class FloatMoE(nn.Module):
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 128,
        output_dim: int = 10,
        num_experts: int = 4,
        top_k: int = 1,
    ):
        super().__init__()
        self.name = f"FloatMoE ({num_experts} experts, top-{top_k})"
        self.moe = MoELayer(input_dim, hidden_dim, output_dim, num_experts, top_k, use_ternary=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output, lb_loss, _ = self.moe(x)
        return output, lb_loss


class TernarySingle(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, output_dim: int = 10):
        super().__init__()
        self.name = "TernarySingle"
        self.net = nn.Sequential(
            TernaryLinear(input_dim, hidden_dim),
            nn.ReLU(),
            TernaryLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            TernaryLinear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.net(x), torch.zeros((), device=x.device)

    def sparsity(self) -> float:
        sparsities = [m.sparsity() for m in self.modules() if isinstance(m, TernaryLinear)]
        return sum(sparsities) / len(sparsities) if sparsities else 0.0


class TernaryMoE(nn.Module):
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 128,
        output_dim: int = 10,
        num_experts: int = 4,
        top_k: int = 1,
    ):
        super().__init__()
        self.name = f"TernaryMoE ({num_experts} experts, top-{top_k})"
        self.moe = MoELayer(input_dim, hidden_dim, output_dim, num_experts, top_k, use_ternary=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output, lb_loss, _ = self.moe(x)
        return output, lb_loss

    def sparsity(self) -> float:
        sparsities = [m.sparsity() for m in self.modules() if isinstance(m, TernaryLinear)]
        return sum(sparsities) / len(sparsities) if sparsities else 0.0


count_params = __import__("ai_unity.utils", fromlist=["count_params"]).count_params
model_summary = __import__("ai_unity.utils", fromlist=["model_summary"]).model_summary
