"""Shared research harness for AI Unity experiments.

The package root stays lightweight so entry-point scripts can set
CUDA_VISIBLE_DEVICES before importing PyTorch.
"""

__all__ = [
    "BaselineMLP",
    "ComplexCapsuleNetA",
    "ComplexCapsuleNetB",
    "ComplexCapsuleNetSpatialB",
    "FloatMoE",
    "FloatSingle",
    "MoELayer",
    "RealCapsuleNet",
    "RealCapsuleNetLarge",
    "TernaryLinear",
    "TernaryMoE",
    "TernarySingle",
]


def __getattr__(name: str):
    if name in {
        "BaselineMLP",
        "ComplexCapsuleNetA",
        "ComplexCapsuleNetB",
        "ComplexCapsuleNetSpatialB",
        "RealCapsuleNet",
        "RealCapsuleNetLarge",
    }:
        from ai_unity import complex_capsules

        return getattr(complex_capsules, name)
    if name in {"FloatMoE", "FloatSingle", "MoELayer", "TernaryLinear", "TernaryMoE", "TernarySingle"}:
        from ai_unity import ternary_moe

        return getattr(ternary_moe, name)
    raise AttributeError(name)
