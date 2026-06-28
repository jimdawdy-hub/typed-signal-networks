from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def _fmt(value: float | int | None, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return f"{value:,}{suffix}"
    return f"{value:.4f}{suffix}"


def load_comparison(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a comparison object.")
    return payload


def phase_spread(record: dict[str, Any]) -> float | None:
    phase = record.get("phase_analysis")
    if not isinstance(phase, dict) or not phase:
        return None
    values = []
    for item in phase.values():
        if isinstance(item, dict) and isinstance(item.get("overall_std"), (int, float)):
            values.append(float(item["overall_std"]))
    if not values:
        return None
    return sum(values) / len(values)


def summarize(path: Path) -> str:
    payload = load_comparison(path)
    rows = []
    for name, record in payload.items():
        rows.append(
            {
                "name": name,
                "params": record.get("params"),
                "test_acc": record.get("final_test_acc"),
                "train_acc": record.get("final_train_acc"),
                "test_loss": record.get("final_test_loss"),
                "time": record.get("total_time"),
                "phase_spread": phase_spread(record),
                "routing_collapse": (record.get("routing") or {}).get("collapse_rate"),
            }
        )

    rows.sort(key=lambda row: (row["test_acc"] is not None, row["test_acc"] or -1), reverse=True)
    lines = [f"# Summary: {path}", ""]
    lines.append("| Rank | Model | Params | Train | Test | Test loss | Time (s) | Phase spread | Routing collapse |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for idx, row in enumerate(rows, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    row["name"],
                    _fmt(row["params"]),
                    _pct(row["train_acc"]),
                    _pct(row["test_acc"]),
                    _fmt(row["test_loss"]),
                    _fmt(row["time"]),
                    _fmt(row["phase_spread"]),
                    _fmt(row["routing_collapse"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize AI Unity comparison JSON files.")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()

    for idx, path in enumerate(args.paths):
        if idx:
            print()
        print(summarize(path))


if __name__ == "__main__":
    main()
