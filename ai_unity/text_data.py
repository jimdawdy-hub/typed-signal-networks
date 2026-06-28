from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ai_unity.utils import ensure_dir, write_json


TEXT_DATASET_ALIASES = {
    "dclm-1b": "codelion/dclm-baseline-1B",
    "ddclm-1b": "codelion/dclm-baseline-1B",
    "dclm-baseline-1b": "codelion/dclm-baseline-1B",
    "fineweb-edu-1b": "codelion/fineweb-edu-1B",
}


def resolve_text_dataset(name: str) -> str:
    return TEXT_DATASET_ALIASES.get(name.lower(), name)


def stream_text_rows(dataset: str, split: str = "train") -> Iterator[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("The text dataset path requires the `datasets` package.") from exc
    yield from load_dataset(resolve_text_dataset(dataset), split=split, streaming=True)


def iter_texts(dataset: str, split: str = "train", text_column: str = "text", max_docs: int | None = None):
    for idx, row in enumerate(stream_text_rows(dataset, split=split)):
        if max_docs is not None and idx >= max_docs:
            break
        text = row.get(text_column)
        if isinstance(text, str) and text.strip():
            yield idx, text, row


def inspect_dataset(
    dataset: str,
    output_dir: str | Path,
    split: str = "train",
    text_column: str = "text",
    max_docs: int = 8,
) -> dict[str, Any]:
    out = ensure_dir(output_dir)
    resolved = resolve_text_dataset(dataset)
    rows = []
    text_path = out / "samples.txt"
    with open(text_path, "w", encoding="utf-8") as f:
        for idx, text, row in iter_texts(resolved, split=split, text_column=text_column, max_docs=max_docs):
            preview = text[:500].replace("\n", "\\n")
            keys = sorted(str(key) for key in row.keys())
            rows.append({"row_index": idx, "text_length": len(text), "keys": keys, "preview": preview})
            f.write(f"--- sample {idx} len={len(text)} ---\n")
            f.write(text[:2000])
            f.write("\n\n")

    payload = {
        "requested_dataset": dataset,
        "resolved_dataset": resolved,
        "split": split,
        "text_column": text_column,
        "sample_count": len(rows),
        "samples": rows,
        "samples_text_file": str(text_path),
    }
    write_json(out / "text_dataset_inspection.json", payload)
    return payload


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Inspect streaming text datasets for AI Unity language experiments.")
    p.add_argument("--dataset", default="fineweb-edu-1b")
    p.add_argument("--split", default="train")
    p.add_argument("--text-column", default="text")
    p.add_argument("--max-docs", type=int, default=8)
    p.add_argument("--output-dir", type=Path, required=True)
    return p


def main() -> None:
    args = parser().parse_args()
    payload = inspect_dataset(
        dataset=args.dataset,
        output_dir=args.output_dir,
        split=args.split,
        text_column=args.text_column,
        max_docs=args.max_docs,
    )
    print(f"Dataset: {payload['resolved_dataset']}")
    print(f"Samples: {payload['sample_count']}")
    for sample in payload["samples"]:
        print(f"  row={sample['row_index']} len={sample['text_length']} keys={sample['keys']}")


if __name__ == "__main__":
    main()
