from __future__ import annotations

import os


def main() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    from ai_unity.training import run_ternary, ternary_parser

    args = ternary_parser().parse_args()
    run_ternary(args)


if __name__ == "__main__":
    main()
