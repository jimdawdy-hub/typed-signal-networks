from __future__ import annotations

import os


def main() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    from ai_unity.training import complex_parser, run_complex

    args = complex_parser().parse_args()
    run_complex(args)


if __name__ == "__main__":
    main()
