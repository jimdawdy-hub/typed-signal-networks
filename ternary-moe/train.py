"""Historical entry point for the Ternary + MoE experiment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_unity.run_ternary import main


if __name__ == "__main__":
    main()
