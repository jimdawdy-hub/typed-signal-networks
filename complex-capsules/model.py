"""Compatibility imports for the historical complex-capsules folder."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_unity.complex_capsules import *  # noqa: F401,F403
