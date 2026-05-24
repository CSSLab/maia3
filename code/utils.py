"""Compatibility imports for the packaged Maia3 utility helpers."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from maia3.utils import *  # noqa: F401,F403
