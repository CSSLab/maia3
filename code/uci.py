"""Compatibility wrapper for the packaged Maia3 UCI engine.

Prefer `maia3-uci` after installing the package, or `python -m maia3.uci`
from the repository root.
"""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from maia3.uci import main


if __name__ == "__main__":
    main()
