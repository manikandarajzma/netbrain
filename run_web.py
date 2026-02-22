"""
Launcher for the Atlas web app. Ensures the parent directory is on sys.path
so that the atlas package is found when running from the project root.
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
_parent = _root.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from atlas.app import app  # noqa: E402

__all__ = ["app"]
