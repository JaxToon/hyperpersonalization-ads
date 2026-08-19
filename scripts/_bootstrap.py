"""Adds the project root to sys.path so `scripts/*.py` can `from src import ...`
when run directly (`python scripts/foo.py`) instead of as a module."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
