"""pytest bootstrap: make the repo root importable so `scripts.m4_routea` and
`scripts.m4_sparse` resolve as namespace packages."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # .../scripts/m4_routea/tests -> repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
