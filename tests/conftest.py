import sys
from pathlib import Path


# Ensure the repo root is importable when tests are run without installing the package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

