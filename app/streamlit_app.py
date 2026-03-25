from __future__ import annotations

import sys
from pathlib import Path

# Keep the wrapper importable from a fresh checkout before the package is installed.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from batteryops.streamlit_app import load_demo_payload, main  # noqa: E402

__all__ = ["load_demo_payload", "main"]


if __name__ == "__main__":
    main()
