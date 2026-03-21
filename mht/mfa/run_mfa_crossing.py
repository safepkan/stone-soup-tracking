from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mht.mfa.mfa_runner import run_mfa

# For VS Code Interactive Window, you can tweak these and re-run:
SHOW_COMPONENTS = True

if __name__ == "__main__":
    run_mfa("crossing", show_components=SHOW_COMPONENTS)
