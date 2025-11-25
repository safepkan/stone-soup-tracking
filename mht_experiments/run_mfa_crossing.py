from __future__ import annotations

from mht_experiments.runners.mfa_runner import run_mfa

# For VS Code Interactive Window, you can tweak these and re-run:
SHOW_COMPONENTS = True

if __name__ == "__main__":
    run_mfa("crossing", show_components=SHOW_COMPONENTS)
