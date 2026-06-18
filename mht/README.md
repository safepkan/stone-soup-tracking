# TO-MHT tracker (`mht`)

A Stone Soup based, track-oriented multiple-hypothesis tracker (TO-MHT).

## Start here

**[`TO_MHT_API.md`](TO_MHT_API.md)** is the API and integration guide, and the
primary reference for integrators. It covers the public surface, the
detection-probability / clutter-density model, internal and external track
starts, confirmation / deletion / publication, public vs. internal track IDs,
and the stable output-metadata contract.

## Public API

Import the stable public names from `mht.api`:

```python
from mht.api import (
    TOMHTTracker,
    TOMHTParams,
    DetectionProbabilityModel,
    ConstantDetectionProbabilityModel,
    MAPAssociationHistorySnapshot,
    MapTrackAssociationHistory,
    MapAssociationStep,
)
```

Integration code should import from `mht.api`, not the internal `mht.tomht_*`
modules. The association-history return types above are part of the stable
public surface. Other inspection/debug snapshot types and the `tomht_*` modules
are not part of the stable surface and may change between releases.

## Layout

- `api.py` — stable public import surface.
- `TO_MHT_API.md` — API and integration guide.
- `tomht_*.py` — tracker implementation; `tomht_tracker.py` is the orchestrator
  and the best entry point for reading the code, with the remaining modules
  holding the per-phase logic.
- `scenarios/`, `runners/` — example scenarios and runnable demos.
- `tests/` — unit and integration tests.

## Requirements

Python 3.10 or newer, with `numpy`, `scipy`, `stonesoup`, `ordered-set`,
`ortools`, and `matplotlib`.

## Running the demos and tests

From a directory containing `mht/`:

```bash
# Run a demo scenario:
python mht/runners/run_tomht_crossing.py
# or:
python mht/runners/run_tomht_bearing_range.py

# Run the test suite:
python -m pytest mht/tests
```
