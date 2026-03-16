from __future__ import annotations

import datetime
import os
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Literal


def _setup_headless_cache_dirs() -> None:
    """Ensure Matplotlib/fontconfig can write caches in headless environments."""
    cache_dir = Path(os.environ.get("XDG_CACHE_HOME", "/tmp/.cache"))
    mpl_dir = Path(os.environ.get("MPLCONFIGDIR", "/tmp/mplconfig"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))


_setup_headless_cache_dirs()

# Matplotlib needs cache env vars set above; keep import after setup.  # noqa: E402
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation  # noqa: E402

from stonesoup.plotter import Plotter  # noqa: E402

from mht_experiments.plotting import plot_tracks_stable_xy  # noqa: E402
from mht_experiments.scenarios.bearing_range import (  # noqa: E402
    create_bearing_range_mht_example,
    external_tomht_tracks_for_bearing_range,
    tomht_initiator_for_bearing_range,
)
from mht_experiments.scenarios.crossing_targets import (  # noqa: E402
    create_crossing_scenario,
    external_tomht_tracks_for_crossing,
    tomht_initiator_for_crossing_simple,
)
from mht_experiments.trackers.tomht_tracker import (  # noqa: E402
    TOMHTParams,
    build_tomht_linear,
    build_tomht_ukf,
)


SetupName = Literal["crossing", "bearing_range"]
OperatingModeName = Literal["CUSTOM", "EXTERNAL", "INTERNAL", "BOTH"]


class TOMHTOperatingMode(str, Enum):
    CUSTOM = "CUSTOM"
    EXTERNAL = "EXTERNAL"
    INTERNAL = "INTERNAL"
    BOTH = "BOTH"

    @classmethod
    def choices(cls) -> tuple[str, ...]:
        return tuple(mode.value for mode in cls)


@dataclass(frozen=True)
class ModeConfiguration:
    use_initiator: bool
    delayed_external_start_scan: int | None


def _normalize_operating_mode(mode: str | None) -> TOMHTOperatingMode:
    if mode is None:
        return TOMHTOperatingMode.CUSTOM
    normalized = mode.upper().replace("_", "-")
    if normalized == TOMHTOperatingMode.CUSTOM.value:
        return TOMHTOperatingMode.CUSTOM
    if normalized == TOMHTOperatingMode.EXTERNAL.value:
        return TOMHTOperatingMode.EXTERNAL
    if normalized == TOMHTOperatingMode.INTERNAL.value:
        return TOMHTOperatingMode.INTERNAL
    if normalized == TOMHTOperatingMode.BOTH.value:
        return TOMHTOperatingMode.BOTH
    raise ValueError(
        "Unknown operating mode. Expected one of "
        f"{', '.join(TOMHTOperatingMode.choices())}, got {mode!r}."
    )


def _resolve_external_start_scan(
    *,
    mode: TOMHTOperatingMode,
    num_scans: int,
    external_start_scan: int | None,
    external_start_delay_scans: int | None,
) -> int | None:
    if mode == TOMHTOperatingMode.INTERNAL:
        return None

    if external_start_scan is not None and external_start_delay_scans is not None:
        raise ValueError(
            "Specify at most one of external_start_scan and "
            "external_start_delay_scans."
        )

    if external_start_scan is None and external_start_delay_scans is None:
        return None

    start_scan = (
        external_start_scan
        if external_start_scan is not None
        else external_start_delay_scans
    )
    assert start_scan is not None

    if start_scan < 0:
        raise ValueError("Delayed external-start scan must be non-negative.")
    if start_scan >= num_scans:
        raise ValueError(
            "Delayed external-start scan must fall within the scenario run. "
            f"Received {start_scan}, but scenario has {num_scans} scans."
        )
    return start_scan


def _none_to_default(value: int | None, *, default: int = 0) -> int:
    return default if value is None else value


def _resolve_mode_configuration(
    *,
    requested_mode: TOMHTOperatingMode,
    configuration: ModeConfiguration,
) -> ModeConfiguration:
    if requested_mode == TOMHTOperatingMode.CUSTOM:
        return configuration
    if requested_mode == TOMHTOperatingMode.EXTERNAL:
        return ModeConfiguration(
            use_initiator=False,
            delayed_external_start_scan=_none_to_default(
                configuration.delayed_external_start_scan
            ),
        )
    if requested_mode == TOMHTOperatingMode.INTERNAL:
        return ModeConfiguration(
            use_initiator=True,
            delayed_external_start_scan=None,
        )
    return ModeConfiguration(
        use_initiator=True,
        delayed_external_start_scan=_none_to_default(
            configuration.delayed_external_start_scan
        ),
    )


def _running_in_ipython_kernel() -> bool:
    try:
        from IPython import get_ipython  # type: ignore

        ip = get_ipython()
        return ip is not None and ip.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def _show_animation(ani) -> None:
    backend = mpl.get_backend().lower()
    headless_backends = ("agg", "pdf", "ps", "svg", "cairo", "pgf")
    force_show = os.environ.get("TOMHT_SHOW", "").lower() in {"1", "true", "yes"}
    force_skip = os.environ.get("TOMHT_NO_SHOW", "").lower() in {"1", "true", "yes"}
    is_headless = backend.startswith(headless_backends)

    if force_skip:
        print("Animation display disabled via TOMHT_NO_SHOW.")
        plt.close(ani._fig)
    elif _running_in_ipython_kernel():
        mpl.rcParams["animation.html"] = "jshtml"
        mpl.rcParams["animation.embed_limit"] = 100
        from IPython.display import HTML, display  # type: ignore

        display(HTML(ani.to_jshtml()))
        plt.close(ani._fig)
    elif is_headless and not force_show:
        # Avoid calling plt.show() on non-interactive backends (e.g., Agg in CI/headless)
        print("Headless backend detected; skipping animation display.")
        plt.close(ani._fig)
    else:
        plt.show()


def run_tomht(
    setup: SetupName,
    *,
    use_initiator: bool = True,
    operating_mode: OperatingModeName | str | None = "CUSTOM",
    external_start_scan: int | None = None,
    external_start_delay_scans: int | None = None,
    debug_display_detections: bool | None = None,
    debug_display_scan_stats: bool | None = None,
    debug_display_hypotheses: bool | None = None,
    debug_display_births: bool | None = None,
) -> None:
    requested_mode_raw = (
        operating_mode.value
        if isinstance(operating_mode, TOMHTOperatingMode)
        else ("CUSTOM" if operating_mode is None else str(operating_mode))
    )
    mode = _normalize_operating_mode(operating_mode)

    def _apply_debug_overrides(params: TOMHTParams) -> TOMHTParams:
        if debug_display_detections is not None:
            params = replace(params, debug_display_detections=debug_display_detections)
        if debug_display_scan_stats is not None:
            params = replace(params, debug_display_scan_stats=debug_display_scan_stats)
        if debug_display_hypotheses is not None:
            params = replace(params, debug_display_hypotheses=debug_display_hypotheses)
        if debug_display_births is not None:
            params = replace(params, debug_display_births=debug_display_births)
        return params

    styles: tuple[str, ...]
    if setup == "crossing":
        truths, scans, start_time, transition_model, measurement_model, config = (
            create_crossing_scenario()
        )
        timestamps = [
            start_time + datetime.timedelta(seconds=i) for i in range(len(scans))
        ]

        def build_external_starts(scan_index: int, timestamp: datetime.datetime):
            return external_tomht_tracks_for_crossing(truths, scan_index, timestamp)

        styles = ("r-", "b-")
    else:
        truths, scans, timestamps, transition_model, measurement_model, config = (
            create_bearing_range_mht_example()
        )

        def build_external_starts(scan_index: int, timestamp: datetime.datetime):
            return external_tomht_tracks_for_bearing_range(
                truths, scan_index, timestamp
            )

        styles = ("g-",)

    delayed_external_start_scan = _resolve_external_start_scan(
        mode=mode,
        num_scans=len(scans),
        external_start_scan=external_start_scan,
        external_start_delay_scans=external_start_delay_scans,
    )
    mode_config = _resolve_mode_configuration(
        requested_mode=mode,
        configuration=ModeConfiguration(
            use_initiator=use_initiator,
            delayed_external_start_scan=delayed_external_start_scan,
        ),
    )
    use_initiator = mode_config.use_initiator
    delayed_external_start_scan = mode_config.delayed_external_start_scan
    if setup == "crossing":
        initiator = (
            tomht_initiator_for_crossing_simple(start_time, measurement_model)
            if use_initiator
            else None
        )
        tracker = build_tomht_linear(
            transition_model,
            measurement_model,
            prob_detect=config.prob_detect,
            clutter_density=config.clutter_density,
            initiator=initiator,
            params=_apply_debug_overrides(
                TOMHTParams(
                    max_children_per_track=5,
                    max_missed=5,
                    prob_gate=0.9999,
                    birth_log_penalty=15.0,
                )
            ),
        )
    else:
        initiator = (
            tomht_initiator_for_bearing_range(
                timestamps[0], transition_model, measurement_model
            )
            if use_initiator
            else None
        )
        tracker = build_tomht_ukf(
            transition_model,
            measurement_model,
            prob_detect=config.prob_detect,
            clutter_density=config.clutter_density,
            initiator=initiator,
            params=_apply_debug_overrides(
                TOMHTParams(
                    max_global_hypotheses=10,
                    max_children_per_track=3,
                    max_missed=5,
                    max_births_per_scan=2,
                    birth_log_penalty=2.0,
                    unused_det_log_penalty=4.0,
                    prob_gate=0.99,
                )
            ),
        )
    print(
        "OPERATING_MODE "
        f"setup={setup} resolved={mode.value} "
        f"requested={requested_mode_raw} "
        f"births={'on' if use_initiator else 'off'} "
        f"external_starts="
        f"{f'scan:{delayed_external_start_scan}' if delayed_external_start_scan is not None else 'off'}"
    )
    if delayed_external_start_scan is not None:
        print(
            "EXTERNAL_STARTS_CONFIG "
            f"setup={setup} mode={mode.value} "
            f"start_scan={delayed_external_start_scan} "
            "source=scenario_truth_confirmed"
        )

    plotter = Plotter()
    # The base Plotter sets aspect='equal', which collides with manual x/y limits
    # and emits Matplotlib warnings in headless smoke runs. Relax to automatic aspect.
    plotter.ax.set_aspect("auto")
    frames: list[list] = []

    for n, (timestamp, detections) in enumerate(zip(timestamps, scans)):
        artists: list = []

        tracks_out = tracker.step(detections, timestamp)
        if delayed_external_start_scan == n:
            external_starts = build_external_starts(n, timestamp)
            tracker.add_external_starts(external_starts, timestamp)
            truth_indices = ",".join(
                str(int(track.metadata["scenario_truth_index"]))
                for track in external_starts
            )
            print(
                "EXTERNAL_STARTS "
                f"setup={setup} scan={n} t={timestamp} count={len(external_starts)} "
                f"truth_indices=[{truth_indices}]"
            )
            if tracker.global_hypotheses:
                tracks_out = set(tracker.global_hypotheses[0].tracks_by_id.values())
            else:
                tracks_out = set()

        ax = plotter.ax
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_xlim(*config.v_bounds[0])
        ax.set_ylim(*config.v_bounds[1])

        artists.extend(
            plotter.plot_ground_truths([t[: n + 1] for t in truths], mapping=[0, 2])
        )
        artists.extend(
            plotter.plot_measurements(
                detections, mapping=[0, 2], measurement_model=measurement_model
            )
        )
        artists.extend(plot_tracks_stable_xy(tracks_out, ax, styles=styles))

        frames.append(artists)

    if tracker.params.collect_stats:
        tracker.print_summary_stats()

    ani = animation.ArtistAnimation(
        plotter.fig, frames, interval=400, blit=True, repeat_delay=1000
    )
    _show_animation(ani)
