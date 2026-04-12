from __future__ import annotations

import datetime
import json
import os
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Mapping


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
from stonesoup.predictor.kalman import (  # noqa: E402
    KalmanPredictor,
    UnscentedKalmanPredictor,
)
from stonesoup.updater.kalman import KalmanUpdater, UnscentedKalmanUpdater  # noqa: E402

from mht.helpers.plotting import plot_tracks_stable_xy  # noqa: E402
from mht.scenarios.bearing_range import (  # noqa: E402
    create_bearing_range_mht_example,
    external_tomht_tracks_for_bearing_range,
    tomht_initiator_for_bearing_range,
)
from mht.scenarios.crossing_targets import (  # noqa: E402
    create_crossing_scenario,
    external_tomht_tracks_for_crossing,
    tomht_initiator_for_crossing_simple,
)
from mht.tomht_tracker import (  # noqa: E402
    TOMHTTracker,
    TOMHTParams,
)

SetupName = Literal["crossing", "bearing_range"]
OperatingModeName = Literal["CUSTOM", "EXTERNAL", "INTERNAL", "BOTH"]


class TOMHTOperatingMode(str, Enum):
    """Preset runner modes for internal births and external-start injection."""

    CUSTOM = "CUSTOM"
    EXTERNAL = "EXTERNAL"
    INTERNAL = "INTERNAL"
    BOTH = "BOTH"

    @classmethod
    def choices(cls) -> tuple[str, ...]:
        """Return valid string values for CLI/user-facing mode parsing."""
        return tuple(mode.value for mode in cls)


@dataclass(frozen=True)
class ModeConfiguration:
    """Resolved mode flags controlling initiator births and external starts."""

    use_initiator: bool
    external_starts_enabled: bool


@dataclass(frozen=True)
class ExternalStartTimingConfiguration:
    """Resolved scan index (and provenance) for external-start injection."""

    start_scan: int | None
    source: str


def _normalize_operating_mode(mode: str | None) -> TOMHTOperatingMode:
    """Normalize and validate a user-provided mode label."""

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


def _resolve_external_start_enablement(
    *,
    mode: TOMHTOperatingMode,
    use_external_starts: bool,
) -> bool:
    """Resolve whether external starts are enabled after mode overrides."""

    if mode == TOMHTOperatingMode.INTERNAL:
        return False
    if mode in {TOMHTOperatingMode.EXTERNAL, TOMHTOperatingMode.BOTH}:
        return True
    return use_external_starts


def _resolve_external_start_timing(
    *,
    mode: TOMHTOperatingMode,
    num_scans: int,
    external_start_scan: int | None,
    external_start_delay_scans: int | None,
) -> ExternalStartTimingConfiguration:
    """Choose and validate the scan index where external starts are injected."""

    if external_start_scan is not None and external_start_delay_scans is not None:
        raise ValueError(
            "Specify at most one of external_start_scan and "
            "external_start_delay_scans."
        )

    if external_start_scan is not None:
        start_scan = external_start_scan
        source = "explicit_scan"
    elif external_start_delay_scans is not None:
        start_scan = external_start_delay_scans
        source = "explicit_delay"
    elif mode in {TOMHTOperatingMode.EXTERNAL, TOMHTOperatingMode.BOTH}:
        start_scan = 0
        source = "mode_default_scan0"
    else:
        raise ValueError(
            "External starts are enabled in CUSTOM mode, but no external-start "
            "timing is configured. Specify external_start_scan or "
            "external_start_delay_scans."
        )

    if start_scan < 0:
        raise ValueError("External-start scan must be non-negative.")
    if start_scan >= num_scans:
        raise ValueError(
            "External-start scan must fall within the scenario run. "
            f"Received {start_scan}, but scenario has {num_scans} scans."
        )
    return ExternalStartTimingConfiguration(start_scan=start_scan, source=source)


def _resolve_mode_configuration(
    *,
    requested_mode: TOMHTOperatingMode,
    configuration: ModeConfiguration,
) -> ModeConfiguration:
    """Map a requested preset mode to concrete runtime toggles."""

    if requested_mode == TOMHTOperatingMode.CUSTOM:
        return configuration
    if requested_mode == TOMHTOperatingMode.EXTERNAL:
        return ModeConfiguration(
            use_initiator=False,
            external_starts_enabled=True,
        )
    if requested_mode == TOMHTOperatingMode.INTERNAL:
        return ModeConfiguration(
            use_initiator=True,
            external_starts_enabled=False,
        )
    return ModeConfiguration(
        use_initiator=True,
        external_starts_enabled=True,
    )


def _running_in_ipython_kernel() -> bool:
    """Return True when running under a Jupyter/IPython ZMQ kernel."""

    try:
        from IPython import get_ipython  # type: ignore

        ip = get_ipython()
        return ip is not None and ip.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def _show_animation(ani) -> None:
    """Display animation when interactive, otherwise close cleanly in headless runs."""

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


def _format_external_start_truth_indices(external_starts) -> str:
    """Format external-start truth IDs for compact logging output."""

    return ",".join(
        str(int(track.metadata["scenario_truth_index"])) for track in external_starts
    )


def load_params_overrides_json(
    path: str | Path | os.PathLike[str],
) -> dict[str, Any]:
    """Load a JSON object that maps TOMHTParams keys to override values."""

    json_path = Path(path)
    with json_path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(
            "Parameter override JSON must contain a top-level object "
            f"(mapping), got {type(loaded).__name__}."
        )
    return loaded


def parse_scenario_start_time(value: str) -> datetime.datetime:
    """Parse one ISO-8601 timestamp used to pin scenario start time."""

    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        return datetime.datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(
            "Invalid scenario start time. Use ISO-8601, for example "
            "'2026-01-01T12:00:00' or '2026-01-01T12:00:00+00:00'."
        ) from exc


def run_tomht(
    setup: SetupName,
    *,
    use_initiator: bool = True,
    use_external_starts: bool = False,
    operating_mode: OperatingModeName | str | None = "CUSTOM",
    external_start_scan: int | None = None,
    external_start_delay_scans: int | None = None,
    debug_display_detections: bool | None = None,
    debug_display_scan_stats: bool | None = None,
    debug_display_hypotheses: bool | None = None,
    debug_display_births: bool | None = None,
    scenario_start_time: datetime.datetime | None = None,
    params_overrides: Mapping[str, Any] | None = None,
) -> None:
    """Run one TOMHT scenario end-to-end and render or suppress animation output.

    This runner wires scenario generation, tracker construction, optional external
    starts, per-scan updates, and plotting into a single callable entrypoint that
    doubles as a lightweight usage example for ``TOMHTTracker``.
    """

    requested_mode_raw = (
        operating_mode.value
        if isinstance(operating_mode, TOMHTOperatingMode)
        else ("CUSTOM" if operating_mode is None else str(operating_mode))
    )
    mode = _normalize_operating_mode(operating_mode)

    def _apply_debug_overrides(params: TOMHTParams) -> TOMHTParams:
        """Apply optional debug flag overrides to baseline tracker parameters."""

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
            create_crossing_scenario(start_time=scenario_start_time)
        )
        timestamps = [
            start_time + datetime.timedelta(seconds=i) for i in range(len(scans))
        ]

        def build_external_starts(scan_index: int, timestamp: datetime.datetime):
            """Build scenario-derived external starts for the current crossing scan."""

            return external_tomht_tracks_for_crossing(truths, scan_index, timestamp)

        styles = ("r-", "b-")
    else:
        truths, scans, timestamps, transition_model, measurement_model, config = (
            create_bearing_range_mht_example(start_time=scenario_start_time)
        )
        start_time = timestamps[0]

        def build_external_starts(scan_index: int, timestamp: datetime.datetime):
            """Build scenario-derived external starts for the current bearing-range scan."""

            return external_tomht_tracks_for_bearing_range(
                truths, scan_index, timestamp
            )

        styles = ("g-",)

    external_starts_enabled = _resolve_external_start_enablement(
        mode=mode,
        use_external_starts=use_external_starts,
    )
    mode_config = _resolve_mode_configuration(
        requested_mode=mode,
        configuration=ModeConfiguration(
            use_initiator=use_initiator,
            external_starts_enabled=external_starts_enabled,
        ),
    )
    if mode_config.external_starts_enabled:
        external_start_timing = _resolve_external_start_timing(
            mode=mode,
            num_scans=len(scans),
            external_start_scan=external_start_scan,
            external_start_delay_scans=external_start_delay_scans,
        )
    else:
        # External starts are disabled; ignore timing args.
        external_start_timing = ExternalStartTimingConfiguration(
            start_scan=None, source="disabled"
        )

    use_initiator = mode_config.use_initiator
    delayed_external_start_scan = external_start_timing.start_scan
    if setup == "crossing":
        initiator = (
            tomht_initiator_for_crossing_simple(start_time, measurement_model)
            if use_initiator
            else None
        )
        predictor = KalmanPredictor(transition_model)
        updater = KalmanUpdater(measurement_model)
        params = TOMHTParams(
            prob_detect=config.prob_detect,
            clutter_density=config.clutter_density,
            hypothesis_backend="pda",
            max_children_per_track=5,
            max_missed=5,
            prob_gate=0.9999,
            birth_log_penalty=15.0,
        )
    else:
        initiator = (
            tomht_initiator_for_bearing_range(
                timestamps[0], transition_model, measurement_model
            )
            if use_initiator
            else None
        )
        predictor = UnscentedKalmanPredictor(transition_model)
        updater = UnscentedKalmanUpdater(measurement_model)
        params = TOMHTParams(
            prob_detect=config.prob_detect,
            clutter_density=config.clutter_density,
            hypothesis_backend="robust_pda",
            max_global_hypotheses=10,
            max_children_per_track=3,
            max_missed=5,
            max_births_per_scan=2,
            birth_log_penalty=2.0,
            unused_det_log_penalty=4.0,
            prob_gate=0.99,
        )
    tracker = TOMHTTracker(
        predictor,
        updater,
        initiator=initiator,
        params=_apply_debug_overrides(params),
        params_overrides=params_overrides,
    )
    print(
        "OPERATING_MODE "
        f"setup={setup} resolved={mode.value} "
        f"requested={requested_mode_raw} "
        f"scenario_start_time={start_time} "
        f"births={'on' if use_initiator else 'off'} "
        f"external_starts={'on' if mode_config.external_starts_enabled else 'off'} "
        "external_start_timing="
        f"{f'scan:{delayed_external_start_scan}' if delayed_external_start_scan is not None else 'off'} "
        f"external_start_timing_source={external_start_timing.source}"
    )
    if mode_config.external_starts_enabled:
        assert delayed_external_start_scan is not None
        print(
            "EXTERNAL_STARTS_CONFIG "
            f"setup={setup} mode={mode.value} "
            f"start_scan={delayed_external_start_scan} "
            f"timing_source={external_start_timing.source} "
            "source=scenario_truth_confirmed"
        )

    plotter = Plotter()
    # The base Plotter sets aspect='equal', which collides with manual x/y limits
    # and emits Matplotlib warnings in headless smoke runs. Relax to automatic aspect.
    plotter.ax.set_aspect("auto")
    frames: list[list] = []

    def build_scan_artists(
        *,
        scan_index: int,
        detections,
        tracks_out,
    ) -> list:
        """Render one scan's truths, detections, and tracker output as artists."""

        ax = plotter.ax
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_xlim(*config.v_bounds[0])
        ax.set_ylim(*config.v_bounds[1])

        artists: list = []
        artists.extend(
            plotter.plot_ground_truths(
                [truth[: scan_index + 1] for truth in truths], mapping=[0, 2]
            )
        )
        artists.extend(
            plotter.plot_measurements(
                detections, mapping=[0, 2], measurement_model=measurement_model
            )
        )
        artists.extend(plot_tracks_stable_xy(tracks_out, ax, styles=styles))
        return artists

    for n, (timestamp, detections) in enumerate(zip(timestamps, scans)):
        _, tracks_out = tracker.update_tracker(timestamp, detections)

        if delayed_external_start_scan == n:
            external_starts = build_external_starts(n, timestamp)
            tracker.add_external_starts(timestamp, external_starts)
            tracks_out = tracker.tracks

            print(
                "EXTERNAL_STARTS "
                f"setup={setup} scan={n} t={timestamp} count={len(external_starts)} "
                "truth_indices="
                f"[{_format_external_start_truth_indices(external_starts)}]"
            )

        frames.append(
            build_scan_artists(
                scan_index=n,
                detections=detections,
                tracks_out=tracks_out,
            )
        )

    if tracker.params.collect_stats:
        tracker.print_summary_stats()

    ani = animation.ArtistAnimation(
        plotter.fig, frames, interval=400, blit=True, repeat_delay=1000
    )
    _show_animation(ani)
