"""Parameter dataclass for the track-oriented TOMHT tracker."""

from __future__ import annotations

from dataclasses import dataclass

from .tomht_scoring import _existence_probability_to_log_odds


@dataclass(frozen=True)
class TOMHTParams:
    """Flat tracker configuration for the track-oriented TO-MHT implementation.

    Stable operational controls:
    - per-leaf local branching and local frontier safety-valves,
    - whole-track miss termination after N-scan pruning,
    - MAP-only N-scan pruning window,
    - optional debug/stat visibility toggles.

    Compatibility note:
    ``max_global_hypotheses`` is retained as a cap for how many rebuilt globals
    are kept per cluster for debug/snapshot storage; it is no longer a persistent
    beam frontier carried scan-to-scan.
    """

    # Local expansion / lifecycle controls.
    max_children_per_track: int = 5
    # Optional pre-solve per-tree frontier cap used only as a safety valve.
    # The high default keeps this in a tractability guardrail role, not as the
    # primary pruning mechanism.
    max_leaves_per_track_tree: int | None = 500
    # Base miss threshold used by node-native post-N-scan whole-track lifecycle.
    # Effective threshold uses an N-scan-aware floor (see helper below).
    max_missed: int = 5
    # Whole-track candidate-leaf selection mode applied after N-scan pruning.
    # This controls which leaves are evaluated for lifecycle termination in both
    # lanes (node-native and optional Stone Soup deleter lane).
    # - "all_active_leaves": terminate only if all active leaves exceed threshold
    # - "map_leaf": terminate if MAP leaf exceeds threshold
    # - "global_k_leaves": terminate if all retained rebuilt-global leaves exceed
    #   threshold (fallback to active leaves if unavailable after N-scan)
    track_miss_termination_mode: str = "map_leaf"

    # Rebuilt-global storage cap (debug/inspection cap, not persistent beam state).
    max_global_hypotheses: int = 20
    # Exact cluster-solver backend.
    # - "branch_and_bound": default exact DFS branch-and-bound backend
    # - "exhaustive": exact reference/fallback backend
    # - "ortools": experimental CP-SAT backend
    cluster_solver_backend: str = "branch_and_bound"
    # Optional hard cap for one cluster's projected Cartesian leaf combinations.
    # If exceeded, cluster rebuild fails explicitly (no adaptive trimming/retry).
    max_projected_cluster_combinations: int | None = None
    # Optional approximate overload mitigation:
    # when a cluster's projected Cartesian combinations exceed this threshold,
    # iteratively sever weakest conflict edges and solve resulting subclusters.
    overload_split_enabled: bool = True
    overload_split_projected_combination_threshold: int | None = 500_000
    overload_split_max_edge_removals_per_cluster: int | None = None
    # Narrow safety-net: if an exact cluster is infeasible, allow relaxation only
    # for forced historical keys that are already shared across tracks.
    historical_conflict_relaxation_enabled: bool = True

    # Scoring / numerical behavior.
    scoring_mode: str = "nll"
    log_epsilon: float = 1e-12
    prob_detect: float = 0.9
    # Main-path gate control: Mahalanobis threshold (non-squared).
    mahalanobis_gate_threshold: float = 3.0
    # Clutter density lambda in measurement-space units:
    # detections per unit measurement-volume per scan. Must match the same
    # measurement coordinates used by the hypothesiser NLL computation.
    clutter_density: float = 0.0
    # Public external-start prior. Internally converted to log-odds for the root
    # log_delta; default 0.95 reflects externally confirmed starts.
    external_start_initial_existence_probability: float = 0.95

    # MAP-only N-scan pruning: boundary is b = k - N.
    ns_scan_window: int = 6

    # Internal birth handling (kept intentionally simple in this phase).
    max_births_per_scan: int = 2
    birth_log_penalty: float = 8.0
    # Birth load guards: skip births once frontier growth is already high.
    birth_skip_if_active_trees_above: int | None = 40
    birth_skip_if_active_leaves_above: int | None = 200

    # Birth sanity guards.
    birth_max_abs_pos: float = 1e5
    birth_max_covar_trace: float = 1e12

    # Debug / instrumentation toggles.
    debug_display_detections: bool = False
    debug_display_config: bool = False
    debug_display_scan_stats: bool = True
    debug_display_hypotheses: bool = True
    debug_display_births: bool = True
    debug_display_map_miss_hist: bool = False
    debug_births_max: int = 5
    debug_globals_max: int = 5
    collect_stats: bool = True

    def __post_init__(self) -> None:
        """Validate parameter values with constrained domains."""
        _existence_probability_to_log_odds(
            self.external_start_initial_existence_probability,
            parameter_name="external_start_initial_existence_probability",
        )
