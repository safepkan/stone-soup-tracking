"""Parameter dataclass for the track-oriented TOMHT tracker."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields, replace
from math import isfinite
from typing import Any

from .tomht_scoring import _existence_probability_to_log_odds


@dataclass(frozen=True)
class TOMHTParams:
    """Flat tracker configuration for the track-oriented TO-MHT implementation.

    Core search controls (part of the normal frontier-control stack):
    - ``max_children_per_leaf``: per-active-leaf local branching breadth
      (core, but can be effectively disabled with a very high value),
    - ``max_global_hypotheses`` (K): per-cluster K-best retained-global breadth,
    - ``ns_scan_window`` (N): MAP-only N-scan pruning depth.

    The remaining fields follow in grouped definition order:
    - detection / scoring model (``prob_detect``, ``clutter_density``, gate,
      numerical floor),
    - start priors (internal initiator and external starts),
    - confirmation, deletion, and publication gates,
    - internal-birth cap and load guards,
    - solver backend and tractability guardrails,
    - internal profiling/debug fast-path switches,
    - debug / instrumentation toggles.
    """

    # Core search controls (N, K, and per-leaf local branching).
    # Per-active-leaf local branching cap: hit/miss continuation candidates
    # retained per expanded leaf (the miss alternative is always preserved). The
    # default is loose enough that it usually does not bind; a very high value
    # disables it, leaving search shape to K and N.
    max_children_per_leaf: int = 5
    # K: per-cluster K-best retained globals per update (the solver's
    # max_results), rebuilt fresh each update. Post-solve supported-leaf pruning
    # keeps only leaves appearing in one of these top-K globals, so K bounds the
    # hypothesis breadth that survives into the next scan.
    max_global_hypotheses: int = 20
    # N: MAP-only N-scan pruning depth; boundary b = k - N. Scans of association
    # ambiguity retained before a tree root commits to its MAP child.
    ns_scan_window: int = 6

    # Detection / scoring model.
    # prob_detect and clutter_density are scalar defaults used by
    # ConstantDetectionProbabilityModel when the constructor receives no dynamic
    # DetectionProbabilityModel. They are validated when that constant model is
    # constructed: prob_detect must satisfy 0 < p < 1, and clutter_density must be
    # finite and > 0. The 0.0 clutter_density default is an invalid sentinel; set
    # an explicit density or pass a custom DPM. Both are ignored when a custom DPM
    # is supplied.
    prob_detect: float = 0.9
    # Clutter density lambda in measurement-space units: detections per unit
    # measurement-volume per scan. Must match the measurement coordinates used by
    # the hypothesiser NLL computation.
    clutter_density: float = 0.0
    # Main-path local-association gate: Mahalanobis threshold (non-squared), used
    # only by the tracker-owned default hypothesiser.
    mahalanobis_gate_threshold: float = 3.0
    # Dimensionless floor for P_D and (1 - P_D) inside scoring logs, keeping
    # legitimate dynamic-DPM endpoints finite. Not a clutter-density floor.
    log_epsilon: float = 1e-12

    # Start priors (initial existence prior for new track roots).
    # Constructor initiator=None disables internal starts and leaves residual
    # detections available via get_unused_detections(). For a one-detection
    # measurement initiator a principled user-side choice is
    # logit(P_init) = log(P_D * beta_NT / lambda), where beta_NT is new-target
    # density and lambda is clutter density in the same measurement-space units;
    # the tracker keeps that as guidance rather than adding beta_NT as a core
    # parameter.
    initiator_start_initial_existence_probability: float = 0.8
    # External-start prior; default 0.95 reflects externally confirmed starts.
    external_start_initial_existence_probability: float = 0.95

    # Confirmation, deletion, and publication.
    # Sticky tree-level confirmation threshold, compared (as log-odds) against the
    # max active-leaf accumulated score.
    track_confirmation_existence_probability: float = 0.9
    # Conservative tree-level deletion threshold, compared (as log-odds) against
    # the max active-leaf accumulated score after N-scan pruning. Confirmation and
    # deletion form a hysteresis-style pair: confirmation is sticky, deletion
    # removes the whole tree.
    track_deletion_existence_probability: float = 0.01
    # Base miss threshold for the default post-N-scan deleter. Effective threshold
    # uses an N-scan-aware floor (see lifecycle helper).
    max_missed: int = 5
    # Candidate-leaf selection mode applied after N-scan pruning. Controls which
    # leaves the configured deleter evaluates (default miss-count deleter and
    # custom Stone Soup deleters alike).
    # - "all_active_leaves": terminate only if all active leaves exceed threshold
    # - "map_leaf": terminate if MAP leaf exceeds threshold
    # - "global_k_leaves": terminate if all retained rebuilt-global leaves exceed
    #   threshold (fallback to active leaves if unavailable after N-scan)
    track_miss_termination_mode: str = "map_leaf"
    # Sticky output-publication gate. By default only confirmed MAP tracks are
    # published; tentative trees remain internal and inspectable.
    publish_lifecycle_states: tuple[str, ...] = ("confirmed",)
    publish_min_hits: int = 0
    publish_min_age: int = 0
    publish_min_existence_probability: float = 0.0

    # Internal births.
    # Deterministic per-scan cap on internal births; a guardrail, not a quality
    # filter (prefer initiator-side filtering if it fires routinely).
    max_births_per_scan: int = 10
    # Optional birth load guards: skip births once frontier growth is already
    # high. Disabled by default; set scenario-specific values only when this
    # emergency safety valve is wanted.
    birth_skip_if_active_trees_above: int | None = None
    birth_skip_if_active_leaves_above: int | None = None

    # Solver and tractability guardrails.
    # Exact cluster-solver backend.
    # - "branch_and_bound": default exact DFS branch-and-bound backend
    # - "exhaustive": exact reference/fallback backend
    # - "ortools": experimental CP-SAT backend
    cluster_solver_backend: str = "branch_and_bound"
    # Optional pre-solve per-tree frontier cap used only as a safety valve. The
    # high default keeps this in a guardrail role, not the primary pruning
    # mechanism.
    max_leaves_per_track_tree: int | None = 500
    # Optional hard cap for one cluster's projected Cartesian leaf combinations.
    # If exceeded, cluster rebuild fails explicitly (no adaptive trimming/retry).
    max_projected_cluster_combinations: int | None = None
    # Overload mitigation: when a cluster's projected Cartesian combinations
    # exceed the threshold, split solving stays internal and returns feasible
    # globals for the original live cluster.
    overload_split_enabled: bool = True
    overload_split_projected_combination_threshold: int | None = 500_000
    overload_split_max_edge_removals_per_cluster: int | None = None
    # Overload split solve strategy:
    # - "greedy_partition": default operational fallback; sound but approximate.
    #   Greedily partitions contested cut detections, falling back to
    #   conditional_exact if the partition cannot produce feasible
    #   original-cluster globals.
    # - "conditional_exact": reference / higher-compute recursive conditional
    #   mode. Enumerates cut assignments to preserve K-best-oriented behavior
    #   under overload.
    overload_split_solution_mode: str = "greedy_partition"
    overload_split_greedy_ownership_metric: str = "best_leaf_score"

    # Internal profiling/debug fast-path switches.
    # Tracker-owned default hypothesiser may expand from the current leaf state
    # without reconstructing Track history. Custom hypothesisers still receive
    # normal Stone Soup Tracks.
    enable_default_hypothesiser_state_fast_path: bool = True
    # Tracker-owned default miss-count deleter may check leaf metadata directly
    # without reconstructing Track history. Custom deleters still receive normal
    # Stone Soup Tracks.
    enable_default_miss_deleter_fast_path: bool = True

    # Debug / instrumentation toggles.
    debug_display_detections: bool = False
    debug_display_config: bool = False
    debug_display_scan_stats: bool = True
    debug_display_hypotheses: bool = True
    debug_display_births: bool = True
    debug_display_map_miss_hist: bool = False
    debug_display_expansion_frontier: bool = False
    debug_births_max: int = 5
    debug_globals_max: int = 5
    collect_stats: bool = True

    def __post_init__(self) -> None:
        """Validate parameter values with constrained domains."""
        _existence_probability_to_log_odds(
            self.external_start_initial_existence_probability,
            parameter_name="external_start_initial_existence_probability",
        )
        _existence_probability_to_log_odds(
            self.track_confirmation_existence_probability,
            parameter_name="track_confirmation_existence_probability",
        )
        _existence_probability_to_log_odds(
            self.track_deletion_existence_probability,
            parameter_name="track_deletion_existence_probability",
        )
        _existence_probability_to_log_odds(
            self.initiator_start_initial_existence_probability,
            parameter_name="initiator_start_initial_existence_probability",
        )
        self._validate_overload_split_params()
        self._validate_publication_params()

    def _validate_overload_split_params(self) -> None:
        """Validate overload split mode controls."""
        valid_solution_modes = {"conditional_exact", "greedy_partition"}
        if self.overload_split_solution_mode not in valid_solution_modes:
            valid_str = ", ".join(repr(mode) for mode in sorted(valid_solution_modes))
            raise ValueError(
                "overload_split_solution_mode must be one of "
                f"({valid_str}); got {self.overload_split_solution_mode!r}."
            )

        valid_ownership_metrics = {"best_leaf_score"}
        if self.overload_split_greedy_ownership_metric not in valid_ownership_metrics:
            valid_str = ", ".join(
                repr(metric) for metric in sorted(valid_ownership_metrics)
            )
            raise ValueError(
                "overload_split_greedy_ownership_metric must be one of "
                f"({valid_str}); got "
                f"{self.overload_split_greedy_ownership_metric!r}."
            )

    def _validate_publication_params(self) -> None:
        """Validate sticky output-publication gate controls."""
        states_raw: object = self.publish_lifecycle_states
        if isinstance(states_raw, str):
            raise ValueError(
                "publish_lifecycle_states must be an iterable of lifecycle states, "
                "not a string."
            )
        if not isinstance(states_raw, Iterable):
            raise ValueError(
                "publish_lifecycle_states must be an iterable of lifecycle states."
            )
        states = tuple(states_raw)

        valid_states = {"tentative", "confirmed"}
        invalid_states = sorted(set(states).difference(valid_states))
        if invalid_states:
            invalid_states_str = ", ".join(repr(state) for state in invalid_states)
            valid_states_str = ", ".join(repr(state) for state in sorted(valid_states))
            raise ValueError(
                "publish_lifecycle_states must contain only valid lifecycle states "
                f"({valid_states_str}); got {invalid_states_str}."
            )
        if int(self.publish_min_hits) < 0:
            raise ValueError("publish_min_hits must be >= 0.")
        if int(self.publish_min_age) < 0:
            raise ValueError("publish_min_age must be >= 0.")

        publish_min_existence_probability = float(
            self.publish_min_existence_probability
        )
        if (
            not isfinite(publish_min_existence_probability)
            or publish_min_existence_probability < 0.0
            or publish_min_existence_probability >= 1.0
        ):
            raise ValueError(
                "publish_min_existence_probability must satisfy 0.0 <= p < 1.0."
            )


def apply_params_overrides(
    params: TOMHTParams,
    params_overrides: Mapping[str, Any] | None,
) -> TOMHTParams:
    """Apply JSON-style parameter overrides onto a frozen ``TOMHTParams``."""
    if params_overrides is None:
        return params
    if not isinstance(params_overrides, Mapping):
        raise TypeError(
            "params_overrides must be a mapping of TOMHTParams field names to values."
        )
    overrides = dict(params_overrides)
    if not overrides:
        return params
    non_string_keys = [key for key in overrides if not isinstance(key, str)]
    if non_string_keys:
        non_string_keys_str = ", ".join(repr(key) for key in non_string_keys)
        raise TypeError(
            "params_overrides keys must be strings matching TOMHTParams fields; "
            f"got: {non_string_keys_str}."
        )
    valid_keys = {field.name for field in fields(TOMHTParams)}
    invalid_keys = sorted(set(overrides).difference(valid_keys))
    if invalid_keys:
        invalid_keys_str = ", ".join(invalid_keys)
        raise ValueError(f"Unknown TOMHTParams override key(s): {invalid_keys_str}.")
    return replace(params, **overrides)
