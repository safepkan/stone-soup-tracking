"""Parameter dataclass for the track-oriented TOMHT tracker."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from math import isfinite
from typing import Any

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
    # Optional overload mitigation:
    # when a cluster's projected Cartesian combinations exceed this threshold,
    # recursively condition on internal weak-link splits while returning feasible
    # globals for the original live cluster.
    overload_split_enabled: bool = True
    overload_split_projected_combination_threshold: int | None = 500_000
    overload_split_max_edge_removals_per_cluster: int | None = None
    # Overload split solve strategy:
    # - "conditional_exact": current sound K-best-oriented recursive conditioning
    # - "greedy_partition": experimental sound approximation with exact fallback
    overload_split_solution_mode: str = "conditional_exact"
    overload_split_greedy_ownership_metric: str = "best_leaf_score"

    # Scoring / numerical behavior.
    # prob_detect and clutter_density are scalar defaults used by
    # ConstantDetectionProbabilityModel when the tracker constructor does not
    # receive a dynamic DetectionProbabilityModel.
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
    # Sticky tree-level confirmation threshold. Internally converted to log-odds
    # and compared against max active-leaf accumulated score.
    track_confirmation_existence_probability: float = 0.9
    # Conservative tree-level deletion threshold. Internally converted to
    # log-odds and compared against max active-leaf accumulated score after
    # N-scan pruning. Confirmation/deletion form a hysteresis-style pair:
    # confirmation is sticky, while deletion removes the whole tree.
    track_deletion_existence_probability: float = 0.01
    # Sticky output-publication gate. By default, only confirmed MAP tracks are
    # published; tentative trees remain internal and inspectable.
    publish_lifecycle_states: tuple[str, ...] = ("confirmed",)
    publish_min_hits: int = 0
    publish_min_age: int = 0
    publish_min_existence_probability: float = 0.0

    # MAP-only N-scan pruning: boundary is b = k - N.
    ns_scan_window: int = 6

    # Internal start handling (kept intentionally simple in this phase).
    # Constructor initiator=None disables internal starts and leaves residual
    # detections available via get_unused_detections().
    # Public initiator-start prior. Internally converted to log-odds for
    # initiator-created roots. For a one-detection measurement initiator, a
    # principled user-side choice is:
    # logit(P_init) = log(P_D * beta_NT / lambda)
    # where beta_NT is new-target density and lambda is clutter density in the
    # same measurement-space units. The tracker keeps that as parameter-choice
    # guidance rather than adding beta_NT as a core parameter.
    initiator_start_initial_existence_probability: float = 0.8
    max_births_per_scan: int = 2
    # Birth load guards: skip births once frontier growth is already high.
    birth_skip_if_active_trees_above: int | None = 40
    birth_skip_if_active_leaves_above: int | None = 200

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
        states_raw = self.publish_lifecycle_states
        if isinstance(states_raw, str):
            raise ValueError(
                "publish_lifecycle_states must be an iterable of lifecycle states, "
                "not a string."
            )
        try:
            states = tuple(states_raw)
        except TypeError as exc:
            raise ValueError(
                "publish_lifecycle_states must be an iterable of lifecycle states."
            ) from exc

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
