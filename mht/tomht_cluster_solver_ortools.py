"""Experimental exact CP-SAT backend for TO-MHT cluster-solver problems.

Status/positioning:
- This backend is exact under the current cluster-solver contract.
- In the current K-best extraction form (repeated solves + no-good cuts), it is
  not a runtime win on the primary replay workload used during this phase.
- It is retained for comparison, fallback exact solving, and future experiments
  (for example hybrid backend selection or alternative K-best extraction methods).

Optional profiling:
- Set ``TOMHT_ORTOOLS_PROFILE=1`` to collect per-solve timing/size metadata.
- Set ``TOMHT_ORTOOLS_PROFILE_PRINT=1`` to emit one JSON line per solve:
  ``ORTOOLS_SOLVE_PROFILE {...}``.
- Set ``TOMHT_ORTOOLS_PROFILE_MIN_TOTAL_MS=<float>`` to suppress printout for
  very small solves.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import time as wall_clock
from typing import Any

from ortools.sat.python import cp_model

from .tomht_cluster_solver import (
    ClusterSolverDiagnostics,
    ClusterSolverLeafOption,
    ClusterSolverProblem,
    ClusterSolverResult,
    ClusterSolverSolution,
    TopKSolutionHeap,
    score_selected_leaf_ids_if_exact_feasible,
    validate_cluster_solver_problem,
)
from .utils import (
    env_flag as _env_flag,
    env_float as _env_float,
    get_process_maxrss_mb as _get_process_maxrss_mb,
    ns_to_ms as _ns_to_ms,
)


@dataclass(frozen=True)
class _LeafVariable:
    """CP-SAT variable metadata for one leaf option."""

    variable: cp_model.IntVar
    leaf_option: ClusterSolverLeafOption


class ORToolsClusterSolver:
    """Experimental exact backend: CP-SAT + repeated no-good cuts for K-best.

    Operational expectation:
    - Keep exhaustive as the recommended default backend for current replay
      workloads in this phase.
    - Use this backend as an exact experimental/reference path and when
      profiling solver behavior.

    Objective scaling policy:
    - CP-SAT objective coefficients are integers, so each leaf score is scaled
      as ``round(score * score_scale)``.
    - The solver optimizes the scaled objective exactly.
    - Returned solution scores are recomputed with the original float scores and
      include ``problem.constant_score_offset``.

    Risk-mitigation knob:
    - ``extra_k_best_iterations`` optionally runs additional repeated solves past
      ``problem.max_results`` and keeps the best exact-float-scored K via
      ``TopKSolutionHeap``.

    Profiling env flags:
    - ``TOMHT_ORTOOLS_PROFILE`` enables collection of per-solve timing and
      size metrics.
    - ``TOMHT_ORTOOLS_PROFILE_PRINT`` controls whether profile JSON lines are
      printed.
    - ``TOMHT_ORTOOLS_PROFILE_MIN_TOTAL_MS`` sets a print threshold.
    """

    def __init__(
        self,
        *,
        score_scale: int = 1_000_000,
        max_time_seconds: float | None = None,
        extra_k_best_iterations: int = 0,
    ) -> None:
        if int(score_scale) <= 0:
            raise ValueError("score_scale must be > 0.")
        if max_time_seconds is not None and float(max_time_seconds) <= 0.0:
            raise ValueError("max_time_seconds must be > 0 when provided.")
        if int(extra_k_best_iterations) < 0:
            raise ValueError("extra_k_best_iterations must be >= 0.")

        self._score_scale = int(score_scale)
        self._max_time_seconds = (
            None if max_time_seconds is None else float(max_time_seconds)
        )
        self._extra_k_best_iterations = int(extra_k_best_iterations)
        self._last_diagnostics: ClusterSolverDiagnostics | None = None
        self._last_profile: dict[str, object] | None = None
        self._profiling_enabled = _env_flag("TOMHT_ORTOOLS_PROFILE", default=False)
        self._profiling_print = _env_flag(
            "TOMHT_ORTOOLS_PROFILE_PRINT",
            default=True,
        )
        self._profiling_min_total_ms = _env_float(
            "TOMHT_ORTOOLS_PROFILE_MIN_TOTAL_MS",
            default=0.0,
        )

    def solve(self, problem: ClusterSolverProblem) -> ClusterSolverResult:
        validate_cluster_solver_problem(problem)
        self._last_profile = None
        profiling_enabled = self._profiling_enabled
        solve_start_ns = wall_clock.perf_counter_ns() if profiling_enabled else 0
        maxrss_before_mb = _get_process_maxrss_mb() if profiling_enabled else 0.0

        model_vars_and_keys_ns = 0
        model_exactly_one_ns = 0
        model_conflict_constraints_ns = 0
        model_objective_ns = 0
        solve_loop_total_ns = 0
        solve_calls_total_ns = 0
        solve_call_max_ns = 0
        decode_selected_ns = 0
        exact_rescore_ns = 0
        topk_push_ns = 0
        nogood_add_ns = 0
        finalize_ns = 0
        status_counts: dict[str, int] = {}

        # Build one CP-SAT model for this cluster problem, then resolve it
        # repeatedly with no-good cuts to extract K-best solutions.
        # OR-Tools runtime exposes these methods, but current typing stubs can
        # miss some CamelCase API members (NewBoolVar/Add/Maximize). Keep one
        # strongly-typed handle for Solve(), and use a narrow Any-typed alias
        # for model-building calls to avoid false-positive mypy attr checks.
        raw_model = cp_model.CpModel()
        model: Any = raw_model
        leaf_option_by_leaf_id: dict[int, ClusterSolverLeafOption] = {}

        leaf_vars: list[_LeafVariable] = []
        conflict_vars_by_key: dict[tuple[int, int], list[cp_model.IntVar]] = {}
        conflict_constraints = 0

        # One Boolean variable per leaf option.
        # Track constraint: exactly one leaf must be selected per track.
        for track in problem.track_options:
            per_track_vars: list[cp_model.IntVar] = []
            if profiling_enabled:
                t_vars_ns = wall_clock.perf_counter_ns()
            for leaf in track.leaf_options:
                leaf_id = int(leaf.leaf_id)
                if leaf_id in leaf_option_by_leaf_id:
                    raise RuntimeError(
                        "Cluster solver problem has duplicate leaf IDs. "
                        f"leaf_id={leaf_id}"
                    )
                leaf_option_by_leaf_id[leaf_id] = leaf

                var = model.NewBoolVar(f"leaf_track_{int(leaf.track_id)}_id_{leaf_id}")
                per_track_vars.append(var)
                leaf_vars.append(_LeafVariable(variable=var, leaf_option=leaf))

                for key in leaf.full_history_conflict_keys:
                    conflict_vars_by_key.setdefault(key, []).append(var)
            if profiling_enabled:
                model_vars_and_keys_ns += wall_clock.perf_counter_ns() - t_vars_ns

            if profiling_enabled:
                t_exactly_one_ns = wall_clock.perf_counter_ns()
            model.Add(sum(per_track_vars) == 1)
            if profiling_enabled:
                model_exactly_one_ns += wall_clock.perf_counter_ns() - t_exactly_one_ns

        # Conflict constraint by history key:
        # for each key, at most one selected leaf may contain it.
        if profiling_enabled:
            t_conflict_constraints_ns = wall_clock.perf_counter_ns()
        for vars_for_key in conflict_vars_by_key.values():
            if len(vars_for_key) > 1:
                model.Add(sum(vars_for_key) <= 1)
                conflict_constraints += 1
        if profiling_enabled:
            model_conflict_constraints_ns += (
                wall_clock.perf_counter_ns() - t_conflict_constraints_ns
            )

        # CP-SAT objective coefficients must be integers, so leaf scores are
        # scaled and rounded. We later recompute exact float scores for ranking.
        if profiling_enabled:
            t_objective_ns = wall_clock.perf_counter_ns()
        scaled_objective_terms = []
        for leaf_var in leaf_vars:
            scaled_score = self._scale_score(float(leaf_var.leaf_option.score))
            if scaled_score == 0:
                continue
            scaled_objective_terms.append(scaled_score * leaf_var.variable)
        model.Maximize(sum(scaled_objective_terms))
        if profiling_enabled:
            model_objective_ns += wall_clock.perf_counter_ns() - t_objective_ns

        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        if self._max_time_seconds is not None:
            solver.parameters.max_time_in_seconds = self._max_time_seconds

        # Keep only the best K by exact float score across all extracted
        # candidates. extra_k_best_iterations lets us sample slightly deeper than
        # K to reduce scaled-objective boundary artifacts.
        top_k = TopKSolutionHeap(k=int(problem.max_results))
        solves_attempted = 0
        feasible_solutions_found = 0
        all_optimal = True
        max_results = int(problem.max_results)
        if max_results <= 0:
            early_stop_reason = "max_results_is_zero"
        else:
            early_stop_reason = "max_results_reached"
        solve_budget = max(0, max_results + self._extra_k_best_iterations)

        # Repeated exact solves:
        # 1) solve current model,
        # 2) decode selection,
        # 3) recompute exact float score under shared contract helper,
        # 4) add a no-good cut to exclude this exact selected set.
        if profiling_enabled:
            t_solve_loop_ns = wall_clock.perf_counter_ns()
        for _ in range(solve_budget):
            solves_attempted += 1
            if profiling_enabled:
                t_solve_call_ns = wall_clock.perf_counter_ns()
            status = solver.Solve(raw_model)
            if profiling_enabled:
                solve_call_ns = wall_clock.perf_counter_ns() - t_solve_call_ns
                solve_calls_total_ns += solve_call_ns
                solve_call_max_ns = max(solve_call_max_ns, solve_call_ns)
                status_key = _cp_sat_status_name(status)
                status_counts[status_key] = status_counts.get(status_key, 0) + 1
            if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                early_stop_reason = "infeasible_or_exhausted"
                break
            if status != cp_model.OPTIMAL:
                all_optimal = False

            selected_leaf_id_by_track_id: dict[int, int] = {}
            selected_solution_vars: list[cp_model.IntVar] = []

            if profiling_enabled:
                t_decode_selected_ns = wall_clock.perf_counter_ns()
            for leaf_var in leaf_vars:
                if solver.Value(leaf_var.variable) == 1:
                    selected_solution_vars.append(leaf_var.variable)
                    leaf = leaf_var.leaf_option
                    selected_leaf_id_by_track_id[int(leaf.track_id)] = int(leaf.leaf_id)
            if profiling_enabled:
                decode_selected_ns += (
                    wall_clock.perf_counter_ns() - t_decode_selected_ns
                )

            if profiling_enabled:
                t_exact_rescore_ns = wall_clock.perf_counter_ns()
            exact_score = score_selected_leaf_ids_if_exact_feasible(
                problem=problem,
                selected_leaf_id_by_track_id=selected_leaf_id_by_track_id,
                leaf_option_by_leaf_id=leaf_option_by_leaf_id,
            )
            if profiling_enabled:
                exact_rescore_ns += wall_clock.perf_counter_ns() - t_exact_rescore_ns
            if exact_score is None:
                raise RuntimeError(
                    "ORToolsClusterSolver produced an infeasible exact selection. "
                    f"selection={selected_leaf_id_by_track_id}"
                )

            feasible_solutions_found += 1
            if profiling_enabled:
                t_topk_push_ns = wall_clock.perf_counter_ns()
            top_k.push(
                candidate=ClusterSolverSolution(
                    selected_leaf_id_by_track_id=selected_leaf_id_by_track_id,
                    score=float(exact_score),
                ),
                insertion_order=feasible_solutions_found,
            )
            if profiling_enabled:
                topk_push_ns += wall_clock.perf_counter_ns() - t_topk_push_ns

            # Standard no-good cut: exclude this exact selected variable set.
            if not selected_solution_vars:
                early_stop_reason = "single_empty_solution"
                break
            if profiling_enabled:
                t_nogood_add_ns = wall_clock.perf_counter_ns()
            model.Add(sum(selected_solution_vars) <= len(selected_solution_vars) - 1)
            if profiling_enabled:
                nogood_add_ns += wall_clock.perf_counter_ns() - t_nogood_add_ns
        if profiling_enabled:
            solve_loop_total_ns += wall_clock.perf_counter_ns() - t_solve_loop_ns

        # Finalize top-K ordering by exact float score (with deterministic
        # insertion-order tie break) and summarize termination reason.
        if profiling_enabled:
            t_finalize_ns = wall_clock.perf_counter_ns()
        solutions_tuple = top_k.finalize()
        if profiling_enabled:
            finalize_ns += wall_clock.perf_counter_ns() - t_finalize_ns
        if not solutions_tuple and max_results > 0:
            early_stop_reason = "infeasible_or_exhausted"
        elif len(solutions_tuple) < max_results:
            early_stop_reason = "infeasible_or_exhausted"
        elif self._extra_k_best_iterations > 0:
            early_stop_reason = "solve_budget_reached"

        self._last_diagnostics = ClusterSolverDiagnostics(
            combinations_evaluated=int(solves_attempted),
            feasible_combinations=feasible_solutions_found,
            backend="ortools_cp_sat",
            optimal=all_optimal,
            solutions_returned=len(solutions_tuple),
            solves_attempted=int(solves_attempted),
            terminated_early=len(solutions_tuple) < max_results,
            early_stop_reason=early_stop_reason,
        )
        if profiling_enabled:
            total_ns = wall_clock.perf_counter_ns() - solve_start_ns
            total_ms = _ns_to_ms(total_ns)
            model_total_ns = (
                model_vars_and_keys_ns
                + model_exactly_one_ns
                + model_conflict_constraints_ns
                + model_objective_ns
            )
            maxrss_after_mb = _get_process_maxrss_mb()
            solve_call_mean_ns = (
                int(round(solve_calls_total_ns / solves_attempted))
                if solves_attempted > 0
                else 0
            )
            profile: dict[str, object] = {
                "tracks": int(len(problem.track_options)),
                "leaf_vars": int(len(leaf_vars)),
                "conflict_keys_unique": int(len(conflict_vars_by_key)),
                "conflict_constraints": int(conflict_constraints),
                "requested_k": int(problem.max_results),
                "solutions_returned": int(len(solutions_tuple)),
                "solves_attempted": int(solves_attempted),
                "status_counts": dict(sorted(status_counts.items())),
                "timing_ms": {
                    "total": total_ms,
                    "model_total": _ns_to_ms(model_total_ns),
                    "model_vars_and_key_map": _ns_to_ms(model_vars_and_keys_ns),
                    "model_exactly_one_constraints": _ns_to_ms(model_exactly_one_ns),
                    "model_conflict_constraints": _ns_to_ms(
                        model_conflict_constraints_ns
                    ),
                    "model_objective": _ns_to_ms(model_objective_ns),
                    "solve_loop_total": _ns_to_ms(solve_loop_total_ns),
                    "solve_calls_total": _ns_to_ms(solve_calls_total_ns),
                    "solve_call_mean": _ns_to_ms(solve_call_mean_ns),
                    "solve_call_max": _ns_to_ms(solve_call_max_ns),
                    "decode_selected_vars_total": _ns_to_ms(decode_selected_ns),
                    "exact_rescore_total": _ns_to_ms(exact_rescore_ns),
                    "topk_push_total": _ns_to_ms(topk_push_ns),
                    "nogood_add_total": _ns_to_ms(nogood_add_ns),
                    "finalize_total": _ns_to_ms(finalize_ns),
                },
                "memory_maxrss_mb": {
                    "maxrss_before": float(maxrss_before_mb),
                    "maxrss_after": float(maxrss_after_mb),
                    "maxrss_delta": float(maxrss_after_mb - maxrss_before_mb),
                },
            }
            self._last_profile = profile
            if self._profiling_print and total_ms >= self._profiling_min_total_ms:
                print(
                    "ORTOOLS_SOLVE_PROFILE "
                    + json.dumps(
                        profile,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
        return ClusterSolverResult(solutions=solutions_tuple)

    def get_last_diagnostics(self) -> ClusterSolverDiagnostics | None:
        return self._last_diagnostics

    def get_last_profile(self) -> dict[str, object] | None:
        """Return profiling metadata for the most recent solve when enabled."""
        return self._last_profile

    def _scale_score(self, score: float) -> int:
        if not math.isfinite(score):
            raise ValueError(
                "Cluster solver score must be finite for OR-Tools backend. "
                f"score={score}"
            )
        scaled = int(round(score * float(self._score_scale)))
        max_abs_coefficient = (1 << 63) - 1
        if abs(scaled) > max_abs_coefficient:
            raise ValueError(
                "Scaled score exceeds CP-SAT 64-bit coefficient range. "
                f"score={score} score_scale={self._score_scale}"
            )
        return scaled


def _cp_sat_status_name(status: int) -> str:
    if status == cp_model.OPTIMAL:
        return "OPTIMAL"
    if status == cp_model.FEASIBLE:
        return "FEASIBLE"
    if status == cp_model.INFEASIBLE:
        return "INFEASIBLE"
    if status == cp_model.MODEL_INVALID:
        return "MODEL_INVALID"
    if status == cp_model.UNKNOWN:
        return "UNKNOWN"
    return f"STATUS_{status}"
