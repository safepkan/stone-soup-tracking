# TO-MHT Tracker: Core Data Structures

A TO-MHT tracker usually revolves around a few nested structures that represent measurements, local track hypotheses, and global hypothesis combinations.

At a high level, the core data structures often look like this:

## 1. Measurement

Represents one sensor return at one scan.

```text
Measurement
- id
- time
- z                 // observation vector
- R                 // measurement covariance
- sensor_id
- attributes         // amplitude, class, etc.
```

You usually keep measurements grouped by scan:

```text
Scan
- time
- measurements[]
```

## 2. Track state / filter state

Each hypothesis about a target needs a dynamic state estimate.

```text
TrackState
- x                 // state mean
- P                 // covariance
- model_id          // motion model
- existence_prob    // optional
```

This is often just the state carried by a Kalman filter, EKF, UKF, IMM, etc.

## 3. Track-hypothesis node

This is the most important object in TO-MHT.

A track in TO-MHT is not a single thing. It is a tree of alternative histories for one putative target. Each node corresponds to one association decision at one scan.

```text
TrackHypothesisNode
- node_id
- parent_id
- children_ids[]
- track_id
- scan_index
- state             // TrackState after update or prediction
- assoc_type        // DETECTION, MISSED_DETECTION, BIRTH, TERMINATION
- measurement_id    // null for miss
- log_likelihood
- cumulative_score
- is_leaf
- is_pruned
```

This naturally forms a per-track hypothesis tree.

## 4. Track tree

All competing histories for one target.

```text
TrackTree
- track_id
- root_node_id
- leaf_node_ids[]
- metadata
```

The root may correspond to a birth hypothesis, tentative track seed, or a dummy origin.

## 5. Global hypothesis

TO-MHT differs from simpler trackers because it keeps multiple consistent combinations of track branches across all tracks.

A global hypothesis is typically a set of one selected leaf per track tree, plus clutter/birth choices, such that no measurement is used twice.

```text
GlobalHypothesis
- global_id
- selected_leaf_by_track   // map: track_id -> node_id
- used_measurement_ids
- log_weight
- normalized_weight
- parent_global_id         // optional, for hypothesis tree over scans
- is_feasible
```

This is the “joint explanation” of the current world.

## 6. Hypothesis pool / forest

You usually maintain many global hypotheses.

```text
HypothesisPool
- globals[]                // active global hypotheses
- best_global_id
- k_best_cache             // optional
```

In practice this is often managed with priority queues, Murty’s algorithm outputs, or ranked assignment solutions.

## 7. Association candidates / validation matrix

Before building hypotheses, you need feasible measurement-to-track associations.

```text
AssociationCandidate
- track_leaf_id
- measurement_id           // null allowed for miss
- innovation
- S
- gated
- log_likelihood
```

And often a bipartite representation:

```text
AssociationGraph
- track_leaf_ids[]
- measurement_ids[]
- edges[]                  // feasible assignments only
```

This is what gating produces before combinatorial hypothesis generation.

## 8. Measurement usage index

Because a measurement cannot usually support two target updates in the same global hypothesis, you want fast checks.

```text
MeasurementUsageIndex
- measurement_id -> set of global_id or node_id
```

Or simply bitsets for speed.

## 9. N-scan backpointer structure

TO-MHT often resolves ambiguity with delayed decisions, then prunes after N scans.

```text
HistoryWindow
- scan_index
- global_hypotheses[]
- resolved_prefix_depth
```

Backpointers let you trace from a current leaf back through its ancestors to reconstruct the trajectory.

## 10. Confirmed track output

Internal hypotheses are messy; output tracks are usually a separate clean structure.

```text
TrackOutput
- public_track_id
- history[]               // times, states, covariances
- status                  // tentative, confirmed, deleted
- source_hypothesis_id
```

## A very common in-memory shape

A practical implementation often ends up as:

- `measurements_by_scan`
- `track_trees`
- `global_hypotheses`
- `node_store` as a flat arena or hash map
- `parent/child backpointers`
- `bitsets` for measurement exclusivity
- `priority queues` for best-first global hypothesis management

For example:

```text
Tracker
- current_scan
- scans[]
- node_store: node_id -> TrackHypothesisNode
- tracks: track_id -> TrackTree
- globals: global_id -> GlobalHypothesis
- active_global_ids[]
- parameters
    - gate_threshold
    - p_det
    - clutter_rate
    - n_scan
    - prune_limits
```

## Why these structures fit TO-MHT

TO-MHT is “track-oriented” because:

- ambiguity is stored inside each track as a tree of alternative histories
- consistency across tracks is handled by separate global hypotheses
- pruning and N-scan resolution operate on those trees and global combinations

So the essential pattern is:

**track tree + global selector**

rather than just a single flat list of tracks.

## Performance-oriented choices

In a real system, these details matter a lot:

- store nodes in arrays/arenas, not pointer-heavy objects, for cache efficiency
- use integer IDs instead of raw pointers
- represent measurement usage with bitsets
- keep only leaf nodes active for assignment at the current scan
- compress shared history through parent pointers rather than copying trajectories

## Minimal conceptual schema

If you wanted the smallest useful TO-MHT design, it would be:

```text
Measurement
TrackHypothesisNode
TrackTree
GlobalHypothesis
Tracker
```

with `TrackHypothesisNode.parent_id` for history and `GlobalHypothesis.selected_leaf_by_track` for joint consistency.
