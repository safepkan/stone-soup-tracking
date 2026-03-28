# TO-MHT Tracker Reference

This file combines the two earlier summaries:

- Core data structures
- Typical high-level update step

---

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


---

# TO-MHT Tracker: Typical High-Level Update Step

At a high level, a TO-MHT update for one detection batch usually looks like:

1. **predict everything to the batch time**
2. **gate detections against active hypothesis leaves**
3. **create child hypotheses for feasible associations**
4. **build consistent global hypotheses**
5. **score and prune**
6. **optionally confirm/delete/output tracks**

A typical flow is something like this.

---

## Inputs

```text
update(timestamp, detections[])
```

Assume all detections share the same time `t_k`.

Internal state before the call:

- active track trees
- active global hypotheses
- filter states sitting at the previous time
- tracker parameters like `P_D`, clutter rate, gating threshold, N-scan depth

---

## 1. Predict active hypothesis leaves to the new timestamp

For every active leaf node in every track tree:

- propagate the dynamic model from `t_{k-1}` to `t_k`
- update covariance with process noise
- produce predicted measurement quantities needed for gating

Conceptually:

```text
for each active leaf:
    x^- , P^- = motion_predict(x, P, t_k)
    z_hat, S = measurement_predict(x^-, P^-)
```

You usually only predict the **current leaves**, not every historical node.

Result: each active leaf now has a predicted state at the batch time.

---

## 2. Gate detections against predicted leaves

For each predicted leaf and each detection:

- compute innovation
- compute Mahalanobis distance or equivalent gating statistic
- keep only feasible detection-track pairs

This gives you a sparse set of candidate associations, not a dense all-to-all matrix.

```text
candidate_edges = []
for each leaf:
    for each detection:
        if inside_gate(leaf, detection):
            candidate_edges.add((leaf, detection, assoc_score))
```

Also include the implicit **missed-detection** option for each leaf.

Result: a per-leaf set of feasible child branches:
- detect with measurement `d_i`
- miss

---

## 3. Generate child hypotheses inside each track tree

For each active leaf, create child nodes for each allowed event.

Typical child types:

- **measurement update child** for each gated detection
- **missed detection child**
- maybe **termination child**
- maybe special handling for maneuver/model branching

For a matched detection:

```text
x^+, P^+ = measurement_update(x^-, P^-, z_i)
child.score = parent.score + log p(z_i | track) + log P_D + other terms
```

For a missed detection:

```text
x^+, P^+ = x^-, P^-   // or a missed-detection handling variant
child.score = parent.score + log(1 - P_D)
```

So each leaf fans out into multiple children.

Important point: this happens **locally per track tree first**. At this stage you have not yet enforced that the same detection cannot be used by two different tracks globally.

Result: every track tree gets a new frontier of child leaves at `t_k`.

---

## 4. Create birth hypotheses for unexplained detections

Detections that are not assigned to an existing track in a given global hypothesis may represent:

- clutter / false alarm
- new target birth

So for relevant detections, create tentative new-track roots or first nodes.

Conceptually:

```text
for each detection:
    create possible birth hypothesis
    create possible clutter explanation
```

In practice, these choices are usually folded into the later global association step rather than committed immediately.

---

## 5. Build global hypotheses from consistent combinations

Now comes the key TO-MHT step.

You have many local branches across many track trees. You need to form **globally consistent combinations** such that:

- each existing track selects at most one child branch
- each detection is used at most once
- unassigned detections are explained as clutter or births
- optional extra logic: exclusivity zones, class constraints, kinematic conflicts

This is often posed as a ranked assignment / multidimensional consistency problem.

Conceptually:

```text
for each prior global hypothesis G_prev:
    collect candidate child branches for the leaves selected by G_prev
    solve for best consistent combinations
    output K best successor global hypotheses
```

Each successor global hypothesis might specify:

- track A -> measurement 3
- track B -> miss
- track C -> measurement 0
- detection 1 -> birth
- detection 2 -> clutter

This is where methods like:

- assignment solvers
- Murty’s algorithm
- branch-and-bound
- network flow variants

often show up.

Result: a new pool of successor global hypotheses at time `t_k`.

---

## 6. Score the new global hypotheses

Each new global hypothesis gets a score or log weight from:

- parent global score
- track update likelihoods
- missed-detection penalties
- clutter model
- birth model
- optional existence probabilities / priors

Typical form:

```text
global_score =
    parent_global_score
  + sum(local_child_scores)
  + clutter_terms
  + birth_terms
  + normalization constants
```

Then normalize if you want posterior-like weights.

---

## 7. Prune aggressively

Without pruning, MHT explodes.

Common pruning steps:

- keep only top `K` global hypotheses
- prune low-scoring global hypotheses below threshold
- prune dominated local branches not used by any surviving global
- merge nearly identical states if your implementation supports it
- apply N-scan pruning / deferred decision logic

Typical logic:

```text
keep best K globals
delete unused child nodes
compress trees
```

This step is central to making the tracker practical.

---

## 8. Apply N-scan logic and resolve old ambiguity

If you use N-scan pruning:

- look back `N` scans
- find branch decisions that are now effectively common across the surviving best globals
- collapse older ambiguity
- promote stable prefixes to confirmed track history

So the tracker keeps ambiguity only in a moving recent window.

---

## 9. Update track status

Based on the surviving hypotheses, update lifecycle state:

- tentative -> confirmed
- confirmed -> coasted
- coasted too long -> deleted

You may also maintain per-track stats like:

- hit count
- consecutive misses
- track existence probability
- age
- confirmation score

---

## 10. Produce output tracks

Finally, derive the user-facing track set, usually from:

- the single best global hypothesis, or
- a weighted summary over several globals

Output each selected track’s current state and maybe its smoothed history.

---

# Pseudocode skeleton

Here is the whole step in a compact form:

```text
function update(t_k, detections):

    # 1. predict active leaves to t_k
    for leaf in active_leaves:
        leaf.pred_state = predict(leaf.state, t_k)
        leaf.pred_meas  = predict_measurement(leaf.pred_state)

    # 2. gate
    association_graph = build_gated_edges(active_leaves, detections)

    # 3. expand local track trees
    for leaf in active_leaves:
        create_missed_detection_child(leaf)
        for det in gated_detections(leaf):
            create_detection_child(leaf, det)

    # 4-5. for each prior global, form consistent successors
    new_globals = []
    for G in active_globals:
        local_choices = collect_child_choices_for_global(G)
        successors = generate_k_best_consistent_global_hypotheses(
            G, local_choices, detections
        )
        new_globals.extend(successors)

    # 6. score
    for G_new in new_globals:
        G_new.score = compute_global_score(G_new)

    # 7. prune
    active_globals = prune_global_hypotheses(new_globals)
    prune_unused_local_nodes(active_globals)

    # 8. N-scan resolution
    apply_n_scan_pruning(active_globals)

    # 9. lifecycle management
    update_track_statuses(active_globals)

    # 10. output
    return extract_output_tracks(active_globals)
```

---

# What is special about TO-MHT here?

The distinctive part is that the update is split into two levels:

## Local level
Each track tree branches into alternatives:
- matched to detection 1
- matched to detection 4
- missed
- etc.

## Global level
The tracker selects compatible combinations across trees:
- no double use of detections
- births/clutter handled consistently

That “local branching + global consistency selection” is the core pattern.

---

# A useful mental model

You can think of one update as:

- **predict** old leaves forward
- **branch** each track on all plausible explanations
- **pack** those branches into a small set of globally consistent worlds
- **prune** almost everything

---

# In practice, a typical implementation often organizes the update around these functions

```text
predict_to_time(t_k)
gate_detections(detections)
expand_track_trees(detections)
generate_global_hypotheses(detections)
score_hypotheses()
prune_hypotheses()
resolve_n_scan()
extract_tracks()
```

---

# Common variations

Depending on the tracker, the update may also include:

- IMM model mixing before/after prediction
- measurement partitioning / clustering before association
- separate initiator logic for births
- existence-probability updates
- track merging / duplicate suppression
- backward smoothing for committed history

---

# Minimal high-level summary

A typical TO-MHT update step is:

1. predict active leaf hypotheses to the detection time  
2. gate detections to form feasible associations  
3. create per-track child hypotheses for matches and misses  
4. combine those into globally consistent hypotheses  
5. score and prune  
6. commit older decisions and output tracks
