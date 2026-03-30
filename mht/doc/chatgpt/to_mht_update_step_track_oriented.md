# TO-MHT Tracker: Typical High-Level Update Step (Track-Oriented)

This version reflects the more typical **track-oriented MHT (TO-MHT)** view:

- what persists from one scan to the next is primarily the **track trees / track hypotheses**
- the **global hypotheses are typically rebuilt at each update** from the current surviving track hypotheses and their incompatibilities
- so the tracker does **not necessarily propagate the old explicit global hypotheses forward as first-class state**

At a high level, one update for a batch of detections at common time `t_k` often looks like this.

---

## Inputs

```text
update(timestamp, detections[])
```

Assume all detections share timestamp `t_k`.

Internal state before the call:

- active track trees
- current leaf hypotheses in those trees
- filter states at the previous time
- track scores / likelihood accumulators
- parameters such as detection probability, clutter rate, gating threshold, and N-scan depth

The key point is that the tracker mainly carries **track-level state and ambiguity** forward.

---

## 1. Predict active track-hypothesis leaves to the new timestamp

For each active leaf node in each track tree:

- propagate the dynamic model from the previous time to `t_k`
- update the covariance with process noise
- compute predicted measurement quantities needed for gating

Conceptually:

```text
for each active leaf:
    x^- , P^- = motion_predict(x, P, t_k)
    z_hat, S = measurement_predict(x^-, P^-)
```

Usually only the **current leaves** are predicted, not the entire historical tree.

Result: each active leaf now has a predicted state at the new scan time.

---

## 2. Gate detections against predicted leaves

For each predicted leaf and each detection:

- compute innovation
- compute Mahalanobis distance or another gating statistic
- keep only feasible leaf-detection pairs

This produces a sparse set of feasible associations.

```text
candidate_edges = []
for each leaf:
    for each detection:
        if inside_gate(leaf, detection):
            candidate_edges.add((leaf, detection, assoc_score))
```

Also include the implicit **missed-detection** option for each leaf.

Result: each leaf has a local menu of plausible child events:

- detection with `d_i`
- missed detection
- possibly termination
- possibly model-branching choices

---

## 3. Expand the track trees locally

For each active leaf, create child hypotheses for the feasible events.

Typical child types:

- one **measurement-update child** for each gated detection
- one **missed-detection child**
- maybe a termination child
- maybe maneuver/model children if the tracker branches on motion model too

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

This creates a new frontier of leaf hypotheses at `t_k`.

At this stage, the branching is still **local to each track tree**. The same detection may still appear in children belonging to different tracks. Global consistency has not yet been enforced.

---

## 4. Create possible birth tracks for unexplained detections

Some detections may represent:

- clutter / false alarms
- new target births

So the tracker may create tentative birth hypotheses from detections that are not obviously explained by existing tracks.

Conceptually:

```text
for each detection:
    create possible birth hypothesis
    create possible clutter explanation
```

Depending on the implementation, births may be generated eagerly for many detections or more selectively.

Result: the active candidate set now includes:

- updated continuations of existing tracks
- missed-detection continuations
- possible new tracks

---

## 5. Prune weak local branches

Before rebuilding global hypotheses, many implementations prune the track trees locally.

Typical local pruning:

- drop very low-score children
- cap the number of children per leaf
- delete branches with too many consecutive misses
- merge near-duplicates if the implementation supports it

Conceptually:

```text
for each track tree:
    prune_bad_children()
    keep_surviving_leaf_frontier()
```

This matters because the next step uses the surviving candidate track hypotheses as input.

---

## 6. Build compatibility / incompatibility relations among surviving track hypotheses

Now take the surviving candidate track hypotheses and determine which can coexist in the same global explanation.

Two track hypotheses are typically **incompatible** if they:

- use the same detection at the same scan
- descend from mutually exclusive alternatives in the same track tree
- violate application-specific constraints

Conceptually:

```text
for each pair of candidate track hypotheses:
    if share_measurement(...) or otherwise_conflict(...):
        mark_incompatible(i, j)
```

This often yields a conflict graph or compatibility structure.

This step is one of the defining features of TO-MHT:

- maintain ambiguity primarily in **track trees**
- rebuild the admissible **global combinations** from the current track set

---

## 7. Rebuild global hypotheses from the current track set

Now form one or more best **globally consistent sets** of track hypotheses.

A global hypothesis is now built from the current surviving track candidates, subject to:

- at most one active branch from a mutually exclusive family
- no double use of the same detection
- suitable treatment of clutter and births

Conceptually:

```text
candidate_tracks = surviving_track_hypotheses()
conflicts = incompatibility_structure(candidate_tracks)
new_globals = solve_for_best_compatible_sets(candidate_tracks, conflicts)
```

This is the step where the tracker may compute:

- the single best global hypothesis
- the top `K` global hypotheses
- a weighted set of feasible global explanations

Methods here may include:

- ranked assignment
- maximum-weight independent-set style formulations
- branch-and-bound
- Murty-style ranking methods in some formulations

The important correction relative to the earlier summary is:

**these globals are often rebuilt from the current tracks, not necessarily propagated directly from last scan's explicit global list.**

---

## 8. Score the rebuilt global hypotheses

Each rebuilt global hypothesis gets a score from the scores of its member track hypotheses plus clutter/birth terms and any normalizing constants.

Conceptually:

```text
global_score =
    sum(track_hypothesis_scores)
  + clutter_terms
  + birth_terms
  + normalization_constants
```

Because the track hypotheses already carry accumulated history-dependent scores, the global score can often be assembled from the current selected tracks without explicitly carrying last scan's global hypothesis objects forward.

---

## 9. Apply global pruning and N-scan resolution

After rebuilding the globals:

- keep only the top `K` globals or those above a threshold
- remove track branches not used by any surviving global if desired
- apply N-scan pruning / deferred decision logic

Typical logic:

```text
keep best K globals
remove unsupported branches
collapse resolved history older than N scans
```

If using N-scan pruning:

- look back `N` scans
- find decisions that have effectively become common across the surviving best globals
- commit those older branches
- discard older unresolved alternatives

This keeps ambiguity confined to a moving recent window.

---

## 10. Update lifecycle state and produce output tracks

Based on the surviving global hypotheses, update track status:

- tentative -> confirmed
- confirmed -> coasted
- coasted too long -> deleted

Then generate user-facing output from:

- the single best global hypothesis, or
- a weighted combination / summary of several globals

Output may include:

- current state estimate
- covariance
- track history
- status flags

---

# Pseudocode skeleton

A compact track-oriented sketch might look like this:

```text
function update(t_k, detections):

    # 1. predict existing leaf hypotheses
    for leaf in active_leaves:
        leaf.pred_state = predict(leaf.state, t_k)
        leaf.pred_meas  = predict_measurement(leaf.pred_state)

    # 2. gate detections
    association_graph = build_gated_edges(active_leaves, detections)

    # 3. expand track trees locally
    for leaf in active_leaves:
        create_missed_detection_child(leaf)
        for det in gated_detections(leaf):
            create_detection_child(leaf, det)

    # 4. propose births
    birth_hypotheses = create_birth_hypotheses(detections)

    # 5. prune locally
    prune_track_trees()

    # 6. build conflicts among surviving track hypotheses
    candidate_tracks = collect_surviving_track_hypotheses()
    conflicts = build_incompatibility_structure(candidate_tracks)

    # 7. rebuild global hypotheses from current tracks
    new_globals = generate_best_compatible_global_sets(
        candidate_tracks,
        conflicts,
        birth_hypotheses
    )

    # 8. score globals
    for G in new_globals:
        G.score = compute_global_score(G)

    # 9. prune / resolve history
    active_globals = prune_global_hypotheses(new_globals)
    apply_n_scan_pruning(active_globals, track_trees)

    # 10. lifecycle and output
    update_track_statuses(active_globals)
    return extract_output_tracks(active_globals)
```

---

# What is special about TO-MHT here?

The distinctive pattern is:

## Local level
Each track tree stores alternative histories and branches forward with feasible local explanations.

## Global level
At each scan, the tracker forms compatible global explanations from the currently surviving track hypotheses.

So a good mental summary is:

- **carry track trees forward**
- **rebuild global hypotheses from the current surviving track set**
- **prune aggressively**

That is a better high-level description of many TO-MHT implementations than saying the tracker explicitly propagates last scan's global hypotheses as the main persistent state.

---

# Relation to the earlier summary

The earlier summary described a more explicit “for each old global hypothesis, generate successor globals” flow.

That is useful conceptually and is closer to classical hypothesis-oriented MHT descriptions, but for **track-oriented MHT** the more typical picture is:

- persist **track trees / track hypotheses**
- use their accumulated scores and incompatibilities
- reconstruct the current best global hypotheses after each update

---

# Minimal high-level summary

A typical track-oriented TO-MHT update step is:

1. predict active leaf hypotheses to the new detection time  
2. gate detections to form feasible local associations  
3. create per-track child hypotheses for matches, misses, and births  
4. prune weak local branches  
5. rebuild globally consistent hypotheses from the surviving track set  
6. score, prune, resolve older ambiguity, and output tracks
