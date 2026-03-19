# TO-MHT Next Steps

## Next phase

**Phase B: explicit track-hypothesis structure and true N-scan pruning**

This phase replaces the old startup/birth-cleanup plan. The external-initiation phase is considered complete enough, and the next priority is to make the tracker structurally match a proper track-oriented MHT.

### Implementation status (2026-03-19)

- Task 2 / Step A representation groundwork is now in code:
  - explicit `TrackHypothesisNode` objects exist and are tracker-owned,
  - globals now store `track_id -> leaf node`,
  - a temporary reconstruction adapter keeps Stone Soup `Track` compatibility at hypothesiser/updater/output boundaries.
- Task 3 / Steps D-F operational migration is now in code:
  - per-track continuation now creates child nodes directly from parent leaves (with Track reconstruction only at compatibility boundaries),
  - global expansion and detection-usage checks operate over node fields,
  - dedupe now uses structural leaf identity (`track_id -> node_id`) rather than history tails,
  - external starts and internal births both create root-like nodes via the same structural helper while keeping distinct provenance/scoring semantics.
- Task 4 / Step G ancestor-based N-scan commitment is now in code:
  - commitment runs after beam pruning and before births using boundary `b = k - N`,
  - per-track agreement checks use explicit ancestor node identity at scan `b`,
  - only globals that still contain a `track_id` participate in that track’s agreement check,
  - tracks without an exact-boundary ancestor (for example born after `b`) are conservatively left uncommitted,
  - commitment bookkeeping is explicit, while physical node cleanup/GC remains deferred.

## Why this phase is next

The current implementation is usable and reasonably clean, but its core representation is still not correct for a true TO-MHT:
- global hypotheses hold copied track objects,
- ancestry is implicit rather than structural,
- current N-scan behavior is only a history-tail approximation.

That is now the main blocker to calling the tracker a proper TO-MHT and to having clearer architecture discussions with the ISAC group.

## Design stance for this phase

### Core representation

Move to an explicit per-track hypothesis-node representation.

Recommended model:
- each logical track has a chain/tree of hypothesis nodes,
- each node points to its parent node for the same logical track,
- a global hypothesis maps `track_id -> current leaf node`.

This keeps `track_id` as the logical track identity while introducing explicit branch ancestry through node identity.

### Phase B invariants

The following should hold throughout this phase:

- `track_id` identifies a logical track across branching alternatives.
- `node_id` identifies one specific hypothesis node for one logical track at one step.
- A hypothesis node has at most one parent pointer.
- A node’s parent pointer always refers to the same logical `track_id`.
- A global hypothesis maps `track_id -> current leaf node`.
- Global hypotheses do not use copied full-history `Track` objects as their primary internal representation.
- Shared ancestry is represented by shared node identity, not by copied history content.
- Commitment and physical memory cleanup are separate concerns.

### Node semantics

A node should represent one step of one logical track hypothesis, not a fully copied long-history `Track` as its primary representation.

Expected contents of a node include:
- node identity,
- logical `track_id`,
- parent pointer,
- timestamp,
- current state payload,
- association info for that step,
- cached maintenance metadata needed for common tracker logic and inspection.

Expected cached maintenance metadata may include:
- age,
- hits,
- missed_count,
- last-hit bookkeeping,
- other small fields that would otherwise require frequent ancestry walking.

The intent is to cache small, operationally useful metadata while keeping the node as a one-step hypothesis representation rather than a disguised copied full track.

### Global semantics

A global hypothesis should remain simple:
- log weight,
- mapping from logical `track_id` to current leaf node.

This preserves the current “one active hypothesis per logical track within a global” idea while making history sharing explicit.

### External starts and births

External starts and internal births should be represented by the same node structure, while remaining semantically distinct in provenance and scoring.

In particular:
- an external start is a confirmed root-like node introduced through the external-start path,
- an internal birth is a root-like node introduced through the internal-birth path.

They should share the same structural representation once in the hypothesis graph, without collapsing their distinct tracker semantics.

### Transitional compatibility

It is acceptable during this phase to reconstruct temporary Stone Soup `Track` objects from node ancestry when needed for:
- updater/hypothesiser compatibility,
- output,
- visualisation,
- existing debugging helpers.

This reconstruction should be treated as an adapter boundary only.

Internal branch identity, deduplication, and N-scan commitment should not rely on reconstructed `Track` object identity or copied track history.

The priority in this phase is structural correctness and clarity, not immediate performance optimisation.

## True N-scan pruning for this tracker

For this phase, true N-scan means:

- after processing scan `k` and applying beam pruning,
- inspect surviving global hypotheses at the boundary `k - N`,
- use explicit ancestor identity rather than recent association-history tails,
- commit older branch distinctions once they are no longer represented among surviving globals.

Important design choice:
- N-scan commitment should be based on surviving globals **after beam pruning**.

### Commitment semantics

For a scan `k` and N-scan window `N`, after beam pruning at scan `k`, consider the surviving global hypotheses and inspect each logical track’s ancestor node at boundary `k - N`.

For a given logical track, if all surviving globals that still contain that track agree on the same ancestor node at that boundary, then branch distinctions earlier than that boundary are considered committed for that track.

This commitment decision is based on explicit ancestor node identity, not on association-history tails.

For the first implementation, it is acceptable to separate:
1. explicit ancestor and commitment logic,
2. physical cleanup / garbage collection of orphaned or no-longer-needed nodes.

The first priority is to make ancestry and commitment semantics explicit and correct.

### What this phase does not require

This phase does **not** require a separate committed-track store or any broader committed-history materialisation design.

It is sufficient to make ancestry explicit and make commitment semantics correct. Any later design for committed outputs, committed prefixes, or detached committed history can follow after this phase.

## Task 1 design baseline

The following section records the intended code-facing design baseline for Task 1.

It is meant to be specific enough to guide implementation while still allowing small adjustments if the code reveals friction during the migration.

### Internal object model

Use three internal levels:

#### Track hypothesis node

A `TrackHypothesisNode` is the canonical internal unit of branching.

One node represents one logical-track hypothesis at one scan step.

Expected contents of a node include:
- `node_id`,
- logical `track_id`,
- parent pointer,
- `scan_index`,
- timestamp,
- per-step state payload,
- per-step association label / used detection identity,
- incremental log contribution for that step,
- cached maintenance metadata,
- root provenance.

Recommended cached maintenance metadata includes:
- `age`,
- `hits`,
- `missed_count`,
- `last_det_key`,
- `last_det_hit`,
- optionally small additional inspection fields such as birth/first-scan bookkeeping.

Recommended provenance fields include:
- `root_source`, with values such as `external_start` or `internal_birth`,
- `birth_scan_index`.

Important invariants:
- a node’s parent pointer always refers to the same logical `track_id`,
- a node stores one step of one hypothesis, not a copied long-history `Track`,
- recent association history may be derived when needed, but is not the primary internal representation.

#### Global hypothesis

A `GlobalHypothesis` remains intentionally simple:
- log weight,
- mapping `track_id -> current leaf node`.

This is the direct structural replacement for the current `track_id -> copied Track` representation.

#### Tracker-owned node registry

The tracker should own the node graph explicitly.

Recommended tracker-owned state includes:
- a monotonic `node_id` allocator,
- a registry such as `node_id -> node`,
- any future commitment / cleanup bookkeeping.

Globals point only to leaf nodes; the tracker owns the full ancestry graph.

### Tracker-owned truth vs reconstructed views

The tracker-owned internal truth should be:
- node graph,
- global hypotheses,
- node metadata,
- per-node association decisions,
- node provenance,
- global log weights.

Temporary Stone Soup `Track` objects should be treated as reconstructed adapter views only.

They may be rebuilt from node ancestry for:
- hypothesiser compatibility,
- updater compatibility,
- public outputs,
- visualisation,
- debugging helpers.

Reconstructed `Track` objects should not define branch identity, deduplication, or N-scan commitment semantics.

### Reconstruction boundary

Task 1 should assume one explicit reconstruction boundary from leaf node to temporary `Track`.

Intended behavior:
- walk ancestry from leaf to root,
- rebuild a chronological temporary `Track`,
- project selected metadata fields needed by existing adapters and outputs.

This reconstruction boundary should be the main compatibility bridge during migration.

### Node creation rules

Every structural update to a logical track should occur by creating exactly one new node.

The design should support four main cases:
- continuation with detection hit,
- continuation with miss,
- external-start root node,
- internal-birth root node.

Expected metadata evolution:
- hit continuation increments `age` and `hits`, resets `missed_count`, and records the used detection,
- miss continuation increments `age`, increments `missed_count`, and records a miss association,
- external-start and internal-birth roots have `parent = None` but retain distinct provenance and scoring semantics.

The structural representation should therefore be shared, while semantics such as scoring and insertion path remain distinct.

### Global expansion semantics

The current global-expansion shape should remain as stable as practical:
- each resulting global contains at most one active leaf per logical `track_id`,
- no two leaves in the same global may claim the same detection,
- a track that exceeds miss limits may be omitted from the resulting global.

The main change in this phase is therefore the payload being branched and stored: leaf nodes instead of copied tracks.

### Dedupe baseline

Task 1 should replace the current history-tail-based deduplication concept with explicit leaf identity.

Baseline intended rule:
- two globals are duplicates only if they contain the same active leaf node for every active `track_id`.

In other words, dedupe should be based on structural hypothesis identity rather than copied recent history content.

This may initially deduplicate less aggressively than the current approximation, but it is the cleaner structural definition.

Any stronger equivalence rules after commitment should be treated as later refinements, not mixed into the basic meaning of a global.

### True N-scan in code-facing terms

Task 1 should treat true N-scan as an ancestor-identity question.

For scan `k` and N-scan depth `N`:
- use surviving globals after beam pruning,
- inspect each active logical track’s ancestor node at boundary `k - N`,
- compare ancestor node identity explicitly.

A logical track is committed through that boundary when all surviving globals that still contain that track agree on the same ancestor node at the boundary.

This is a per-track commitment rule, not a requirement that whole globals agree everywhere.

Including `scan_index` directly on nodes is recommended so that boundary queries are explicit and not inferred indirectly from timestamps or history length.

### Commitment vs cleanup

Task 1 should explicitly separate:
1. commitment semantics,
2. physical cleanup / garbage collection.

Commitment means older branch distinctions are no longer represented among surviving globals at the relevant boundary.

Cleanup means old or orphaned nodes may later be deleted or compacted.

Correctness should depend on explicit ancestry and commitment logic, not on immediate node deletion.

### Public behavior during migration

Public behavior should remain as stable as practical during this phase.

In particular:
- the internal truth moves to nodes and leaf-based globals,
- public-facing outputs may still be temporary reconstructed `Track` objects,
- runner and debugging workflows should continue to work through that adapter layer while the internal representation changes.


## Implementation-oriented migration outline

This section sits between the design baseline and the task list on purpose.

The Task 1 design baseline defines the intended representation and semantics.
The following outline maps that baseline onto the current tracker structure so that implementation can proceed in a controlled order without turning the task list itself into a wall of low-level detail.

### Why keep this as a separate section

A separate implementation-oriented section keeps two levels distinct:
- the design baseline remains the source of truth for representation and semantics,
- the migration strategy remains a readable phase/task summary,
- this section records how the current code is expected to move from one to the other.

This also makes it easier to refine implementation sequencing later without rewriting the higher-level task descriptions.

### Current code seams that the migration should follow

The current tracker already exposes the main seams that the migration should attach to:
- `GlobalHypothesis` currently stores `track_id -> copied Track`,
- `ChildCandidate` currently carries a copied child `Track`,
- `_candidates_for_track(...)` currently reconstructs branch alternatives by copying a full `Track` and appending one more state,
- `_expand_global_hypothesis(...)` currently performs the cross-track consistency search over those copied child tracks,
- detection-usage helpers currently read `last_det_key` from `Track.metadata`,
- external-start insertion and internal-birth insertion already enter the tracker through distinct paths and should keep those semantics,
- MAP/debug/output code currently expects temporary `Track` objects and can initially continue to do so through reconstruction.

The migration should therefore aim to replace the payload being stored and branched, while preserving the overall control-flow shape as much as practical.

### Recommended implementation order

The following order is recommended even if some edits overlap in practice.

#### Step A — add the new internal types without switching behavior yet

Introduce the new structural types first:
- `TrackHypothesisNode`,
- updated `ChildCandidate` carrying `child_node`,
- updated `GlobalHypothesis` carrying `leaves_by_track_id`,
- tracker-owned node registry / node-id allocator.

At this point, temporary compatibility code may still allow parts of the tracker to operate on reconstructed `Track` views.

#### Step B — add the reconstruction adapter and make it explicit

Create one clear helper that rebuilds a temporary Stone Soup `Track` from a leaf node ancestry chain.

This helper should become the main compatibility bridge for:
- hypothesiser input,
- updater input,
- MAP/public output,
- debug display.

The intent is that copied `Track` objects stop being internal truth even before every downstream helper is fully migrated.

#### Step C — switch globals from copied tracks to leaf nodes

Once node creation and reconstruction exist, move the core global representation from:
- `track_id -> copied Track`

to:
- `track_id -> current leaf node`.

This is the key representational change. It should happen as early as practical once the compatibility bridge exists.

#### Step D — migrate per-track branching to node creation

Refactor the current continuation logic so that per-track branching creates nodes rather than copied tracks.

In practice this means replacing the current “copy track, append state, mutate metadata” pattern with:
- reconstruct temporary `Track` view if needed for hypothesiser/updater interaction,
- create one new child node for hit or miss continuation,
- store cached maintenance metadata on the node.

This is the point where the current `_candidates_for_track(...)` logic changes most substantially.

#### Step E — migrate global expansion, detection-usage helpers, and dedupe

Once candidates carry nodes, update the global expansion logic so that:
- cross-track consistency is enforced over leaf nodes,
- miss-drop logic uses node metadata rather than `Track.metadata`,
- detection-usage helpers read from nodes rather than tracks,
- dedupe is based on active leaf-node identity rather than history tails.

This step is where a large amount of copied-track bookkeeping should disappear.

#### Step F — migrate births and external starts into the same node structure

After normal continuation/miss branching works with nodes, convert both root-creation paths:
- external starts create root-like nodes with external-start provenance,
- internal births create root-like nodes with internal-birth provenance.

The structure should be shared, while insertion path and scoring semantics remain distinct.

#### Step G — replace N-scan-lite with explicit ancestor-based commitment

Only after the node graph is real should the tracker replace the history-tail approximation.

This step should:
- inspect ancestor identity at boundary `k - N`,
- make per-track commitment decisions,
- keep commitment semantics separate from physical cleanup,
- avoid coupling correctness to immediate node deletion.

#### Step H — cleanup, instrumentation, and documentation pass

Once the new structure and commitment logic are working:
- remove or simplify leftover copied-track helpers,
- update debug output so it talks about nodes / ancestry where appropriate,
- keep public outputs stable where useful via reconstruction,
- update `CURRENT_STATE`, `NEXT_STEPS`, and `ROADMAP`.

### How this maps onto Tasks 2–5

The phase-level tasks still make sense, but the code-level work is a little more fine-grained:
- **Task 2** mainly covers Steps A–C,
- **Task 3** mainly covers Steps D–F,
- **Task 4** mainly covers Step G,
- **Task 5** mainly covers Step H.

So the current task list is still reasonable, but actual implementation work will likely be tracked in smaller subtasks or checkpoints inside those larger tasks.

### Expected areas of overlap

Some overlap between Tasks 2 and 3 is expected. In the current code, representation and branching are tightly coupled, so it may not be possible to complete all of Task 2 before touching parts of Task 3.

That is acceptable as long as the migration keeps the following discipline:
- representation changes are driven by the Task 1 baseline,
- reconstructed `Track` objects remain an adapter layer rather than regaining internal-truth status,
- external-start and internal-birth semantics are preserved while their structural representation changes.

### Working style during implementation

When actual coding begins, it is reasonable to break Tasks 2 and 3 into implementation-sized subtasks or patches.

A good default is to prefer a few coherent updates rather than a long series of tiny edits, but not so large that it becomes hard to verify behavior or reason about breakage.

## Migration strategy

This phase should be staged rather than attempted as one giant rewrite.

### Task 1 — design sketch and internal representation choice

Use the “Task 1 design baseline” section above as the intended representation and semantics baseline for implementation.

Before coding heavily, confirm that the implementation will follow that baseline in code-facing terms:
- node structure,
- global structure,
- node ownership of cached maintenance metadata,
- reconstruction boundary for temporary `Track` objects,
- tracker-owned vs reconstructed data,
- explicit ancestor-based N-scan commitment semantics,
- separation between commitment and physical cleanup,
- replacement of history-tail-based deduplication with node-identity-based deduplication.

Deliverable:
- updated design notes in this document and/or code comments sufficient to guide implementation without relying on the current copied-track representation.

### Task 2 — introduce explicit node-based track representation

Refactor the tracker so globals point to leaf nodes rather than copied `Track` objects.

Primary emphasis in this task:
- Step A: add the new internal node/global/candidate types and tracker-owned node registry,
- Step B: add and standardise the reconstruction adapter from leaf node to temporary `Track`,
- Step C: switch globals from copied tracks to leaf nodes as the primary internal representation.

Constraints:
- keep overall tracker behavior as stable as practical,
- keep public APIs stable where possible,
- keep external-start and internal-birth semantics unchanged.

Expected result:
- explicit ancestry exists,
- track-history sharing becomes structural,
- the tracker still produces the same kind of outputs via temporary reconstruction where necessary.

### Task 3 — adapt update / branching flow to the node representation

Make sure normal continuation, miss handling, births, and external starts all create/update nodes consistently.

Primary emphasis in this task:
- Step D: migrate per-track continuation branching from copied-track mutation to explicit node creation,
- Step E: migrate global expansion, detection-usage helpers, and dedupe to operate on nodes/leaves,
- Step F: migrate internal births and external starts into the same shared node structure while preserving distinct semantics.

This is where the transitional reconstruction layer may be needed most heavily.

Expected result:
- one clear path for extending a logical track hypothesis by one step,
- births and external starts become root-like nodes in the same structural system,
- globals still behave as before from the outside.

### Task 4 — replace N-scan-lite with explicit ancestor-based N-scan

Remove the current history-tail approximation and introduce real ancestor-based commitment/pruning logic.

Expected result:
- commitment semantics are explicit,
- code refers to ancestor identity rather than association-history heuristics,
- beam-pruned surviving globals determine what gets committed.

### Task 5 — cleanup, instrumentation, and docs

Once the structure and N-scan logic are in place:
- update debug/instrumentation to describe node/ancestor behavior clearly,
- simplify or remove no-longer-relevant copied-track logic,
- update `CURRENT_STATE`, `NEXT_STEPS`, and `ROADMAP` accordingly.

## Acceptance criteria for the phase

This phase should be considered complete when:

- global hypotheses no longer store copied full-track objects as the primary representation,
- explicit parent-linked per-track hypothesis structure exists,
- the tracker can still run the existing scenarios with comparable external behavior,
- external starts and internal births both work within the new structure,
- true ancestor-based N-scan commitment replaces the current approximation,
- shared ancestry is represented structurally rather than by copied history,
- the resulting implementation is easier to explain as an actual TO-MHT in architecture discussions.

## Things explicitly out of scope for this phase

To keep the phase focused, do **not** broaden scope into:
- richer external-start scheduling,
- pre-first-step external starts,
- scoring redesign,
- principled existence modeling,
- committed-history output design,
- deep performance optimisation.

Those remain important, but they should come after the structure is corrected.

## Notes for later phases

Once this phase is complete, the next choice will likely be between:
- scoring refinement, especially if the ISAC workflow is primarily external-start based,
- or deeper birth/existence cleanup, if internal-birth behavior remains important.

A later phase may also decide whether committed ancestry should be materialised into a separate committed-history or output structure, but that is intentionally deferred here.
