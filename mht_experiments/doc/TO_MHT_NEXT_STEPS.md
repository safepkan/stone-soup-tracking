# TO-MHT Next Steps

## Next phase

**Phase B: explicit track-hypothesis structure and true N-scan pruning**

This phase replaces the old startup/birth-cleanup plan. The external-initiation phase is considered complete enough, and the next priority is to make the tracker structurally match a proper track-oriented MHT.

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

## Migration strategy

This phase should be staged rather than attempted as one giant rewrite.

### Task 1 — design sketch and internal representation choice

Before coding heavily, lock down the representation in code-facing terms:
- node structure,
- global structure,
- node ownership of cached maintenance metadata,
- how temporary `Track` reconstruction works,
- what data remains tracker-owned vs reconstructed,
- the exact meaning of N-scan commitment in this implementation,
- the separation between commitment and physical node cleanup,
- the intended replacement for history-tail-based deduplication.

Deliverable:
- updated design notes in this document and/or code comments sufficient to guide implementation without relying on the current copied-track representation.

### Task 2 — introduce explicit node-based track representation

Refactor the tracker so globals point to leaf nodes rather than copied `Track` objects.

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