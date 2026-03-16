# TO-MHT Next Steps

## Next phase

**Phase B: explicit track-hypothesis structure and true N-scan pruning**

This phase replaces the old startup/birth-cleanup plan. The external-initiation phase is considered complete enough, and the next priority is to make the tracker structurally match a proper track-oriented MHT.

## Why this phase is next

The current implementation is usable and reasonably clean, but its core representation is still wrong for a true TO-MHT:
- global hypotheses hold copied track objects,
- ancestry is implicit,
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

### Node semantics

A node should represent one step of one logical track hypothesis, not a fully copied long-history `Track` as its primary representation.

Expected contents of a node include:
- node identity,
- logical `track_id`,
- parent pointer,
- timestamp,
- current state payload,
- association info for that step,
- maintenance metadata such as age/hits/missed_count/last-hit bookkeeping.

### Global semantics

A global hypothesis should remain simple:
- log weight,
- mapping from logical `track_id` to current leaf node.

This preserves the current “one active hypothesis per logical track within a global” idea while making history sharing explicit.

### Transitional compatibility

It is acceptable during this phase to reconstruct temporary Stone Soup `Track` objects from node ancestry when needed for:
- updater/hypothesiser compatibility,
- output,
- visualisation,
- existing debugging helpers.

The priority in this phase is structural correctness and clarity, not immediate performance optimisation.

## True N-scan pruning for this tracker

For this phase, true N-scan means:

- after processing scan `k` and applying beam pruning,
- inspect surviving global hypotheses at the boundary `k - N`,
- use explicit ancestor identity rather than recent association-history tails,
- commit/prune older branch distinctions once they are no longer represented among surviving globals.

Important design choice:
- N-scan commitment should be based on surviving globals **after beam pruning**.

For the first implementation, it is acceptable to separate:
1. explicit ancestor/commitment logic,
2. physical cleanup/garbage collection of orphaned nodes.

The first priority is to make ancestry and commitment semantics explicit and correct.

## Migration strategy

This phase should be staged rather than attempted as one giant rewrite.

### Task 1 — design sketch and internal representation choice

Before coding heavily, lock down the representation in code-facing terms:
- node structure,
- global structure,
- how temporary `Track` reconstruction works,
- what data remains tracker-owned vs reconstructed.

Deliverable:
- updated design notes in this document and/or code comments sufficient to guide implementation.

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
- the resulting implementation is easier to explain as an actual TO-MHT in architecture discussions.

## Things explicitly out of scope for this phase

To keep the phase focused, do **not** broaden scope into:
- richer external-start scheduling,
- pre-first-step external starts,
- scoring redesign,
- principled existence modeling,
- deep performance optimisation.

Those remain important, but they should come after the structure is corrected.

## Notes for later phases

Once this phase is complete, the next choice will likely be between:
- scoring refinement, especially if the ISAC workflow is primarily external-start based,
- or deeper birth/existence cleanup, if internal-birth behavior remains important.
