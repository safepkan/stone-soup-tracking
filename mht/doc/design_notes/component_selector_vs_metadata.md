> **Design note — 2026-09-01.** Prepared for the ISAC GitLab draft-MR discussion on
> per-track IMM profile switching: `component_selector` (their draft integration) vs.
> the TOMHT caller-metadata path. Multi-model review: Claude (Opus, Fable) and Codex.
> Decision pending on the ISAC side; the §4/§11 API-guide updates happen regardless.
> Drop this block when pasting into the discussion thread.

# Per-track IMM profile switching: `component_selector` vs. the metadata path

## Summary

Both designs can deliver on-the-fly A→B profile switching. The recommendation from the TOMHT side is to do it through caller metadata: the channel exists end-to-end today, and the adapter sketched below builds on documented extension points with no tracker changes. `component_selector` is a small diff in tracker code, but it weakens properties the tracker currently guarantees; the specifics are below. If there is a structural reason on the integration side that makes the callback a materially better fit, it can be accepted as an explicitly marked escape hatch under the conditions listed at the end.

## The metadata path, concretely

To make sure "the metadata path" means the same thing to everyone:

1. Between scans, the caller decides a track should switch and records it:

   ```python
   tracker.update_track_metadata(
       internal_track_id=tid,          # or public_track_id=...
       updates={"imm_profile": "B"},
   )
   ```

2. At the next expansion, that value is present on the `Track` handed to the custom hypothesiser (`track.metadata["imm_profile"]`), for every leaf, every scan. Dispatch happens inside caller-owned components.

3. The active profile is tracker state: it shows up on output tracks and in `get_track_tree_snapshot()`, so every per-scan capture of tracker output includes it automatically.

One genuine gap: the updater receives only the hypothesis (`updater.update(hypothesis)`), not the track, so metadata does not reach it directly. §4 of the API guide permits `SingleDistanceHypothesis` subclasses, which closes the gap: the hypothesiser stamps the chosen profile onto each hypothesis it emits, and a paired dispatching updater reads the stamp.

Both dispatch halves fit in one small adapter, so the per-profile components stay clean and single-profile — nothing around the tracker needs reshaping:

```python
class ProfileSwitchingHypothesiser:
    # profiles: {"A": (hyp_a, upd_a), "B": (hyp_b, upd_b)}
    # exposes .predictor for tracker wiring (§4)

    def hypothesise(self, track, detections, timestamp, **kwargs):
        profile = track.metadata.get("imm_profile", self.default)
        hyp, _ = self.profiles[profile]
        hypotheses = hyp.hypothesise(track, detections, timestamp, **kwargs)
        return stamp_with_profile(hypotheses, profile)

    @property
    def updater(self):
        # Updater whose update(hypothesis) reads the stamp and
        # delegates to the matching per-profile updater
        ...


switcher = ProfileSwitchingHypothesiser(profiles={...})
tracker = TOMHTTracker(updater=switcher.updater, hypothesiser=switcher, ...)
```

This is a sketch; a full worked version can be supplied. It will also be added to the API guide (§11) as a documented example regardless of where this discussion lands — the guide is currently not explicit enough that caller metadata is the intended channel for per-track behavior changes, and that will be fixed.

## The two designs, walked through the loop

|                     | metadata path | `component_selector` |
|---------------------|---------------|----------------------|
| after scan N        | read output tracks, classify | read output tracks, classify |
| record the decision | `update_track_metadata(...)` | update a caller-held `{internal_id: profile}` store |
| during scan N+1     | hypothesiser reads `track.metadata` and dispatches | tracker calls the selector; the store answers; tracker swaps components |

As far as can be judged from the tracker side, these are the same decision, made at the same point, from the same data (output tracks carry `internal_track_id`), with the same one-scan latency. And because the callback's only input is a track id, any policy that adapts to observed target behaviour must consult caller-held per-track state — this is forced by the signature, not an assumption about the implementation. The selector does not remove per-track bookkeeping; it relocates the store outside the tracker and moves the final dispatch from caller components into the tracker. In other words: **`component_selector` is the metadata path with the metadata held where the tracker cannot see it.** The costs below all follow from that relocation.

- **Auditability and reproducibility.** Under the metadata design the active profile is in-band tracker state: any per-scan capture of snapshots or outputs includes it automatically, and a recorded stream of caller→tracker calls (detections plus metadata updates) replays a run exactly, with none of the caller's decision logic in the loop. Reconstructing history requires per-scan capture or a change log under either design; the difference is that the selector's backing state is invisible to the tracker's inspection surfaces, so it needs separate instrumentation, and replay needs live caller code rather than recorded inputs.
- **Contract checking.** Today the component set is structurally validated once at construction (`predictor` XOR `hypothesiser`, `.predictor` exposed); the behavioural §4 contract — including the NLL convention — is the caller's to uphold, for exactly one component set. A dispatching adapter holds a finite profile registry, so every pair can be checked up front and TOMHT keeps one stable component interface. A selector makes variable pairs part of the tracker's runtime contract: each returned pair independently owes the full §4 contract — and, sharper: distances from profile A and profile B accumulate into one score history per track, judged against one set of confirmation/deletion thresholds. A convention mismatch between profiles does not fail loudly; it silently miscalibrates exactly the tracks that switched. That risk exists under both designs, but a single dispatching component contains it in one place instead of N.
- **Id spaces.** The draft selector receives the internal track id, so the caller-side store must live in internal-id space and bridge to whatever the classification logic keys on. `update_track_metadata` accepts either internal or public id (and fails loudly for a track that no longer exists); `get_map_output_tracks(include_unpublished=True)` supports deciding switches before publication.
- **Switch timing.** In the single-threaded call pattern the API assumes, caller code runs between tracker calls, so metadata writes are serialized against scans: a switch takes effect at the next `update_tracker()`, as a fact. The selector is consulted mid-scan; even single-threaded, "switches land on scan boundaries" becomes a convention the caller must uphold — evaluated once per tree per scan, backing state not mutated during the scan — rather than a property the API gives you.
- **The real cost is the invariant, not the lines.** Agreed that the selector is a small diff. What it changes is that "the tracker holds one (hypothesiser, updater) pair for its lifetime" stops being true, and every future tracker change, doc statement, and validation has to hold under "components may vary per track per scan."

## A caveat that applies to both designs

Under either mechanism, a switch is tree-level — it applies to all active leaves of the track at once — and takes effect at expansion going forward. Hypothesis branches expanded under profile A keep their A-scored history through the N-scan window, so mixed-profile branches coexist for a while, and score continuity across the switch remains the responsibility of the components performing it. Neither design helps or hurts here.

## If `component_selector` anyway

If there is a real ergonomic or structural win on the integration side, the callback can be accepted as an explicitly marked escape hatch, under conditions:

1. Marked experimental where it is visible: a constructor argument on the exported tracker is public de facto, so the argument name and its documentation must carry the escape-hatch status themselves, with no stability promise.
2. Every returned pair satisfies the §4 custom-hypothesiser contract, and all profiles emit distances under the same NLL convention.
3. Defined call discipline: consulted once per track tree per scan at a defined point, with the result used for that whole scan.
4. Defined failure semantics: `None` selects the main pair, malformed return values are rejected loudly, and callback exceptions propagate.
5. The selection is auditable: the callback returns a named result carrying a profile identity — e.g. `(profile_id, hypothesiser, updater)` — which the tracker mirrors into caller metadata each scan. An anonymous `(hypothesiser, updater)` tuple gives the tracker nothing it can record.

Worth noting: with these conditions met — especially 5 — the selector has effectively become the metadata path with an extra callback: the profile identity ends up in track metadata either way, and the remaining difference is the dispatch adapter sketched above.

## The question that decides it

The one thing not visible from the tracker side is whether a structural constraint in the surrounding system makes the callback a better fit — for example, if the module that decides profiles cannot reach the tracker handle to call `update_track_metadata`. If something like that is the case, that is the input that would tip this toward the escape-hatch route; otherwise, the metadata path plus the adapter above is the recommended shape.
