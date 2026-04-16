# ts_bridge schema — surface ↔ substrate interface

## Purpose

Define the contract between TensionLM (the TS-native language surface) and
TS-Core (the `UniversalLivingGraph` substrate), so the two can be coupled
without either depending on the internals of the other.

This document is the contract. When `ts_bridge` merges into TS-Core, the
**data shapes** below become the stable interface; the implementation may
be swapped freely on either side of it.

## Graph shape

### Node

```python
Node {
    id:            str            # unique within the graph
    content:       str            # human-readable label (token, concept, entity)
    topics:        list[str]      # topic tags (empty until concept extraction runs)
    activation:    float ∈ [0, 1] # current activation
    stability:     float ∈ [0, 1] # resistance to change (damping factor)
    base_strength: float ∈ [0, 1] # long-term importance floor
    metadata:      dict           # free-form; reserved keys below
}
```

**Reserved metadata keys** (set by TauExporter, read by TS-Core without
translation):

| Key              | Type | Meaning |
|------------------|------|---------|
| `absolute_pos`   | int  | Position in the original token stream |
| `token_id`       | int  | Tokenizer id |
| `source_model`   | str  | Checkpoint identifier (set at ingest site) |
| `emitted_at`     | str  | ISO-8601 timestamp of the ingest call |

### Edge

```python
Edge {
    src:      str              # source node id (past token)
    dst:      str              # destination node id (query/current token)
    weight:   float ∈ [0, 1]   # aggregated τ
    relation: str              # "tension" for τ-derived edges; future typed edges TBD
    metadata: dict             # free-form; reserved keys below
}
```

**Reserved metadata keys**:

| Key         | Type | Meaning |
|-------------|------|---------|
| `distance`  | int  | `dst.absolute_pos - src.absolute_pos` (positive — causal) |
| `layer`     | int  | Layer that emitted this edge (absent when aggregated across layers) |
| `head`      | int  | Head index (absent when aggregated) |

## Edge-emission protocol (TauExporter → graph)

For each generated or seen token at position `t`:

1. Gather τ from every local tension layer: `tau[L, T, H, W]`.
2. Classify each head via `head_filter.profile_heads`; select heads whose
   role is `LONG_RANGE` (and optionally `MID_RANGE`).
3. Aggregate across selected heads:
   `aggregated[t, w] = mean over selected (l, h) of τ[l, t, h, w]`.
4. For each window offset `w`, compute key absolute position
   `s = t - (W - w)`.  Skip `s < 0`.
5. If `aggregated[t, w] > edge_threshold` (default 0.3):
   - `upsert_node(query_id)` with `activation = 0.3 + mean(aggregated[t])`.
   - `upsert_node(key_id)` with `activation = 0.3 + aggregated[t, w]`.
   - `add_edge(src=key_id, dst=query_id, weight=aggregated[t, w])`.

`upsert_node` is idempotent; repeated inserts reinforce via max, not
overwrite.  Edges accumulate; `add_edge` on an existing `(src, dst, relation)`
triple promotes to `max(old, new)` weight.

## Identity convention

Default node id is position-scoped: `f"{content}#{absolute_pos}"`.  This keeps
repeated tokens distinct at the graph level — necessary because a future
concept-extraction pass needs to see the full token lattice before it can
collapse repeats into concepts.

`position_scoped_ids=False` is available for diagnostics (makes the graph
content-addressed and much smaller) but is **not recommended** as the default
because it fuses semantically-distinct occurrences.

## Direction of edges

All τ-derived edges run **src = past token → dst = current token**.  This
matches sigmoid tension's causal semantics ("this token is pulled on by these
prior tokens") and matches TS-Core's existing convention (edges point from
supporting concept → supported concept).

## Merge path into TS-Core

When this code lands in TS-Core:

1. Replace `ts_bridge.graph.UniversalLivingGraph` with the canonical TS-Core
   implementation (import only; delete the local class).
2. `TauExporter` moves to `TS-Core/ingest/tension_exporter.py` unchanged.
3. `HeadRole` / `HeadStat` / `profile_heads` move to
   `TS-Core/ingest/head_filter.py` unchanged.
4. `ts_bridge` becomes a thin shim in this repo that re-exports the TS-Core
   names for backward compatibility with generation scripts.

No protocol change is required: the Node/Edge shapes above are
TS-Core-compatible by construction.

## What is deliberately NOT in scope here

- **Graph → surface biasing.**  The reverse-direction interface (let the
  graph condition TensionLM generation) is Phase 2.  Out of scope for this
  schema; will get its own `bias.py` + schema amendment.
- **Concept extraction / deduplication.**  Phase 1 emits position-scoped
  token nodes.  Concept nodes (e.g. "mammals" as a single node regardless of
  where it was mentioned) are a later pass that runs over the accumulated
  graph, not during export.
- **Embedding storage.**  BoggersTheAI nodes carry optional embeddings; we
  don't emit them here because computing them during τ-export is expensive
  and orthogonal to the tension-structure thesis.  TS-Core's embedding
  pipeline can populate them post-hoc.
- **Wave cycle integration.**  The exported graph is raw input to TS-Core's
  wave-cycle runner.  The runner (`propagate → relax → prune → merge ...`)
  is TS-Core's job and intentionally not duplicated here.
