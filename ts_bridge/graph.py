"""
ts_bridge.graph
================

A minimal constraint graph shape compatible with `TS-Core/UniversalLivingGraph`.

Dict-backed and JSON-serialisable. Deliberately dependency-free — this module is
the contract, not the runtime. The goal is that the exported JSON can be loaded
into a real `UniversalLivingGraph` without translation once this code merges
into TS-Core.

Node fields mirror BoggersTheAI's node schema:
  - content       : str       — human-readable label (token or concept)
  - topics        : list[str] — topic tags (empty for now, filled by a later
                                 concept-extraction pass)
  - activation    : float     — current activation in [0, 1]
  - stability     : float     — resistance to change in [0, 1]
  - base_strength : float     — long-term importance floor in [0, 1]
  - metadata      : dict      — free-form (source_token_id, layer, etc.)

Edge fields:
  - src, dst : str    — node ids
  - weight   : float  — edge strength (aggregated τ)
  - relation : str    — label (default "tension"; room for future typed edges)
  - metadata : dict   — free-form (heads that contributed, layer mask, etc.)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class Node:
    id:            str
    content:       str
    topics:        list[str] = field(default_factory=list)
    activation:    float     = 0.5
    stability:     float     = 0.5
    base_strength: float     = 0.5
    metadata:      dict      = field(default_factory=dict)


@dataclass
class Edge:
    src:      str
    dst:      str
    weight:   float
    relation: str  = "tension"
    metadata: dict = field(default_factory=dict)


class UniversalLivingGraph:
    """
    Local stand-in for TS-Core's UniversalLivingGraph.  Minimal by design — we
    only need the pieces that the τ-export pipeline touches.  When this code
    merges into TS-Core, replace this class with an import.
    """

    def __init__(self):
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge]      = []

    # ── Node ops ──────────────────────────────────────────────────────────────

    def upsert_node(
        self,
        id:            str,
        content:       str,
        activation:    float | None = None,
        stability:     float | None = None,
        base_strength: float | None = None,
        topics:        list[str] | None = None,
        metadata:      dict | None  = None,
    ) -> Node:
        """
        Insert or reinforce a node.  If it already exists, `activation` moves
        toward the new value with momentum (new activation = max(existing, new)
        to reflect positive reinforcement — consistent with BoggersTheAI's
        "activation spreads, does not overwrite" convention).
        """
        existing = self.nodes.get(id)
        if existing is None:
            node = Node(
                id            = id,
                content       = content,
                topics        = topics or [],
                activation    = 0.5 if activation    is None else activation,
                stability     = 0.5 if stability     is None else stability,
                base_strength = 0.5 if base_strength is None else base_strength,
                metadata      = metadata or {},
            )
            self.nodes[id] = node
            return node

        if activation    is not None: existing.activation    = max(existing.activation,    activation)
        if stability     is not None: existing.stability     = max(existing.stability,     stability)
        if base_strength is not None: existing.base_strength = max(existing.base_strength, base_strength)
        if topics:
            # union — topics are labels, order not meaningful
            existing.topics = sorted(set(existing.topics) | set(topics))
        if metadata:
            existing.metadata.update(metadata)
        return existing

    # ── Edge ops ──────────────────────────────────────────────────────────────

    def add_edge(
        self,
        src:      str,
        dst:      str,
        weight:   float,
        relation: str  = "tension",
        metadata: dict | None = None,
    ) -> Edge:
        """
        Add an edge, or reinforce an existing (src, dst, relation) triple by
        taking the max of the observed weights.  Max-reinforcement mirrors
        the sigmoid tension semantics: a constraint is as strong as the
        strongest evidence seen for it, not the average.
        """
        for e in self.edges:
            if e.src == src and e.dst == dst and e.relation == relation:
                if weight > e.weight:
                    e.weight = weight
                if metadata:
                    e.metadata.update(metadata)
                return e
        edge = Edge(src=src, dst=dst, weight=weight, relation=relation,
                    metadata=metadata or {})
        self.edges.append(edge)
        return edge

    # ── Aggregate stats (used by Phase 1 diagnostics) ─────────────────────────

    def density(self) -> float:
        """Edges / (nodes choose 2).  Matches BoggersTheAI's /metrics endpoint."""
        n = len(self.nodes)
        if n < 2:
            return 0.0
        return len(self.edges) / (n * (n - 1) / 2)

    def mean_activation(self) -> float:
        if not self.nodes:
            return 0.0
        return sum(n.activation for n in self.nodes.values()) / len(self.nodes)

    def mean_edge_weight(self) -> float:
        if not self.edges:
            return 0.0
        return sum(e.weight for e in self.edges) / len(self.edges)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "nodes": {id: asdict(n) for id, n in self.nodes.items()},
            "edges": [asdict(e) for e in self.edges],
        }

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, data: dict) -> "UniversalLivingGraph":
        g = cls()
        for id, n in data.get("nodes", {}).items():
            g.nodes[id] = Node(**n)
        for e in data.get("edges", []):
            g.edges.append(Edge(**e))
        return g

    @classmethod
    def from_json(cls, path: str | Path) -> "UniversalLivingGraph":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def __repr__(self) -> str:
        return (
            f"UniversalLivingGraph(nodes={len(self.nodes)}, "
            f"edges={len(self.edges)}, density={self.density():.3f}, "
            f"mean_act={self.mean_activation():.3f})"
        )
