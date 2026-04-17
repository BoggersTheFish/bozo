"""
ts_bridge — surface ↔ substrate interface for TensionLM and TS-Core.

Phase 1 (current): export TensionLM's internal τ field as weighted edges on a
`UniversalLivingGraph`-shaped target.  See PLAN.md §Phase 1 and schema.md for
the contract.
"""

from .graph       import Edge, Node, UniversalLivingGraph
from .head_filter import (
    HeadRole, HeadStat,
    format_profile, profile_heads, select_signal_heads,
)
from .export      import ExportStats, TauExporter
from .streaming   import StreamStats, StreamingTauExporter
from .bias        import BiasStats, GraphBias

__all__ = [
    "Edge", "Node", "UniversalLivingGraph",
    "HeadRole", "HeadStat",
    "format_profile", "profile_heads", "select_signal_heads",
    "ExportStats", "TauExporter",
    "StreamStats", "StreamingTauExporter",
    "BiasStats", "GraphBias",
]
