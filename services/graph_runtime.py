"""Backward-compatible facade for graph runtime helpers.

Prefer importing from the narrower modules instead:
- services.checkpointer_runtime
- services.graph_payloads
- services.graph_invoker
"""
from __future__ import annotations

try:
    from atlas.services.checkpointer_runtime import ensure_checkpointer
    from atlas.services.graph_invoker import invoke_atlas_graph
    from atlas.services.graph_payloads import (
        build_graph_config,
        build_initial_state,
        extract_final_response,
    )
except ImportError:
    from services.checkpointer_runtime import ensure_checkpointer  # type: ignore
    from services.graph_invoker import invoke_atlas_graph  # type: ignore
    from services.graph_payloads import (  # type: ignore
        build_graph_config,
        build_initial_state,
        extract_final_response,
    )
