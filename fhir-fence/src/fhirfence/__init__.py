"""FHIR-Fence: task-conditioned FHIR minimization against privacy leakage in
clinical LLM agents.

Public API:

    from fhirfence import load_settings, build_context, select_nodes, run_grid, score
"""

from .config import Settings, load_settings
from .graph import (
    build_canary_registry,
    build_patient_graph,
    count_tokens,
    encounter_members,
    leaked_canaries,
    load_medication_dictionary,
    serialize_graph,
    serialize_patient_resource,
    CANARY_FIELDS,
    CANARY_TO_CLASS,
)
from .minimize import build_context, graph_relevance, score_node, select_nodes
from .llm import LLMClient, make_llm_client
from .agent import build_prompt, call_model, load_done, run_grid
from .evaluate import complete_cases, load_runs, score, write_tables

__version__ = "2.1"

__all__ = [
    "Settings", "load_settings",
    "build_canary_registry", "build_patient_graph", "count_tokens",
    "encounter_members", "leaked_canaries", "load_medication_dictionary",
    "serialize_graph", "serialize_patient_resource", "CANARY_FIELDS", "CANARY_TO_CLASS",
    "build_context", "graph_relevance", "score_node", "select_nodes",
    "LLMClient", "make_llm_client",
    "build_prompt", "call_model", "load_done", "run_grid",
    "complete_cases", "load_runs", "score", "write_tables",
]
