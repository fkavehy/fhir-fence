"""FHIR-Fence minimizer: scoring function, node selection, and the five defense
context builders. Refactored from fhirfence.py; parameters come from Settings.

Scoring:  Score(v,q) = alpha*R_policy + beta*R_graph - lambda*S(v),  retain iff >= tau.
The one-hop graph-proximity weight is gamma (was a literal 0.7 in the original).
"""

from __future__ import annotations

from .config import Settings
from .graph import (
    REDACT_IDENTIFIERS,
    encounter_members,
    serialize_graph,
)


def graph_relevance(G, node, focal_encounter, focal_members=None, gamma=0.7):
    """R_graph: proximity to the encounter the task is about.

    1.0    the focal encounter itself, or a resource attached to it
    gamma  one hop from a focal-encounter resource
    0.0    everything else
    """
    if node == focal_encounter:
        return 1.0
    if focal_members is None:
        focal_members = set(encounter_members(G, focal_encounter))
    if node in focal_members:
        return 1.0
    und = G.to_undirected(as_view=True)
    for m in focal_members:
        if und.has_edge(node, m):
            return gamma
    return 0.0


def score_node(cfg: Settings, G, node, authorized_types, focal_encounter,
               focal_members=None, task_required_override=None):
    """Score one node. Sensitivity is conditioned on authorization: an authorized
    resource type gets the task_required weight (0.0) so task-needed sensitive data is
    not penalized below tau. Attribute nodes (canaries) never get the override."""
    s = cfg.scoring
    if task_required_override is None:
        task_required_override = bool(s["task_required_override"])
    d = G.nodes[node]
    r_policy = 1.0 if d["resource_type"] in authorized_types else 0.0
    r_graph = graph_relevance(G, node, focal_encounter, focal_members,
                              gamma=float(s["gamma"]))
    sens = d["sensitivity"]
    if task_required_override and r_policy == 1.0 and d["node_kind"] == "resource":
        sens = cfg.sensitivity_weights["task_required"]
    return float(s["alpha"]) * r_policy + float(s["beta"]) * r_graph - float(s["lambda"]) * sens


def select_nodes(cfg: Settings, G, authorized_types, focal_encounter, tau=None, **kw):
    """Node selection. The Patient root is always kept; its exposure is governed by the
    field-level policy, not by node selection."""
    if tau is None:
        tau = float(cfg.scoring["tau"])
    pk = G.graph["patient_key"]
    root = f"Patient/{pk}"
    focal_members = set(encounter_members(G, focal_encounter))
    keep, scores = {root}, {}
    for n, d in G.nodes(data=True):
        if n == root:
            continue
        s = score_node(cfg, G, n, authorized_types, focal_encounter, focal_members, **kw)
        scores[n] = s
        if d["node_kind"] == "resource" and s >= tau:
            keep.add(n)
    return keep, scores


def build_context(cfg: Settings, G, case, defense, tau=None, **kw):
    """Serialized context for one (case, defense) pair."""
    pk = G.graph["patient_key"]
    root = f"Patient/{pk}"
    enc = case["encounter_node"]
    auth = case["authorized_types"]
    fields = case["patient_fields"]

    if defense in ("B0", "B1"):
        # B1 differs from B0 only in the system prompt; the context is identical.
        return serialize_graph(G)

    if defense == "B2":
        # Direct identifiers removed; clinical content (incl. the synthetic clinical
        # canary) deliberately retained — the contrast that B2's leakage breakdown shows.
        nodes = [n for n, d in G.nodes(data=True)
                 if d["node_kind"] == "resource"
                 or d.get("sensitivity_class") != "direct_identifier"]
        return serialize_graph(G, nodes=nodes, include_attributes=True,
                               patient_field_policy=REDACT_IDENTIFIERS)

    if defense == "B3":
        nodes = [root] + [n for n, d in G.nodes(data=True)
                          if d["node_kind"] == "resource" and d["resource_type"] in auth]
        return serialize_graph(G, nodes=nodes, include_attributes=False,
                               patient_field_policy=fields)

    if defense == "B4":
        nodes, _ = select_nodes(cfg, G, auth, enc, tau=tau, **kw)
        return serialize_graph(G, nodes=nodes, include_attributes=False,
                               patient_field_policy=fields)

    raise ValueError(f"unknown defense {defense}")
