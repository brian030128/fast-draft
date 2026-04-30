"""Named tree archetypes for testing the prefix-sharing-benefit hypothesis.

Question we want to answer: when does cascade-style prefix sharing actually
beat flat batched attention? Hypothesis: cascade wins only when *enough*
queries (q) share a *long enough* prefix (P) to amortize the cascade
kernel's launch + merge overhead. The archetypes here span the (q, P, G)
space and add the ragged shapes that fall out of beam search / diverse
beam search so a follow-up bench can trace the boundary.

Each builder returns ``List[KVTreeNode]``: the same data structure
``tests/bench_tree_attn.py:build_tree_info`` produces, so the existing
``prepare_data`` / flat / cascade / ml_cascade harness consumes them
unchanged.

Categories (see plan file):
  Group A -- single-prefix sharing-degree boundary
    single_q1, single_q4, single_q16, single_q64
  Group B -- multi-prefix forest scaling
    forest_balanced, forest_ragged
  Group C -- depth (multi-level cascade)
    eagle_pyramid, variable_depth
  Group D -- beam-search edge cases
    beam_converged, beam_diverged
  Group E -- diverse beam search (3-level shared structure)
    dbs_balanced, dbs_pruned, dbs_late
"""

import os
import sys
from typing import Callable, Dict, List, Tuple

_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(_root, "3rdparty", "FastTree-Artifact", "kernel_bench"))

from kv_tree_simple import KVTreeNode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _propagate_request_ids(nodes: List[KVTreeNode]) -> None:
    """Walk leaf-to-root, attaching the leaf's request id to every ancestor.
    Mirrors bench_tree_attn.build_tree_info's bookkeeping."""
    req_id = 0
    for n in range(len(nodes)):
        if nodes[n].num_children == 0:
            idx = n
            while idx != -1:
                nodes[idx].requests.append(req_id)
                idx = nodes[idx].parent
            req_id += 1


def _new_node(parent: int, node_id: int, seqlen: int, num_children: int) -> KVTreeNode:
    n = KVTreeNode()
    n.parent = parent
    n.id = node_id
    n.seqlen = seqlen
    n.num_children = num_children
    return n


# ---------------------------------------------------------------------------
# Group A: single-prefix shallow trees (depth = 2)
# ---------------------------------------------------------------------------

def build_single_prefix(P: int, q: int, suffix_len: int = 8) -> List[KVTreeNode]:
    """1 prefix of length P, fanning out to q leaves of length suffix_len.

    Sharing degree = q. Boundary case for "is sharing beneficial?".
    """
    nodes: List[KVTreeNode] = []
    nodes.append(_new_node(parent=-1, node_id=0, seqlen=P, num_children=q))
    for k in range(q):
        nodes.append(_new_node(parent=0, node_id=1 + k, seqlen=suffix_len, num_children=0))
    _propagate_request_ids(nodes)
    return nodes


# ---------------------------------------------------------------------------
# Group B: multi-prefix forest
# ---------------------------------------------------------------------------

def build_forest_balanced(P: int, q: int, G: int, suffix_len: int = 8) -> List[KVTreeNode]:
    """G independent prefixes (each length P), each fanning out to q leaves.

    Implemented as a 1-token dummy root over G mid-level prefix nodes (matches
    bench_tree_attn.py num_prefixes>1 convention). Sharing degree = q within
    each group; G groups in parallel.
    """
    nodes: List[KVTreeNode] = []
    nodes.append(_new_node(parent=-1, node_id=0, seqlen=1, num_children=G))
    for g in range(G):
        nodes.append(_new_node(parent=0, node_id=1 + g, seqlen=P, num_children=q))
    nid = 1 + G
    for g in range(G):
        for k in range(q):
            nodes.append(_new_node(parent=1 + g, node_id=nid, seqlen=suffix_len, num_children=0))
            nid += 1
    _propagate_request_ids(nodes)
    return nodes


def build_forest_ragged(P: int, qs: List[int], suffix_len: int = 8) -> List[KVTreeNode]:
    """len(qs) groups, group g has qs[g] queries. Models post-pruning beam
    survival where some groups dominate and others trail."""
    G = len(qs)
    nodes: List[KVTreeNode] = []
    nodes.append(_new_node(parent=-1, node_id=0, seqlen=1, num_children=G))
    for g in range(G):
        nodes.append(_new_node(parent=0, node_id=1 + g, seqlen=P, num_children=qs[g]))
    nid = 1 + G
    for g in range(G):
        for k in range(qs[g]):
            nodes.append(_new_node(parent=1 + g, node_id=nid, seqlen=suffix_len, num_children=0))
            nid += 1
    _propagate_request_ids(nodes)
    return nodes


# ---------------------------------------------------------------------------
# Group C: multi-level cascade
# ---------------------------------------------------------------------------

def build_eagle_pyramid(P: int, fanout: List[int], step_len: int = 1) -> List[KVTreeNode]:
    """EAGLE multi-step canonical: prompt P -> fanout[0] children -> fanout[1]
    each -> ... Each non-leaf intermediate is one drafted token (step_len=1).
    Total leaves = product(fanout). Last fanout level is the leaves.
    """
    assert len(fanout) >= 1
    nodes: List[KVTreeNode] = []
    nodes.append(_new_node(parent=-1, node_id=0, seqlen=P, num_children=fanout[0]))
    # BFS: expand one level at a time.
    frontier = [0]
    nid = 1
    for d, f in enumerate(fanout):
        new_frontier: List[int] = []
        is_leaf_level = (d == len(fanout) - 1)
        seqlen = step_len if not is_leaf_level else step_len  # uniform; semantics same as EAGLE 1-token leaves
        children_for_each = (fanout[d + 1] if d + 1 < len(fanout) else 0)
        for parent in frontier:
            for _ in range(f):
                nodes.append(_new_node(parent=parent, node_id=nid, seqlen=seqlen,
                                        num_children=children_for_each))
                new_frontier.append(nid)
                nid += 1
        frontier = new_frontier
    _propagate_request_ids(nodes)
    return nodes


def build_variable_depth(P: int, depths: List[int], per_depth: int = 4,
                         step_len: int = 1) -> List[KVTreeNode]:
    """1 prefix -> per_depth `independent chains` per requested depth.
    e.g. depths=[4,3,2,1], per_depth=4 produces 4 chains of depth 4, 4 of
    depth 3, 4 of depth 2, 4 of depth 1 (each chain is a vertical strip
    of step_len-token nodes, leaf at the bottom of the strip).

    Models beam pruning where different beams are cut off at different
    depths.
    """
    nodes: List[KVTreeNode] = []
    total_leaves = per_depth * len(depths)
    nodes.append(_new_node(parent=-1, node_id=0, seqlen=P, num_children=total_leaves))
    nid = 1
    # For each depth bucket, build per_depth independent chains. Each chain
    # is a linear sequence of step_len nodes; the very last node has 0
    # children (= leaf). Chain head's parent is the prefix root.
    for d in depths:
        for _ in range(per_depth):
            chain_head_id = nid
            for level in range(d):
                parent = 0 if level == 0 else nid - 1
                nodes.append(_new_node(parent=parent, node_id=nid, seqlen=step_len,
                                        num_children=(1 if level < d - 1 else 0)))
                nid += 1
            # Update the prefix-root's child count tracking is implicit
            # via num_children=total_leaves set above; the head's parent is
            # already the root (level==0 case).
    _propagate_request_ids(nodes)
    return nodes


# ---------------------------------------------------------------------------
# Group D: beam-search edge cases
# ---------------------------------------------------------------------------

def build_chain(P: int, chain_len: int, step_len: int = 1) -> List[KVTreeNode]:
    """Single chain: prompt prefix -> chain_len drafted tokens, no fanout.
    Models a beam search collapsed to one survivor (degenerate). Cascade
    has nothing to share over -- expect parity with flat sequential."""
    nodes: List[KVTreeNode] = []
    nodes.append(_new_node(parent=-1, node_id=0, seqlen=P, num_children=1))
    for i in range(chain_len):
        is_leaf = (i == chain_len - 1)
        nodes.append(_new_node(parent=i, node_id=i + 1, seqlen=step_len,
                                num_children=(0 if is_leaf else 1)))
    _propagate_request_ids(nodes)
    return nodes


def build_independent(P: int, k: int, suffix_len: int = 1) -> List[KVTreeNode]:
    """k fully-independent prefixes, 1 query each (no inter-request sharing).
    Models beam-search step 0 with all-different prefixes -- cascade has no
    sharing payoff, pairs with single_q1 to confirm cascade tracks flat at
    the lower bound."""
    nodes: List[KVTreeNode] = []
    nodes.append(_new_node(parent=-1, node_id=0, seqlen=1, num_children=k))
    for g in range(k):
        # Each "group" is a single prefix-then-leaf stripped of cross-group sharing.
        nodes.append(_new_node(parent=0, node_id=1 + g, seqlen=P, num_children=1))
    nid = 1 + k
    for g in range(k):
        nodes.append(_new_node(parent=1 + g, node_id=nid, seqlen=suffix_len, num_children=0))
        nid += 1
    _propagate_request_ids(nodes)
    return nodes


# ---------------------------------------------------------------------------
# Group E: diverse beam search (3-level shared structure)
# ---------------------------------------------------------------------------

def build_dbs_balanced(P: int, G: int, B: int, C: int, S: int = 8) -> List[KVTreeNode]:
    """Diverse beam search canonical 3-level tree.
        prompt prefix (P)
          -> G group-context nodes (C tokens each, shared by group's B beams)
          -> G*B per-beam suffix nodes (S tokens each)
    Sharing degrees [G*B, B, 1] across the three levels."""
    nodes: List[KVTreeNode] = []
    nodes.append(_new_node(parent=-1, node_id=0, seqlen=P, num_children=G))
    # Group-context nodes
    for g in range(G):
        nodes.append(_new_node(parent=0, node_id=1 + g, seqlen=C, num_children=B))
    # Per-beam leaves
    nid = 1 + G
    for g in range(G):
        for b in range(B):
            nodes.append(_new_node(parent=1 + g, node_id=nid, seqlen=S, num_children=0))
            nid += 1
    _propagate_request_ids(nodes)
    return nodes


def build_dbs_pruned(P: int, G: int, Bs: List[int], C: int, S: int = 8) -> List[KVTreeNode]:
    """DBS mid-run with uneven survival: G groups, group g has Bs[g] beams.
    Tests cascade when the per-group sharing degree is asymmetric (the
    realistic mid-run shape)."""
    assert len(Bs) == G
    nodes: List[KVTreeNode] = []
    nodes.append(_new_node(parent=-1, node_id=0, seqlen=P, num_children=G))
    for g in range(G):
        nodes.append(_new_node(parent=0, node_id=1 + g, seqlen=C, num_children=Bs[g]))
    nid = 1 + G
    for g in range(G):
        for b in range(Bs[g]):
            nodes.append(_new_node(parent=1 + g, node_id=nid, seqlen=S, num_children=0))
            nid += 1
    _propagate_request_ids(nodes)
    return nodes


def build_dbs_late(P: int, G: int, B: int, S: int = 8) -> List[KVTreeNode]:
    """DBS late-run: groups have diverged so far that the L1 group context
    is too short to bother sharing. Equivalent shape: forest_balanced(P, B, G)
    but flagged separately for readability (sharing degrees [G*B, 1] only).
    Used as the comparison to dbs_balanced -- if dbs_balanced beats dbs_late
    at the same (P, G, B), the L1 group-context cascade level is paying off."""
    return build_forest_balanced(P=P, q=B, G=G, suffix_len=S)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Each entry: name -> (builder, default_kwargs, sweep_axes_dict).
# The sweep_axes_dict maps a parameter name to the list of values to try.
# The bench driver expands the cartesian product of sweep axes around the
# default kwargs to produce one row per cell.
ARCHETYPES: Dict[str, Tuple[Callable, dict, dict]] = {
    # Group A: q × P boundary sweep
    "single_q1":  (build_single_prefix, dict(q=1),  {"P": [128, 1024, 4096, 16384]}),
    "single_q4":  (build_single_prefix, dict(q=4),  {"P": [128, 1024, 4096, 16384]}),
    "single_q16": (build_single_prefix, dict(q=16), {"P": [128, 1024, 4096, 16384]}),
    "single_q64": (build_single_prefix, dict(q=64), {"P": [128, 1024, 4096, 16384]}),
    # Group B: forest scaling
    "forest_balanced": (build_forest_balanced, dict(P=2048, q=8), {"G": [1, 4, 16, 64]}),
    "forest_ragged":   (build_forest_ragged,   dict(P=2048, qs=[16, 8, 4, 1]), {}),
    # Group C: depth
    "eagle_pyramid":   (build_eagle_pyramid,   dict(P=2048, fanout=[8, 4, 2, 1]), {}),
    "variable_depth":  (build_variable_depth,  dict(P=2048, depths=[4, 3, 2, 1], per_depth=4), {}),
    # Group D: beam edge cases
    "beam_converged":  (build_chain,           dict(P=2048, chain_len=8), {}),
    "beam_diverged":   (build_independent,     dict(P=2048, k=64), {}),
    # Group E: diverse beam search
    "dbs_balanced": (build_dbs_balanced, dict(P=2048, G=4, B=4, C=128), {}),
    "dbs_pruned":   (build_dbs_pruned,   dict(P=2048, G=4, Bs=[8, 4, 2, 1], C=128), {}),
    "dbs_late":     (build_dbs_late,     dict(P=2048, G=4, B=4), {}),
}


def expand_archetype(name: str) -> List[Tuple[str, dict]]:
    """Return list of (label, kwargs) cells for a given archetype.
    label includes the swept parameter values for legibility."""
    builder, defaults, sweeps = ARCHETYPES[name]
    if not sweeps:
        return [(name, dict(defaults))]
    # Expand cartesian product
    keys = list(sweeps.keys())
    out: List[Tuple[str, dict]] = []

    def _recurse(idx: int, cur: dict, label_parts: List[str]):
        if idx == len(keys):
            out.append((f"{name}[{','.join(label_parts)}]", cur))
            return
        k = keys[idx]
        for v in sweeps[k]:
            new = dict(cur)
            new[k] = v
            _recurse(idx + 1, new, label_parts + [f"{k}={v}"])

    _recurse(0, dict(defaults), [])
    return out


def all_cells() -> List[Tuple[str, Callable, dict]]:
    """Flatten all archetypes into (label, builder, kwargs) rows."""
    out: List[Tuple[str, Callable, dict]] = []
    for name, (builder, _, _) in ARCHETYPES.items():
        for label, kwargs in expand_archetype(name):
            out.append((label, builder, kwargs))
    return out
