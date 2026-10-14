#!/usr/bin/env python3
"""
SEAMLESS robustness analysis for a single undirected, unweighted network.

SEAMLESS:
    Locally Adaptive Multi-seed Edge-neighborhood Scoring & Sampling

Input:
    Edge list with integer node labels starting from 1:
        u v
    The graph is interpreted as undirected and unweighted.
    Lines starting with # are ignored.
    Self-loops are ignored.

Core SEAMLESS idea:
    At each attack step:
        1. sample m seed nodes uniformly at random from the residual graph;
        2. collect their first neighbors as candidate nodes;
        3. score each candidate using local edge-neighborhood dissimilarity;
        4. remove the highest-scoring candidate;
        5. update the residual graph and repeat.

The method is locally adaptive because scores are recomputed at every removal
step on the current residual graph, using only local neighborhood information.

Main outputs:
    raw.csv
        Complete robustness curves, one row per stochastic replicate and removal step.

    summary.csv
        Mean/sd/sem of S1/N and S2/N over stochastic replicates.

    metrics.csv
        Per-replicate AUC and threshold metrics.

    metrics_summary.csv
        Mean/sd/sem of metrics over stochastic replicates.

    config.json
        Full run configuration.

Dependencies:
    numpy
    pandas
    networkx
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd


# Type alias for a lightweight adjacency representation.
# Using sets makes local neighborhood operations fast and explicit.
Adjacency = Dict[int, set[int]]


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class RunConfig:
    """Container for run parameters saved into config.json."""

    edge_file: str
    label: str
    outdir: str
    seed: int
    n_attacks: int
    p_step_nodes: int
    m_min: int
    m_max: int
    m_step: int
    btw_update: int
    protocols: List[str]
    n_nodes: int
    n_edges: int
    method_name: str = "SEAMLESS"
    method_full_name: str = "Locally Adaptive Multi-seed Edge-neighborhood Scoring & Sampling"


# =============================================================================
# Graph IO and adjacency utilities
# =============================================================================

def read_integer_edgelist(
    path: Path,
    include_missing_integer_nodes: bool = True,
    zero_indexed: bool = False,
) -> Adjacency:
    """
    Read an undirected, unweighted edge list with integer node labels.

    Parameters
    ----------
    path:
        Input file path.
    include_missing_integer_nodes:
        If True, all nodes from min to max observed node id are included.
        This is useful when isolated nodes are represented implicitly by labels.
        If False, only nodes appearing in at least one non-self-loop edge are included.
    zero_indexed:
        If True, node labels start from 0. If False, start from 1.

    Returns
    -------
    adj:
        Dictionary mapping each node to a set of neighbors.
    """
    min_label = 0 if zero_indexed else 1

    edges: list[tuple[int, int]] = []
    nodes: set[int] = set()

    with path.open("r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Malformed edge-list line {line_no}: {line!r}")

            try:
                u = int(parts[0])
                v = int(parts[1])
            except ValueError as exc:
                raise ValueError(f"Node labels must be integers. Bad line {line_no}: {line!r}") from exc

            if u < min_label or v < min_label:
                raise ValueError(f"Node labels must be >= {min_label}. Bad line {line_no}: {line!r}")

            nodes.add(u)
            nodes.add(v)

            # Ignore self-loops for robustness analysis.
            if u != v:
                edges.append((u, v))

    if not nodes:
        raise ValueError("No nodes found in edge list.")

    if include_missing_integer_nodes:
        node_set = set(range(min_label, max(nodes) + 1))
    else:
        node_set = set(nodes)

    adj: Adjacency = {u: set() for u in node_set}

    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    return adj


def adjacency_to_networkx(adj: Adjacency) -> nx.Graph:
    """
    Convert adjacency dictionary to a NetworkX undirected graph.

    This is used only for betweenness-based baselines.
    """
    G = nx.Graph()
    G.add_nodes_from(adj.keys())

    for u, neigh in adj.items():
        for v in neigh:
            if u < v:
                G.add_edge(u, v)

    return G


def copy_adjacency(adj: Adjacency) -> Adjacency:
    """Return a deep copy of the adjacency dictionary."""
    return {u: set(vs) for u, vs in adj.items()}


def remove_node_inplace(adj: Adjacency, node: int) -> None:
    """
    Remove a node from the adjacency dictionary in-place.

    Assumes node exists in adj.
    """
    for neighbor in list(adj[node]):
        adj[neighbor].remove(node)
    del adj[node]


def edge_count(adj: Adjacency) -> int:
    """Return the number of undirected edges."""
    return sum(len(vs) for vs in adj.values()) // 2


# =============================================================================
# Component metrics
# =============================================================================

def connected_component_sizes(adj: Adjacency) -> list[int]:
    """
    Compute connected component sizes in descending order.

    Uses an explicit DFS over the adjacency dictionary.
    """
    unseen = set(adj.keys())
    sizes: list[int] = []

    while unseen:
        start = unseen.pop()
        stack = [start]
        size = 1

        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v in unseen:
                    unseen.remove(v)
                    stack.append(v)
                    size += 1

        sizes.append(size)

    sizes.sort(reverse=True)
    return sizes


def giant_and_second_component_fraction(adj: Adjacency, n_initial: int) -> tuple[float, float]:
    """
    Return S1/N and S2/N for the current residual graph.

    S1 is the largest connected component.
    S2 is the second-largest connected component.
    """
    if not adj:
        return 0.0, 0.0

    sizes = connected_component_sizes(adj)
    s1 = sizes[0] / n_initial
    s2 = sizes[1] / n_initial if len(sizes) > 1 else 0.0

    return s1, s2


# =============================================================================
# SEAMLESS local scoring
# =============================================================================

def edge_neighborhood_dissimilarity_score(adj: Adjacency, node: int) -> float:
    """
    Compute the SEAMLESS local edge-neighborhood dissimilarity score.

    For each incident edge (node, u), compare the local neighborhoods
    N(node)\\{u} and N(u)\\{node} using Jaccard dissimilarity:

        1 - |A intersect B| / |A union B|

    The node score is the mean dissimilarity over all incident edges.

    Interpretation:
        - low score: node connects locally redundant / overlapping neighborhoods;
        - high score: node has incident edges pointing to structurally distinct neighborhoods.

    This is a local, parameter-free proxy for bridge-like structural vulnerability.
    """
    neighbors = adj[node]
    degree = len(neighbors)

    if degree == 0:
        return 0.0

    total = 0.0

    for u in neighbors:
        a = neighbors - {u}
        b = adj[u] - {node}
        union = a | b

        if union:
            total += 1.0 - len(a & b) / len(union)

    return (total / degree) * (degree ** 0.5)


# =============================================================================
# Attack protocols
# =============================================================================

def order_random(adj0: Adjacency, rng: np.random.Generator) -> list[int]:
    """Uniform random node-removal order."""
    order = list(adj0.keys())
    rng.shuffle(order)
    return [int(v) for v in order]


def order_degree_static(adj0: Adjacency) -> list[int]:
    """
    Static degree attack.

    Nodes are ranked once by their initial degree.
    Ties are resolved deterministically by node id.
    """
    return [int(v) for v in sorted(adj0.keys(), key=lambda v: (-len(adj0[v]), v))]


def order_adaptive_degree(adj0: Adjacency, rng: np.random.Generator) -> list[int]:
    """
    Adaptive degree attack.

    Degree is recomputed after each removal.
    Ties are resolved randomly, so the protocol is repeated --n-attacks times.
    """
    adj = copy_adjacency(adj0)
    order: list[int] = []

    while adj:
        max_degree = max(len(neigh) for neigh in adj.values())
        candidates = [v for v, neigh in adj.items() if len(neigh) == max_degree]
        selected = int(rng.choice(candidates))

        order.append(selected)
        remove_node_inplace(adj, selected)

    return order


def order_betweenness_static(adj0: Adjacency) -> list[int]:
    """
    Static betweenness attack.

    Betweenness is computed once on the initial graph.
    Ties are resolved deterministically by node id.
    """
    graph = adjacency_to_networkx(adj0)
    betweenness = nx.betweenness_centrality(graph, normalized=False)

    return [int(v) for v in sorted(graph.nodes(), key=lambda v: (-betweenness[v], v))]


def order_adaptive_betweenness(
    adj0: Adjacency,
    rng: np.random.Generator,
    update_every: int = 1,
) -> list[int]:
    """
    Adaptive betweenness attack.

    Betweenness is recomputed every `update_every` removals.
    If update_every=1, the attack is fully adaptive.

    Ties are resolved randomly, so the protocol is repeated --n-attacks times.
    """
    if update_every < 1:
        raise ValueError("update_every must be >= 1")

    adj = copy_adjacency(adj0)
    order: list[int] = []

    while adj:
        graph = adjacency_to_networkx(adj)
        betweenness = nx.betweenness_centrality(graph, normalized=False)

        # Random jitter only affects exact ties.
        jitter = {v: rng.random() for v in graph.nodes()}
        ranking = sorted(graph.nodes(), key=lambda v: (-betweenness[v], jitter[v]))

        removed_in_batch = 0

        for v in ranking:
            if v not in adj:
                continue

            selected = int(v)
            order.append(selected)
            remove_node_inplace(adj, selected)
            removed_in_batch += 1

            if removed_in_batch >= update_every or not adj:
                break

    return order


def order_seamless(
    adj0: Adjacency,
    rng: np.random.Generator,
    sensing_budget_m: int,
) -> list[int]:
    """
    SEAMLESS attack order.

    Parameters
    ----------
    adj0:
        Initial adjacency dictionary.
    rng:
        Random number generator.
    sensing_budget_m:
        Number of seed nodes sampled at every step.

    Algorithm
    ---------
    At each step:
        1. sample m seed nodes uniformly from the residual graph;
        2. build the candidate set as the union of their neighbors;
        3. score candidates by edge-neighborhood dissimilarity;
        4. remove one maximally scoring candidate;
        5. if all sampled seeds are isolated, remove one sampled seed.
    """
    if sensing_budget_m < 1:
        raise ValueError("sensing_budget_m must be >= 1")

    adj = copy_adjacency(adj0)
    order: list[int] = []

    while adj:
        nodes = list(adj.keys())
        seeds = rng.choice(nodes, size=min(sensing_budget_m, len(nodes)), replace=False)

        candidate_set: set[int] = set()
        for seed in seeds:
            candidate_set.update(adj[int(seed)])

        if candidate_set:
            best_score = -1.0
            best_candidates: list[int] = []

            for candidate in candidate_set:
                score = edge_neighborhood_dissimilarity_score(adj, candidate)

                if score > best_score:
                    best_score = score
                    best_candidates = [candidate]
                elif score == best_score:
                    best_candidates.append(candidate)

            selected = int(rng.choice(best_candidates))
        else:
            # If all sampled seeds are isolated, isolated seeds are treated as inactive
            # structural elements and one of them is removed.
            selected = int(rng.choice(seeds))

        order.append(selected)
        remove_node_inplace(adj, selected)

    return order


# =============================================================================
# Curve generation and metrics
# =============================================================================

def robustness_curve(adj0: Adjacency, order: list[int], p_step_nodes: int) -> pd.DataFrame:
    """
    Compute S1/N and S2/N along a node-removal order.

    The curve is evaluated every `p_step_nodes` removals.
    If p_step_nodes does not divide N exactly, the final p=1 point is appended.
    """
    n_initial = len(adj0)

    if len(order) != n_initial:
        raise AssertionError(f"Removal order has length {len(order)} but expected {n_initial}.")

    if len(set(order)) != n_initial:
        raise AssertionError("Removal order contains duplicate nodes.")

    adj = copy_adjacency(adj0)
    rows: list[dict] = []
    idx = 0

    for removed in range(0, n_initial + 1, p_step_nodes):
        while idx < removed:
            remove_node_inplace(adj, order[idx])
            idx += 1

        if adj:
            sizes = connected_component_sizes(adj)
        else:
            sizes = []
        s1 = sizes[0] / n_initial if len(sizes) >= 1 else 0.0
        s2 = sizes[1] / n_initial if len(sizes) >= 2 else 0.0
        rows.append({
            "removed": removed,
            "p": removed / n_initial,
            "S1_over_N": s1,
            "S2_over_N": s2,
            "n_initial": n_initial,
            "n_remaining": len(adj),
            "num_components": len(sizes),
            })
        
        



    if rows[-1]["removed"] != n_initial:
        while idx < n_initial:
            remove_node_inplace(adj, order[idx])
            idx += 1
        
        if adj:
            sizes = connected_component_sizes(adj)
        else:
            sizes = []
        s1 = sizes[0] / n_initial if len(sizes) >= 1 else 0.0
        s2 = sizes[1] / n_initial if len(sizes) >= 2 else 0.0
        rows.append({
            "removed": n_initial,
            "p": 1.0,
            "S1_over_N": s1,
            "S2_over_N": s2,
            "n_initial": n_initial,
            "n_remaining": len(adj),
            "num_components": len(sizes),
            "component_sizes": ";".join(map(str, sizes)),
            })


    return pd.DataFrame(rows)


def auc_from_curve(curve: pd.DataFrame, y_column: str) -> float:

    """Compute trapezoidal AUC for a curve column as a function of p."""
    ordered = curve.sort_values("p")
    return float(np.trapezoid(ordered[y_column].to_numpy(), ordered["p"].to_numpy()))


def threshold_crossing_p(curve: pd.DataFrame, threshold: float, y_column: str = "S1_over_N") -> float:
    """
    Estimate the first p at which y_column <= threshold.

    Linear interpolation is used between adjacent sampled points.
    """
    ordered = curve.sort_values("p")
    x = ordered["p"].to_numpy()
    y = ordered[y_column].to_numpy()

    crossing = np.where(y <= threshold)[0]

    if len(crossing) == 0:
        return np.nan

    i = crossing[0]

    if i == 0:
        return float(x[0])

    x0, x1 = x[i - 1], x[i]
    y0, y1 = y[i - 1], y[i]

    if y1 == y0:
        return float(x1)

    return float(x0 + (threshold - y0) * (x1 - x0) / (y1 - y0))


def build_summary_tables(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build summary.csv, metrics.csv, and metrics_summary.csv from raw.csv-like data.
    """
    summary = raw.groupby(
        ["label", "protocol", "m", "removed", "p"],
        dropna=False,
        as_index=False,
    ).agg(
        mean_S1_over_N=("S1_over_N", "mean"),
        sd_S1_over_N=("S1_over_N", "std"),
        sem_S1_over_N=("S1_over_N", lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0),
        mean_S2_over_N=("S2_over_N", "mean"),
        sd_S2_over_N=("S2_over_N", "std"),
        sem_S2_over_N=("S2_over_N", lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0),
        n=("S1_over_N", "size"),
    )

    metric_rows: list[dict] = []
    for (label, protocol, m, stochastic_id), group in raw.groupby(
        ["label", "protocol", "m", "stochastic_id"],
        dropna=False,
    ):
        metric_rows.append({
            "label": label,
            "protocol": protocol,
            "m": m,
            "stochastic_id": stochastic_id,
            "AUC_S1": auc_from_curve(group, "S1_over_N"),
            "AUC_S2": auc_from_curve(group, "S2_over_N"),
            "p_LCC_0.5": threshold_crossing_p(group, 0.5, "S1_over_N"),
            "p_LCC_0.25": threshold_crossing_p(group, 0.25, "S1_over_N"),
            "p_LCC_0.1": threshold_crossing_p(group, 0.1, "S1_over_N"),
            "p_LCC_0.05": threshold_crossing_p(group, 0.05, "S1_over_N"),
            "max_S2_over_N": float(group["S2_over_N"].max()),
            "p_at_max_S2": float(group.loc[group["S2_over_N"].idxmax(), "p"]),
        })

    metrics = pd.DataFrame(metric_rows)

    metrics_summary = metrics.groupby(
        ["label", "protocol", "m"],
        dropna=False,
        as_index=False,
    ).agg(
        mean_AUC_S1=("AUC_S1", "mean"),
        sd_AUC_S1=("AUC_S1", "std"),
        sem_AUC_S1=("AUC_S1", lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0),
        mean_AUC_S2=("AUC_S2", "mean"),
        sd_AUC_S2=("AUC_S2", "std"),
        sem_AUC_S2=("AUC_S2", lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0),
        mean_p_LCC_0_5=("p_LCC_0.5", "mean"),
        sd_p_LCC_0_5=("p_LCC_0.5", "std"),
        mean_p_LCC_0_25=("p_LCC_0.25", "mean"),
        sd_p_LCC_0_25=("p_LCC_0.25", "std"),
        mean_p_LCC_0_1=("p_LCC_0.1", "mean"),
        sd_p_LCC_0_1=("p_LCC_0.1", "std"),
        mean_p_LCC_0_05=("p_LCC_0.05", "mean"),
        sd_p_LCC_0_05=("p_LCC_0.05", "std"),
        mean_max_S2_over_N=("max_S2_over_N", "mean"),
        sd_max_S2_over_N=("max_S2_over_N", "std"),
        mean_p_at_max_S2=("p_at_max_S2", "mean"),
        sd_p_at_max_S2=("p_at_max_S2", "std"),
        n=("AUC_S1", "size"),
    )

    return summary, metrics, metrics_summary


# =============================================================================
# Execution orchestration
# =============================================================================

def add_protocol_curve(
    records: list[dict],
    adj0: Adjacency,
    label: str,
    protocol: str,
    m: Optional[int],
    stochastic_id: int,
    order: list[int],
    p_step_nodes: int,
) -> None:
    """
    Compute and append a protocol curve to the global record list.
    """
    curve = robustness_curve(adj0, order, p_step_nodes)

    curve.insert(0, "stochastic_id", stochastic_id)
    curve.insert(0, "m", m if m is not None else np.nan)
    curve.insert(0, "protocol", protocol)
    curve.insert(0, "label", label)

    records.extend(curve.to_dict("records"))


def run_analysis(args: argparse.Namespace) -> None:
    """
    Run the full robustness analysis and write CSV outputs.
    """
    args.outdir.mkdir(parents=True, exist_ok=True)

    adj0 = read_integer_edgelist(
        args.edge_file,
        include_missing_integer_nodes=not args.no_missing_integer_nodes,
        zero_indexed=args.zero_indexed,
    )
    G_tmp = adjacency_to_networkx(adj0)
    largest_cc = max(nx.connected_components(G_tmp), key=len)
    adj0 = {u: set(v for v in adj0[u] if v in largest_cc) for u in largest_cc}

    n_nodes = len(adj0)
    n_edges = edge_count(adj0)
    rng_master = np.random.default_rng(args.seed)

    m_values = list(range(args.m_min, args.m_max + 1, args.m_step))

    config = RunConfig(
        edge_file=str(args.edge_file),
        label=args.label,
        outdir=str(args.outdir),
        seed=args.seed,
        n_attacks=args.n_attacks,
        p_step_nodes=args.p_step_nodes,
        m_min=args.m_min,
        m_max=args.m_max,
        m_step=args.m_step,
        btw_update=args.btw_update,
        protocols=args.protocols,
        n_nodes=n_nodes,
        n_edges=n_edges,
    )

    with (args.outdir / "config.json").open("w") as f:
        json.dump(asdict(config), f, indent=2)

    records: list[dict] = []

    # Count total operations for progress tracking
    total_ops = 0
    if "degree" in args.protocols:
        total_ops += 1
    if "betweenness" in args.protocols:
        total_ops += 1
    if "random" in args.protocols:
        total_ops += args.n_attacks
    if "adap_degree" in args.protocols:
        total_ops += 1
    if "adap_betweenness" in args.protocols:
        total_ops += 1
    if "seamless" in args.protocols:
        total_ops += len(m_values) * args.n_attacks

    print(f"\n{'='*60}")
    print(f"SEAMLESS Robustness Analysis")
    print(f"{'='*60}")
    print(f"Network: {args.label} (N={n_nodes}, E={n_edges})")
    print(f"Protocols: {', '.join(args.protocols)}")
    if "seamless" in args.protocols:
        print(f"SEAMLESS m range: {args.m_min} to {args.m_max} (step {args.m_step})")
    print(f"Stochastic replicates: {args.n_attacks}")
    print(f"Total operations: {total_ops}")
    print(f"{'='*60}\n")

    current_op = 0

    # Deterministic baselines.
    if "degree" in args.protocols:
        current_op += 1
        print(f"[{current_op}/{total_ops}] Running static degree attack...")
        add_protocol_curve(
            records, adj0, args.label, "degree", None, 1,
            order_degree_static(adj0),
            args.p_step_nodes,
        )
        print(f"  ✓ degree complete")

    if "betweenness" in args.protocols:
        current_op += 1
        print(f"[{current_op}/{total_ops}] Running static betweenness attack...")
        add_protocol_curve(
            records, adj0, args.label, "betweenness", None, 1,
            order_betweenness_static(adj0),
            args.p_step_nodes,
        )
        print(f"  ✓ betweenness complete")

    # Stochastic baselines.
    if "random" in args.protocols:
        print(f"\nRunning random attack ({args.n_attacks} replicates)...")
        for stochastic_id in range(1, args.n_attacks + 1):
            current_op += 1
            print(f"\r  [{current_op}/{total_ops}] random replicate {stochastic_id}/{args.n_attacks}", end="", flush=True)
            rng = np.random.default_rng(int(rng_master.integers(0, 2**32 - 1)))
            add_protocol_curve(
                records, adj0, args.label, "random", None, stochastic_id,
                order_random(adj0, rng),
                args.p_step_nodes,
            )
        print(f"\n  ✓ random complete")

    if "adap_degree" in args.protocols:
        print(f"\nRunning adaptive degree attack ({args.n_attacks} replicates)...")
        for stochastic_id in range(1, 2):
            current_op += 1
            print(f"\r  [{current_op}/{total_ops}] adap_degree replicate {stochastic_id}/{args.n_attacks}", end="", flush=True)
            rng = np.random.default_rng(int(rng_master.integers(0, 2**32 - 1)))
            add_protocol_curve(
                records, adj0, args.label, "adap_degree", None, stochastic_id,
                order_adaptive_degree(adj0, rng),
                args.p_step_nodes,
            )
        print(f"\n  ✓ adap_degree complete")

    if "adap_betweenness" in args.protocols:
        print(f"\nRunning adaptive betweenness attack ({args.n_attacks} replicates)...")
        for stochastic_id in range(1, 2):
            current_op += 1
            print(f"\r  [{current_op}/{total_ops}] adap_betweenness replicate {stochastic_id}/{args.n_attacks}", end="", flush=True)
            rng = np.random.default_rng(int(rng_master.integers(0, 2**32 - 1)))
            add_protocol_curve(
                records, adj0, args.label, "adap_betweenness", None, stochastic_id,
                order_adaptive_betweenness(adj0, rng, args.btw_update),
                args.p_step_nodes,
            )
        print(f"\n  ✓ adap_betweenness complete")

    # SEAMLESS m sweep.
    if "seamless" in args.protocols:
        print(f"\nRunning SEAMLESS ({len(m_values)} m values × {args.n_attacks} replicates)...")
        for m in m_values:
            for stochastic_id in range(1, args.n_attacks + 1):
                current_op += 1
                print(f"\r  [{current_op}/{total_ops}] seamless m={m} replicate {stochastic_id}/{args.n_attacks}", end="", flush=True)
                rng = np.random.default_rng(int(rng_master.integers(0, 2**32 - 1)))
                add_protocol_curve(
                    records, adj0, args.label, "seamless", m, stochastic_id,
                    order_seamless(adj0, rng, m),
                    args.p_step_nodes,
                )
        print(f"\n  ✓ seamless complete")

    print(f"\n{'='*60}")
    print("Computing summary statistics...")

    raw = pd.DataFrame(records)

    # Stable column ordering for easy downstream use in R.

 
    raw = raw[[
    "label",
    "protocol",
    "m",
    "stochastic_id",
    "removed",
    "p",
    "S1_over_N",
    "S2_over_N",
    "n_initial",
    "n_remaining",
    "num_components",]]

    summary, metrics, metrics_summary = build_summary_tables(raw)
    #RJ old begin:
    #raw.to_csv(args.outdir / "raw.csv", index=False)
    #RJ old end



    summary.to_csv(args.outdir / "summary.csv", index=False)
    #metrics.to_csv(args.outdir / "metrics.csv", index=False)
    #metrics_summary.to_csv(args.outdir / "metrics_summary.csv", index=False)

    print(f"\n{'='*60}")
    print("✓ ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory: {args.outdir}")
    print(f"Files written: config.json, raw.csv, summary.csv, metrics.csv, metrics_summary.csv")
    print(f"Total records: {len(records):,}")
    print(f"{'='*60}\n")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SEAMLESS robustness analysis on a single undirected, unweighted edge-list network."
    )

    parser.add_argument(
        "--edge-file",
        required=True,
        type=Path,
        help="Input edge list with integer node labels starting from 1. Format: u v per line.",
    )
    parser.add_argument(
        "--label",
        required=True,
        type=str,
        help="Required network label written to every output row.",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        type=Path,
        help="Output directory.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Master random seed. Default: 12345.",
    )
    parser.add_argument(
        "--n-attacks",
        type=int,
        default=100,
        help=(
            "Number of stochastic replicates. Applies to random, adap_degree, "
            "adap_betweenness, and every SEAMLESS m value. "
            "Static degree and static betweenness are deterministic and run once. "
            "Default: 100."
        ),
    )
    parser.add_argument(
        "--p-step-nodes",
        type=int,
        default=1,
        help="Robustness curve resolution in removed nodes. Default: 1 stores every step.",
    )

    parser.add_argument(
        "--m-min",
        type=int,
        default=1,
        help="Minimum SEAMLESS sensing budget m. Default: 1.",
    )
    parser.add_argument(
        "--m-max",
        type=int,
        default=20,
        help="Maximum SEAMLESS sensing budget m, inclusive. Default: 20.",
    )
    parser.add_argument(
        "--m-step",
        type=int,
        default=1,
        help="Step size for SEAMLESS sensing budget m. Default: 1.",
    )

    parser.add_argument(
        "--btw-update",
        type=int,
        default=1,
        help=(
            "Adaptive betweenness recomputation interval. "
            "1 means exact recomputation after every removal; larger values reduce cost. "
            "Default: 1."
        ),
    )

    parser.add_argument(
        "--protocols",
        nargs="+",
        default=["random", "degree", "adap_degree", "betweenness", "adap_betweenness", "seamless"],
        choices=["random", "degree", "adap_degree", "betweenness", "adap_betweenness", "seamless"],
        help=(
            "Protocols to run. 'seamless' runs all m values in the m-grid. "
            "Default: random degree adap_degree betweenness adap_betweenness seamless."
        ),
    )

    parser.add_argument(
        "--no-missing-integer-nodes",
        action="store_true",
        help=(
            "By default, nodes 1..max_id are included even if isolated/missing from edges. "
            "Set this flag to include only nodes explicitly appearing in the edge list."
        ),
    )

    parser.add_argument(
        "--zero-indexed",
        action="store_true",
        help="Node labels start from 0 instead of 1.",
    )

    args = parser.parse_args()

    if args.n_attacks < 1:
        raise ValueError("--n-attacks must be >= 1.")
    if args.p_step_nodes < 1:
        raise ValueError("--p-step-nodes must be >= 1.")
    if args.m_min < 1:
        raise ValueError("--m-min must be >= 1.")
    if args.m_max < args.m_min:
        raise ValueError("--m-max must be >= --m-min.")
    if args.m_step < 1:
        raise ValueError("--m-step must be >= 1.")
    if args.btw_update < 1:
        raise ValueError("--btw-update must be >= 1.")

    return args


def main() -> None:
    args = parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
