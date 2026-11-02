
"""
SEAMLESS network evolution from the largest connected component of PPI 9669.

Each parent generates 10 offspring through random structural mutations.
Only offspring with strictly higher mean AUC(S1/N) than their parent
are eligible for selection.

Three retention rates (20%, 25%, and 30%) are examined. Retention
quotas are calculated per parent using half-up rounding, with a minimum
of one eligible offspring for positive rates. If the total exceeds
the population cap of 1,000, quotas are proportionally reduced using
the largest remainder method, with ties broken by parent ID.
Eligible offspring are randomly selected using deterministic seeds.

Evolution terminates after 50 generations or population extinction.
All three rates share the same initial network and 100 attack seeds.

All generated networks, evaluation metrics, and mean dismantling curves
are saved. Checkpoints and evaluation journals support resuming runs.

Master seed: 42.
Dependencies: numpy, numba, joblib, seamless_robustness.py, efficiency.py.
"""

from __future__ import annotations
import csv
import importlib.util
import math
import os
import pickle
import json
import sys
import hashlib
import shutil
import time
from pathlib import Path
import numpy as np
from numba import njit
from joblib import Parallel, delayed, parallel_config

# ============================================================
# 1. Project settings
# ============================================================
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ANCESTOR_PATH = BASE_DIR / "benchmarks" / "PPI" / "9669.edges"
SEAMLESS_PY = BASE_DIR / "seamless_robustness.py"
EFFICIENCY_DIR = BASE_DIR
OUTPUT_ROOT = BASE_DIR / "natural_selection_PPI_9669"
SEED               = 42
N_GENERATIONS        = 50
OFFSPRING_PER_PARENT = 10
MAX_POPULATION       = 1000
RETENTION_RATES      = (0.20, 0.25, 0.30)
EVOLUTION_MODES      = ("add_edges", "grow_nodes", "delete_edges")
EDGES_PER_MUTATION   = 10
GROWTH_NODES         = 10
DELETE_MAX_TRIES     = 50_000
SEAMLESS_M, SEAMLESS_N_ATTACKS = 4, 100
SEAMLESS_P_STEP_NODES          = 1
WORKERS_PER_RUN = 24
RESUME          = True
CSV_FIELDS = [
    "Generation", "Network_ID", "Parent_Network_ID", "Offspring_Number",
    "Evolution_Mode", "Mutation_Completed", "N", "E",
    "Integrated_Efficiency", "Mean_AUC(S1/N)", "Mean_AUC(S2/N)",
    "Parent_Mean_AUC(S1/N)", "Relative_Robustness_Change",
    "Selection_Probability", "Retained", "Mutation_Seed", "Selection_Score",
]
_seamless   = None
_efficiency = None

# ============================================================
# 2. Backend loading
# ============================================================
def load_module(path: Path, name: str):
    if not path.is_file():
        raise FileNotFoundError(f"Required module not found: {path}")
    directory = str(path.resolve().parent)
    if directory not in sys.path:
        sys.path.insert(0, directory)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def load_backends() -> None:
    global _seamless, _efficiency
    _seamless = load_module(SEAMLESS_PY, "natural_selection_seamless_backend")
    @njit
    def _is_connected_dense_kernel(adjacency):
        n = adjacency.shape[0]
        if n == 0:
            return False
        seen  = np.zeros(n, dtype=np.uint8)
        queue = np.empty(n, dtype=np.int64)
        seen[0] = 1; queue[0] = 0; head = 0; tail = 1
        while head < tail:
            u = queue[head]; head += 1
            for v in range(n):
                if adjacency[u, v] != 0 and seen[v] == 0:
                    seen[v] = 1; queue[tail] = v; tail += 1
        return tail == n
    stub = type(sys)("net_analysis")
    stub._is_connected_dense_kernel = _is_connected_dense_kernel
    sys.modules["net_analysis"] = stub
    eff = load_module(EFFICIENCY_DIR / "efficiency.py",
                      "natural_selection_efficiency_backend")
    if not hasattr(eff, "compute_integrated_efficiency"):
        raise AttributeError("compute_integrated_efficiency not found in efficiency.py")
    _efficiency = eff.compute_integrated_efficiency

# ============================================================
# 3. Network utilities
# ============================================================
def is_connected(adj: np.ndarray) -> bool:
    n = len(adj)
    if n == 0:
        return False
    seen = np.zeros(n, dtype=bool); seen[0] = True; stack = [0]
    while stack:
        u = stack.pop()
        for v in np.flatnonzero(adj[u]):
            if not seen[v]:
                seen[v] = True; stack.append(int(v))
    return bool(seen.all())

def load_ancestor(path: Path) -> np.ndarray:
    """Load PPI edge list, deduplicate undirected edges, and take the LCC.

    Original node labels may be arbitrary integers. The retained LCC is
    deterministically relabeled 1..N in ascending original-label order.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Ancestor edge list not found: {path}")
    unique_edges = set()
    all_nodes = set()
    records = loops = 0
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"Expected two node IDs at {path}:{line_number}")
        try:
            u, v = map(int, fields)
        except ValueError as exc:
            raise ValueError(f"Non-integer node ID at {path}:{line_number}") from exc
        records += 1
        all_nodes.update((u, v))
        if u == v:
            loops += 1
            continue
        unique_edges.add((min(u, v), max(u, v)))
    if not all_nodes:
        raise ValueError(f"Empty ancestor edge list: {path}")

    neighbors = {u: set() for u in all_nodes}
    for u, v in unique_edges:
        neighbors[u].add(v)
        neighbors[v].add(u)
    unseen = set(all_nodes)
    components = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        component = {start}
        stack = [start]
        while stack:
            u = stack.pop()
            for v in sorted(neighbors[u]):
                if v in unseen:
                    unseen.remove(v)
                    component.add(v)
                    stack.append(v)
        components.append(component)

    lcc = sorted(components, key=lambda c: (-len(c), min(c)))[0]
    retained = sorted(lcc)
    index = {old: i for i, old in enumerate(retained)}
    adjacency = np.zeros((len(retained), len(retained)), dtype=np.uint8)
    for u, v in unique_edges:
        if u in index and v in index:
            i, j = index[u], index[v]
            adjacency[i, j] = adjacency[j, i] = 1
    assert is_connected(adjacency)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    mapping_path = OUTPUT_ROOT / "ancestor_node_mapping.csv"
    with mapping_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Original_Node_ID", "Ancestor_Node_ID"])
        writer.writerows((old, i + 1) for i, old in enumerate(retained))
    lcc_path = OUTPUT_ROOT / "ancestor_LCC.edges"
    save_edges(adjacency, lcc_path)
    print(f"Original PPI: nodes={len(all_nodes)}, edge_records={records}, "
          f"unique_edges={len(unique_edges)}, duplicates={records-loops-len(unique_edges)}, "
          f"self_loops={loops}, components={len(components)}", flush=True)
    print(f"Ancestor LCC: N={len(retained)}, E={edge_count(adjacency)} "
          f"(saved to {lcc_path})", flush=True)
    return adjacency

def edge_count(adj: np.ndarray) -> int:
    return int(np.count_nonzero(np.triu(adj, k=1)))

def mutate(parent: np.ndarray, mode: str,
           rng: np.random.Generator) -> tuple[np.ndarray, bool]:
    n     = len(parent)
    child = parent.copy()
    if mode == "add_edges":
        rows, cols = np.triu_indices(n, k=1)
        absent = np.flatnonzero(child[rows, cols] == 0)
        if len(absent) < EDGES_PER_MUTATION:
            return child, False
        chosen = rng.choice(absent, EDGES_PER_MUTATION, replace=False)
        child[rows[chosen], cols[chosen]] = 1
        child[cols[chosen], rows[chosen]] = 1
        return child, True
    if mode == "grow_nodes":
        new_n   = GROWTH_NODES
        total_e = max(new_n, int(round(edge_count(parent) * new_n / n)))
        child   = np.zeros((n + new_n, n + new_n), dtype=np.uint8)
        child[:n, :n] = parent
        base, rem = divmod(total_e, new_n)
        counts = np.full(new_n, base, dtype=int)
        if rem:
            counts[rng.choice(new_n, rem, replace=False)] += 1
        for offset, count in enumerate(counts):
            node = n + offset
            if count > node:
                return parent.copy(), False
            nbrs = rng.choice(node, int(count), replace=False)
            child[node, nbrs] = child[nbrs, node] = 1
        return child, True
    if mode == "delete_edges":
        deleted = attempts = 0
        while deleted < EDGES_PER_MUTATION and attempts < DELETE_MAX_TRIES:
            attempts += 1
            edges = np.argwhere(np.triu(child, k=1) > 0)
            if len(edges) <= n - 1:
                break
            u, v = edges[int(rng.integers(len(edges)))]
            child[u, v] = child[v, u] = 0
            if is_connected(child):
                deleted += 1
            else:
                child[u, v] = child[v, u] = 1
        return (child, True) if deleted == EDGES_PER_MUTATION else (parent.copy(), False)
    raise ValueError(f"Unknown evolution mode: {mode}")

# ============================================================
# 4. Deterministic seeds & identities
# ============================================================
def stable_seed(namespace: str, *parts) -> int:
    payload = json.dumps([SEED, namespace, *parts], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:16], "big")

def child_identity(parent_id: str, generation: int, number: int) -> str:
    payload = json.dumps([parent_id, number], separators=(",", ":"))
    digest  = hashlib.sha256(payload.encode()).hexdigest()[:32]
    return f"g{generation:04d}_{digest}"

def make_candidate(parent: dict, generation: int, number: int) -> dict:
    identity     = child_identity(parent["id"], generation, number)
    seed         = stable_seed("mutation", parent["id"], number)
    mutation_rng = np.random.default_rng(seed)
    mode         = str(mutation_rng.choice(EVOLUTION_MODES))
    adj, done    = mutate(parent["adj"], mode, mutation_rng)
    score        = float(np.random.default_rng(
                       stable_seed("selection", parent["id"], number)).random())
    return {"id": identity, "parent": parent, "number": number, "mode": mode,
            "completed": done, "adj": adj, "mutation_seed": seed,
            "selection_score": score}

# ============================================================
# 5. Selection
# ============================================================
def select_per_parent(candidates, evaluated, rate):
    if not 0 <= rate <= 1:
        raise ValueError("Retention rate must be in [0,1].")
    pools = {}
    for i, c in enumerate(candidates):
        if evaluated[i][0] > c["parent"]["auc1"]:
            pools.setdefault(c["parent"]["id"], []).append(i)
    quotas = {pid: (min(len(idx), max(1, int(math.floor(rate * len(idx) + 0.5))))
                    if rate > 0 else 0)
              for pid, idx in pools.items()}
    total = sum(quotas.values())
    if total > MAX_POPULATION:
        alloc, rems = {}, {}
        for pid, q in quotas.items():
            alloc[pid], rems[pid] = divmod(q * MAX_POPULATION, total)
        remaining = MAX_POPULATION - sum(alloc.values())
        for pid in sorted(quotas, key=lambda p: (-rems[p], p))[:remaining]:
            alloc[pid] += 1
        quotas = alloc
    kept, probs = set(), {}
    for pid, idx in pools.items():
        q      = quotas[pid]
        ranked = sorted(idx, key=lambda i: (candidates[i]["selection_score"],
                                            candidates[i]["id"]))
        kept.update(ranked[:q])
        probs.update({i: q / len(idx) for i in idx})
    assert len(kept) == min(total, MAX_POPULATION)
    return kept, probs

# ============================================================
# 6. Evaluation
# ============================================================
def curve_arrays(curve, n: int):
    if isinstance(curve, (list, tuple)) and curve and isinstance(curve[0], dict):
        cols = {k: np.asarray([r[k] for r in curve]) for k in curve[0]}
    elif hasattr(curve, "keys"):
        cols = {k: np.asarray(curve[k]) for k in curve.keys()}
    elif isinstance(curve, np.ndarray) and curve.dtype.names:
        cols = {k: np.asarray(curve[k]) for k in curve.dtype.names}
    else:
        raise TypeError("Unsupported robustness_curve format.")
    s1 = np.asarray(cols["S1_over_N"], float).reshape(-1)
    s2 = np.asarray(cols["S2_over_N"], float).reshape(-1)
    fraction = None
    for key in ("removed_fraction", "fraction_removed", "p", "q", "f"):
        if key in cols:
            fraction = np.asarray(cols[key], float).reshape(-1); break
    if fraction is None:
        for key in ("removed_nodes", "n_removed", "removed", "k"):
            if key in cols:
                fraction = np.asarray(cols[key], float).reshape(-1) / n; break
    if fraction is None:
        removed = np.arange(0, n + 1, SEAMLESS_P_STEP_NODES)
        if removed[-1] != n:
            removed = np.r_[removed, n]
        if len(removed) != len(s1):
            raise ValueError("Cannot infer removed fraction.")
        fraction = removed / n
    if (len(fraction) != len(s1) or len(s1) != len(s2) or not len(s1) or
            not np.isfinite(np.r_[fraction, s1, s2]).all() or
            np.any(np.diff(fraction) < 0) or
            fraction.min() < 0 or fraction.max() > 1):
        raise ValueError("Invalid robustness curve arrays.")
    return fraction, s1, s2

def save_mean_curve(path: Path, fraction, s1, s2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".npz.tmp")
    with tmp.open("wb") as fh:
        np.savez_compressed(fh, removed_fraction=fraction,
                            mean_s1_over_n=s1, mean_s2_over_n=s2)
        fh.flush(); os.fsync(fh.fileno())
    tmp.replace(path)

def mean_curve_exists(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as d:
            f, s1, s2 = d["removed_fraction"], d["mean_s1_over_n"], d["mean_s2_over_n"]
            return (f.ndim == s1.ndim == s2.ndim == 1 and
                    len(f) == len(s1) == len(s2) > 0 and
                    np.isfinite(np.r_[f, s1, s2]).all())
    except (OSError, ValueError, KeyError, EOFError):
        return False

def evaluate(index: int, adj: np.ndarray, seeds: list[int],
             curve_path: Path):
    node_map = {u + 1: {int(v) + 1 for v in np.flatnonzero(adj[u])}
                for u in range(len(adj))}
    if not seeds:
        raise ValueError("At least one attack seed is required.")
    auc_s1, auc_s2 = [], []
    fraction = sum_s1 = sum_s2 = None
    for seed in seeds:
        order = _seamless.order_seamless(node_map, np.random.default_rng(seed), SEAMLESS_M)
        curve = _seamless.robustness_curve(node_map, order, SEAMLESS_P_STEP_NODES)
        auc_s1.append(float(_seamless.auc_from_curve(curve, "S1_over_N")))
        auc_s2.append(float(_seamless.auc_from_curve(curve, "S2_over_N")))
        grid, s1, s2 = curve_arrays(curve, len(adj))
        if fraction is None:
            fraction, sum_s1, sum_s2 = grid.copy(), s1.copy(), s2.copy()
        else:
            if fraction.shape != grid.shape or not np.allclose(fraction, grid):
                raise ValueError("Attack replicates use different removal grids.")
            sum_s1 += s1; sum_s2 += s2
    efficiency = float(_efficiency(np.asarray(adj, dtype=np.float64), L_type="RW"))
    save_mean_curve(curve_path, fraction, sum_s1 / len(seeds), sum_s2 / len(seeds))
    return index, float(np.mean(auc_s1)), float(np.mean(auc_s2)), efficiency

# ============================================================
# 7. Worker (self-contained for parallel execution)
# ============================================================
_WORKER_SOURCE = """
import importlib.util, os, sys
from pathlib import Path
import numpy as np
from numba import njit
_seamless   = None
_efficiency = None

def _load_module(path, name):
    path = Path(path)
    directory = str(path.resolve().parent)
    if directory not in sys.path:
        sys.path.insert(0, directory)
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def _load_backends(seamless_py, efficiency_dir):
    global _seamless, _efficiency
    _seamless = _load_module(seamless_py, "_w_seamless")
    @njit
    def _is_connected_dense_kernel(adjacency):
        n = adjacency.shape[0]
        if n == 0: return False
        seen  = np.zeros(n, dtype=np.uint8)
        queue = np.empty(n, dtype=np.int64)
        seen[0] = 1; queue[0] = 0; head = 0; tail = 1
        while head < tail:
            u = queue[head]; head += 1
            for v in range(n):
                if adjacency[u, v] != 0 and seen[v] == 0:
                    seen[v] = 1; queue[tail] = v; tail += 1
        return tail == n
    stub = type(sys)("net_analysis")
    stub._is_connected_dense_kernel = _is_connected_dense_kernel
    sys.modules["net_analysis"] = stub
    eff = _load_module(str(Path(efficiency_dir) / "efficiency.py"), "_w_efficiency")
    _efficiency = eff.compute_integrated_efficiency

def _curve_arrays(curve, n, step):
    if isinstance(curve, (list, tuple)) and curve and isinstance(curve[0], dict):
        cols = {k: np.asarray([r[k] for r in curve]) for k in curve[0]}
    elif hasattr(curve, "keys"):
        cols = {k: np.asarray(curve[k]) for k in curve.keys()}
    elif isinstance(curve, np.ndarray) and curve.dtype.names:
        cols = {k: np.asarray(curve[k]) for k in curve.dtype.names}
    else:
        raise TypeError("Unsupported robustness_curve format.")
    s1 = np.asarray(cols["S1_over_N"], float).reshape(-1)
    s2 = np.asarray(cols["S2_over_N"], float).reshape(-1)
    fraction = None
    for key in ("removed_fraction", "fraction_removed", "p", "q", "f"):
        if key in cols:
            fraction = np.asarray(cols[key], float).reshape(-1); break
    if fraction is None:
        for key in ("removed_nodes", "n_removed", "removed", "k"):
            if key in cols:
                fraction = np.asarray(cols[key], float).reshape(-1) / n; break
    if fraction is None:
        removed = np.arange(0, n + 1, step)
        if removed[-1] != n: removed = np.r_[removed, n]
        fraction = removed / n
    return fraction, s1, s2

def _save_mean_curve(path, fraction, s1, s2):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".npz.tmp")
    with tmp.open("wb") as fh:
        np.savez_compressed(fh, removed_fraction=fraction,
                            mean_s1_over_n=s1, mean_s2_over_n=s2)
        fh.flush(); os.fsync(fh.fileno())
    tmp.replace(path)

def run(index, adj, seeds, curve_path, seamless_py, efficiency_dir, m, step):
    global _seamless, _efficiency
    if _seamless is None:
        _load_backends(seamless_py, efficiency_dir)
    node_map = {u + 1: {int(v) + 1 for v in np.flatnonzero(adj[u])}
                for u in range(len(adj))}
    auc_s1, auc_s2 = [], []
    fraction = sum_s1 = sum_s2 = None
    for seed in seeds:
        order = _seamless.order_seamless(node_map, np.random.default_rng(seed), m)
        curve = _seamless.robustness_curve(node_map, order, step)
        auc_s1.append(float(_seamless.auc_from_curve(curve, "S1_over_N")))
        auc_s2.append(float(_seamless.auc_from_curve(curve, "S2_over_N")))
        grid, s1, s2 = _curve_arrays(curve, len(adj), step)
        if fraction is None:
            fraction, sum_s1, sum_s2 = grid.copy(), s1.copy(), s2.copy()
        else:
            sum_s1 += s1; sum_s2 += s2
    eff = float(_efficiency(np.asarray(adj, dtype=np.float64), L_type="RW"))
    _save_mean_curve(curve_path, fraction, sum_s1 / len(seeds), sum_s2 / len(seeds))
    return index, float(np.mean(auc_s1)), float(np.mean(auc_s2)), eff
"""

def evaluate_in_process(index, adj, seeds, curve_path):
    """Execute in a loky worker process; all arguments are plain Python objects."""
    import types
    module_name = "_seamless_evolution_worker"
    if module_name not in sys.modules:
        mod = types.ModuleType(module_name)
        exec(compile(_WORKER_SOURCE, "<worker>", "exec"), mod.__dict__)
        sys.modules[module_name] = mod
    return sys.modules[module_name].run(
        index, adj, seeds, curve_path,
        str(SEAMLESS_PY), str(EFFICIENCY_DIR),
        SEAMLESS_M, SEAMLESS_P_STEP_NODES,
    )

# ============================================================
# 8. IO helpers
# ============================================================
def save_edges(adj: np.ndarray, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for u, v in np.argwhere(np.triu(adj, k=1) > 0):
            fh.write(f"{int(u)+1} {int(v)+1}\n")

def write_csv(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader(); w.writerows(rows)
    tmp.replace(path)

def save_checkpoint(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".pkl.tmp")
    with tmp.open("wb") as fh:
        pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)

def read_evaluation_journal(path: Path) -> dict[int, tuple]:
    evaluated = {}
    if not path.is_file():
        return evaluated
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
                evaluated[int(r["index"])] = (
                    float(r["auc1"]), float(r["auc2"]), float(r["efficiency"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return evaluated

def append_evaluation_journal(path: Path, index: int,
                              values: tuple) -> None:
    record = {"index": index, "auc1": values[0],
              "auc2": values[1], "efficiency": values[2]}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, separators=(",", ":")) + "\n")
        fh.flush(); os.fsync(fh.fileno())

# ============================================================
# 9. Ancestor caching
# ============================================================
def cached_ancestor(ancestor, attack_seeds):
    cache_path = OUTPUT_ROOT / "_ancestor_cache.pkl"
    curve_path = OUTPUT_ROOT / "_ancestor_mean_curve.npz"
    signature  = {
        "version": 1,
        "network": hashlib.sha256(ancestor.tobytes()).hexdigest(),
        "shape": ancestor.shape, "master_seed": SEED,
        "attack_seeds": attack_seeds,
        "m": SEAMLESS_M, "step": SEAMLESS_P_STEP_NODES,
        "backend_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (SEAMLESS_PY, EFFICIENCY_DIR / "efficiency.py")},
    }
    if cache_path.is_file():
        with cache_path.open("rb") as fh:
            cache = pickle.load(fh)
        if cache.get("signature") != signature:
            raise RuntimeError(
                "Ancestor cache settings changed; use a new OUTPUT_ROOT.")
        if mean_curve_exists(curve_path):
            return tuple(cache["metrics"])
    _, auc1, auc2, eff = evaluate(0, ancestor, attack_seeds, curve_path)
    metrics = (auc1, auc2, eff)
    save_checkpoint(cache_path, {"signature": signature, "metrics": metrics})
    return metrics

# ============================================================
# 10. Generation processing
# ============================================================
def process_generation(rate, generation, candidates, evaluated, rows, parents,
                       generation_stats, rate_dir, csv_path, checkpoint_path,
                       settings):
    journal_path = rate_dir / f"_generation_{generation:04d}_evaluated.jsonl"
    total        = len(candidates)
    evaluated    = {i: v for i, v in evaluated.items() if 0 <= i < total}
    curve_dir    = rate_dir / "mean_curves"
    missing      = [i for i in range(total)
                    if i not in evaluated or
                    not mean_curve_exists(curve_dir / f"{candidates[i]['id']}.npz")]
    completed    = total - len(missing)
    last_update  = time.monotonic()
    print(f"r={rate:.0%} | generation {generation}/{N_GENERATIONS} | "
          f"evaluated={completed}/{total}", end="\r", flush=True)
    jobs = (
        delayed(evaluate_in_process)(
            i, candidates[i]["adj"], candidates[i]["parent"]["seeds"],
            str(curve_dir / f"{candidates[i]['id']}.npz"),
        )
        for i in missing
    )
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(
            n_jobs=WORKERS_PER_RUN, return_as="generator_unordered",
            batch_size=1, pre_dispatch=WORKERS_PER_RUN, max_nbytes=None,
        )(jobs)
        for i, auc1, auc2, eff in results:
            append_evaluation_journal(journal_path, i, (auc1, auc2, eff))
            evaluated[i] = (auc1, auc2, eff)
            completed   += 1
            now = time.monotonic()
            if now - last_update >= 5 or completed == total:
                print(f"r={rate:.0%} | generation {generation}/{N_GENERATIONS} | "
                      f"evaluated={completed}/{total}", end="\r", flush=True)
                last_update = now
    kept, probs = select_per_parent(candidates, evaluated, rate)
    print(f"r={rate:.0%} | generation {generation}/{N_GENERATIONS} | "
          f"parents={len(parents)}, candidates={total}, retained={len(kept)}",
          flush=True)
    next_parents, new_rows = [], []
    for i, c in enumerate(candidates):
        auc1, auc2, eff = evaluated[i]
        parent_auc = c["parent"]["auc1"]
        relative   = (auc1 - parent_auc) / parent_auc if parent_auc else 0.0
        retained   = i in kept
        new_rows.append({
            "Generation": generation, "Network_ID": c["id"],
            "Parent_Network_ID": c["parent"]["id"],
            "Offspring_Number": c["number"], "Evolution_Mode": c["mode"],
            "Mutation_Completed": c["completed"], "N": len(c["adj"]),
            "E": edge_count(c["adj"]),
            "Integrated_Efficiency": f"{eff:.12g}",
            "Mean_AUC(S1/N)": f"{auc1:.12g}", "Mean_AUC(S2/N)": f"{auc2:.12g}",
            "Parent_Mean_AUC(S1/N)": f"{parent_auc:.12g}",
            "Relative_Robustness_Change": f"{relative:.12g}",
            "Selection_Probability": f"{probs.get(i, 0.0):.12g}",
            "Retained": retained,
            "Mutation_Seed": str(c["mutation_seed"]),
            "Selection_Score": repr(c["selection_score"]),
        })
        if retained:
            next_parents.append({"id": c["id"], "adj": c["adj"],
                                 "auc1": auc1, "seeds": c["parent"]["seeds"]})
    rows.extend(new_rows)
    generation_stats.append((generation, len(parents), total, len(next_parents)))
    parents = next_parents
    write_csv(csv_path, rows)
    save_checkpoint(checkpoint_path, {
        "phase": "ready", "settings": settings, "rows": rows,
        "parents": parents, "next_generation": generation + 1,
        "generation_stats": generation_stats, "extinct": not parents,
    })
    try:
        journal_path.unlink()
    except FileNotFoundError:
        pass
    return rows, parents, generation_stats, not parents

# ============================================================
# 11. Per-rate evolution loop
# ============================================================
def run_rate(rate: float, ancestor: np.ndarray, attack_seeds: list[int],
             ancestor_metrics: tuple):
    rate_dir      = OUTPUT_ROOT / f"{rate:.0%}_{N_GENERATIONS}gen"
    network_dir   = rate_dir / "networks"
    rate_dir.mkdir(parents=True, exist_ok=True)
    network_dir.mkdir(parents=True, exist_ok=True)
    csv_path        = rate_dir / "evolution_metrics.csv"
    checkpoint_path = rate_dir / "_checkpoint.pkl"
    summary_path    = rate_dir / "evolution_summary.txt"
    complete_path   = rate_dir / "_COMPLETE.txt"
    settings = {
        "checkpoint_version": 5, "max_population": MAX_POPULATION,
        "mean_curves": "npz_mean_only_v1",
        "selection": "per_parent_half_up_largest_remainder_parent_id_v1",
        "seed_scheme": "sha256_lineage_separate_streams_v1",
        "r": rate, "master_seed": SEED,
        "ancestor_file": str(ANCESTOR_PATH.relative_to(BASE_DIR)),
        "ancestor_sha256": hashlib.sha256(ancestor.tobytes()).hexdigest(),
        "generations": N_GENERATIONS, "offspring": OFFSPRING_PER_PARENT,
        "attacks": SEAMLESS_N_ATTACKS, "m": SEAMLESS_M,
        "attack_step": SEAMLESS_P_STEP_NODES, "modes": EVOLUTION_MODES,
        "mutation_edges": EDGES_PER_MUTATION, "growth_nodes": GROWTH_NODES,
        "delete_max_tries": DELETE_MAX_TRIES,
        "backend_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (SEAMLESS_PY, EFFICIENCY_DIR / "efficiency.py")},
    }
    if not RESUME and (checkpoint_path.exists() or csv_path.exists()):
        raise RuntimeError(
            "Output already exists. Choose a new OUTPUT_ROOT for a fresh run.")
    if csv_path.exists() and not checkpoint_path.exists():
        raise RuntimeError(
            f"Existing results have no checkpoint: {csv_path}. "
            "Restore the checkpoint or use a new directory.")
    current = None
    if RESUME and checkpoint_path.is_file():
        with checkpoint_path.open("rb") as fh:
            current = pickle.load(fh)
        if current.get("settings") != settings:
            raise RuntimeError(
                f"Checkpoint settings mismatch for r={rate:.0%}: {checkpoint_path}")
        rows, parents       = current["rows"], current["parents"]
        next_generation     = current["next_generation"]
        generation_stats    = current["generation_stats"]
        extinct             = current["extinct"]
        if current.get("phase") == "ready" and (
                not parents or next_generation > N_GENERATIONS):
            return rate, len(rows), len(parents), not parents
    else:
        ancestor_id = "g0000_n000000"
        save_edges(ancestor, network_dir / f"{ancestor_id}.edges")
        auc1, auc2, eff = ancestor_metrics
        rows = [{"Generation": 0, "Network_ID": ancestor_id,
                 "Parent_Network_ID": "", "Offspring_Number": "",
                 "Evolution_Mode": "initial_PPI_9669", "Mutation_Completed": True,
                 "N": len(ancestor), "E": edge_count(ancestor),
                 "Integrated_Efficiency": f"{eff:.12g}",
                 "Mean_AUC(S1/N)": f"{auc1:.12g}",
                 "Mean_AUC(S2/N)": f"{auc2:.12g}",
                 "Parent_Mean_AUC(S1/N)": "",
                 "Relative_Robustness_Change": "",
                 "Selection_Probability": "", "Retained": ""}]
        parents          = [{"id": ancestor_id, "adj": ancestor,
                              "auc1": auc1, "seeds": attack_seeds}]
        next_generation  = 1
        generation_stats = []
        extinct          = False
        write_csv(csv_path, rows)
        save_checkpoint(checkpoint_path, {
            "phase": "ready", "settings": settings, "rows": rows,
            "parents": parents, "next_generation": 1,
            "generation_stats": generation_stats, "extinct": False,
        })
    if current is None:
        curve_path = rate_dir / "mean_curves" / "g0000_n000000.npz"
        curve_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(OUTPUT_ROOT / "_ancestor_mean_curve.npz", curve_path)

    if current and current.get("phase") == "evaluating":
        generation  = current["current_generation"]
        candidates  = current["candidates"]
        evaluated   = read_evaluation_journal(
            rate_dir / f"_generation_{generation:04d}_evaluated.jsonl")
        rows, parents, generation_stats, extinct = process_generation(
            rate, generation, candidates, evaluated, rows, parents,
            generation_stats, rate_dir, csv_path, checkpoint_path, settings)
        next_generation = generation + 1
    for generation in range(next_generation, N_GENERATIONS + 1):
        if not parents:
            extinct = True; break
        candidates = []
        for parent in parents:
            for num in range(1, OFFSPRING_PER_PARENT + 1):
                c = make_candidate(parent, generation, num)
                save_edges(c["adj"], network_dir / f"{c['id']}.edges")
                candidates.append(c)
        save_checkpoint(checkpoint_path, {
            "phase": "evaluating", "settings": settings, "rows": rows,
            "parents": parents, "next_generation": generation,
            "current_generation": generation, "candidates": candidates,
            "generation_stats": generation_stats, "extinct": False,
        })
        rows, parents, generation_stats, extinct = process_generation(
            rate, generation, candidates, {}, rows, parents,
            generation_stats, rate_dir, csv_path, checkpoint_path, settings)
        if extinct:
            break
    summary = [
        f"SEAMLESS branching evolution | r={rate:.0%}",
        f"Master seed: {SEED}",
        f"Population cap: {MAX_POPULATION}",
        "Selection: parent quotas with largest remainder; seeded ranking",
        f"Completed generations: {generation_stats[-1][0] if generation_stats else 0}",
        f"Saved/evaluated networks: {len(rows)}",
        f"Retained offspring: {sum(r['Retained'] is True for r in rows[1:])}",
        "Mean curves: mean_curves/<Network_ID>.npz",
        f"Natural extinction: {'Yes' if extinct or not parents else 'No'}",
        "", "Generation\tParents\tGenerated\tRetained",
    ]
    summary.extend("\t".join(map(str, s)) for s in generation_stats)
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    complete_path.write_text("Completed\n", encoding="utf-8")
    return rate, len(rows), len(parents), extinct

# ============================================================
# 12. Entry point
# ============================================================
def main():
    print(f"Initializing: up to {WORKERS_PER_RUN} independent processes",
          flush=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    load_backends()
    ancestor = load_ancestor(ANCESTOR_PATH)
    attack_rng  = np.random.default_rng(stable_seed("attack"))
    attack_seeds = [int(attack_rng.integers(0, 2**32 - 1))
                    for _ in range(SEAMLESS_N_ATTACKS)]
    ancestor_metrics = cached_ancestor(ancestor, attack_seeds)
    results = []
    for rate in RETENTION_RATES:
        results.append(run_rate(rate, ancestor, attack_seeds, ancestor_metrics))
    for rate, n_nets, n_parents, extinct in sorted(results):
        print(f"r={rate:.0%}: networks={n_nets}, "
              f"final parents={n_parents}, extinct={extinct}")
    print(f"Results saved in: {OUTPUT_ROOT}/")
if __name__ == "__main__":
    main()
