"""Figure SI1: serial runtime benchmark of six original dismantling methods."""

# ============================================================
# 1. Imports and Configuration
# ============================================================
import gc
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from seamless_robustness import (
    read_integer_edgelist, adjacency_to_networkx, edge_count,
    order_random, order_degree_static, order_adaptive_degree,
    order_betweenness_static, order_adaptive_betweenness, order_seamless,
)

ROOT = Path.cwd()
BENCHMARK_DIR = ROOT / "benchmarks"
OUTPUT_DIR = ROOT / "results_synthetic"
OUTPUT_CSV = OUTPUT_DIR / "runtime_backup.csv"

NETWORK_TYPES = ["BA", "ER", "HSBM4", "LFR", "RGG", "SBM4",
                 "SBM8", "SBM16", "SF2", "WS"]
METHOD_ORDER = ["random", "degree", "adap_degree", "betweenness",
                "adap_betweenness", "seamless"]
REPEATS = {"random": 100, "degree": 1, "adap_degree": 10,
           "betweenness": 1, "adap_betweenness": 10, "seamless": 100}
ATTACK_MASTER_SEED = 12345
SEAMLESS_M = 4
BETWEENNESS_UPDATE_EVERY = 1
ZERO_INDEXED = True
INCLUDE_MISSING_INTEGER_NODES = True
RESUME_FROM_EXISTING_CSV = True

BASE_COLUMNS = ["network_type", "instance", "N", "M", "method", "m",
                "n_repeats", "mean_runtime_s"]
OUTPUT_COLUMNS = BASE_COLUMNS + ["cumulative_runtime_s", "cumulative_runtime_min"]


# ============================================================
# 2. Network Loading
# ============================================================
def network_files(network_type):
    folder = BENCHMARK_DIR / f"{network_type}_N256"
    if not folder.is_dir():
        print(f"WARNING: missing directory: {folder}")
        return []
    # Preserve the original benchmark's filename sorting exactly.
    files = sorted(folder.glob("*.edges"),
                   key=lambda p: int(p.stem.rsplit("r", 1)[1]))
    if not files:
        print(f"WARNING: no .edges files in: {folder}")
    return files


def load_initial_lcc(path):
    adj = read_integer_edgelist(
        path, include_missing_integer_nodes=INCLUDE_MISSING_INTEGER_NODES,
        zero_indexed=ZERO_INDEXED,
    )
    graph = adjacency_to_networkx(adj)
    largest_cc = max(nx.connected_components(graph), key=len)
    # Keep the original iteration and adjacency construction semantics.
    return {u: {v for v in adj[u] if v in largest_cc} for u in largest_cc}


# ============================================================
# 3. Attack Methods and Random Seeds
# ============================================================
def create_repeat_rng(network_type_index, instance_index,
                      method_index, repeat_index):
    """Identical five-part SeedSequence used by the 1507-line benchmark."""
    return np.random.default_rng(np.random.SeedSequence([
        ATTACK_MASTER_SEED, network_type_index, instance_index,
        method_index, repeat_index,
    ]))


def run_attack(method, adj0, rng):
    if method == "random":
        return order_random(adj0, rng)
    if method == "degree":
        return order_degree_static(adj0)
    if method == "adap_degree":
        return order_adaptive_degree(adj0, rng)
    if method == "betweenness":
        return order_betweenness_static(adj0)
    if method == "adap_betweenness":
        return order_adaptive_betweenness(
            adj0, rng, update_every=BETWEENNESS_UPDATE_EVERY)
    if method == "seamless":
        return order_seamless(adj0, rng, sensing_budget_m=SEAMLESS_M)
    raise ValueError(f"Unknown method: {method}")


# ============================================================
# 4. Runtime Benchmark
# ============================================================
def benchmark_method(method, adj0, network_type_index, instance_index):
    runtimes = []
    repeats = REPEATS[method]
    method_index = METHOD_ORDER.index(method)
    for repeat_index in range(repeats):
        rng = create_repeat_rng(network_type_index, instance_index,
                                method_index, repeat_index)
        gc.collect()  # Outside timed region, as in the original benchmark.
        print(f"\r      repeat {repeat_index + 1}/{repeats}", end="", flush=True)
        start = time.perf_counter()
        order = run_attack(method, adj0, rng)
        elapsed = time.perf_counter() - start
        if len(order) != len(adj0) or len(set(order)) != len(adj0):
            raise RuntimeError(f"Invalid node-removal order: {method}")
        runtimes.append(elapsed)
    print("\r" + " " * 45 + "\r", end="", flush=True)
    return repeats, float(np.mean(runtimes))


# ============================================================
# 5. Results and Checkpoint
# ============================================================
def load_checkpoint():
    if not (RESUME_FROM_EXISTING_CSV and OUTPUT_CSV.exists()):
        return pd.DataFrame(columns=BASE_COLUMNS)
    df = pd.read_csv(OUTPUT_CSV)
    missing = set(BASE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Existing CSV missing columns: {sorted(missing)}")
    if df.duplicated(["network_type", "instance", "method"]).any():
        raise ValueError("Existing CSV has duplicate network-method rows")
    return df[BASE_COLUMNS].copy()


def save_checkpoint(df):
    # Preserve the original stable network/instance/method ordering.
    work = df[BASE_COLUMNS].copy()
    work["_type"] = work["network_type"].map(
        {name: i for i, name in enumerate(NETWORK_TYPES)}).fillna(len(NETWORK_TYPES))
    work["_method"] = work["method"].map(
        {name: i for i, name in enumerate(METHOD_ORDER)}).fillna(len(METHOD_ORDER))
    work = work.sort_values(["_type", "instance", "_method"],
                            kind="mergesort").reset_index(drop=True)
    work["mean_runtime_s"] = pd.to_numeric(work["mean_runtime_s"], errors="coerce")
    work["cumulative_runtime_s"] = work.groupby(
        "method", sort=False)["mean_runtime_s"].cumsum()
    work["cumulative_runtime_min"] = work["cumulative_runtime_s"] / 60.0
    work[OUTPUT_COLUMNS].to_csv(OUTPUT_CSV, index=False)
    return work[BASE_COLUMNS].copy()


def already_done(df, network_type, instance, method):
    return bool(((df["network_type"] == network_type) &
                 (df["instance"] == instance) &
                 (df["method"] == method)).any())


# ============================================================
# 6. Main
# ============================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = {name: network_files(name) for name in NETWORK_TYPES}
    total_networks = sum(len(paths) for paths in selected.values())
    if not total_networks:
        raise RuntimeError(f"No benchmark networks found under {BENCHMARK_DIR}")
    df = load_checkpoint()
    total_jobs = total_networks * len(METHOD_ORDER)
    print("=" * 65)
    print("Serial runtime benchmark")
    print(f"Networks: {total_networks} | Jobs: {total_jobs} | Existing: {len(df)}")
    print(f"SEAMLESS m={SEAMLESS_M} | Seed={ATTACK_MASTER_SEED}")
    print(f"Output: {OUTPUT_CSV}")
    print("=" * 65)

    job = 0
    started = time.perf_counter()
    for type_index, network_type in enumerate(NETWORK_TYPES):
        for instance_index, path in enumerate(selected[network_type]):
            # Avoid reading files if every method is already checkpointed.
            if all(already_done(df, network_type, path.stem, method)
                   for method in METHOD_ORDER):
                job += len(METHOD_ORDER)
                print(f"{network_type}/{path.stem}: all methods already complete")
                continue
            adj0 = load_initial_lcc(path)
            n_nodes, n_edges = len(adj0), edge_count(adj0)
            print(f"\n{network_type}/{path.stem}: N={n_nodes}, M={n_edges}")
            for method in METHOD_ORDER:
                job += 1
                if already_done(df, network_type, path.stem, method):
                    print(f"  [{job}/{total_jobs}] {method}: SKIPPED")
                    continue
                print(f"  [{job}/{total_jobs}] {method}: {REPEATS[method]} repeats")
                n_repeats, mean_runtime = benchmark_method(
                    method, adj0, type_index, instance_index)
                df.loc[len(df)] = {
                    "network_type": network_type, "instance": path.stem,
                    "N": n_nodes, "M": n_edges, "method": method,
                    "m": SEAMLESS_M if method == "seamless" else np.nan,
                    "n_repeats": n_repeats, "mean_runtime_s": mean_runtime,
                }
                df = save_checkpoint(df)
                print(f"      mean runtime = {mean_runtime:.6f} s | saved")

    df = save_checkpoint(df)
    summary = df.groupby("method")["mean_runtime_s"].sum()
    print("\n" + "=" * 65)
    print("COMPLETE | Cumulative mean runtime by method")
    for method in METHOD_ORDER:
        seconds = float(summary.get(method, 0.0))
        print(f"  {method:<18} {seconds:>12.6f} s ({seconds / 60:.4f} min)")
    print(f"Session wall time: {(time.perf_counter() - started) / 60:.2f} min")
    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
