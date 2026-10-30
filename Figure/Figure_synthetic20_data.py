#Figure 2 data
"""Serial SEAMLESS robustness analysis: 10 network types × 20 replicates × 100 attacks.
Requires the original, unchanged seamless_robustness.py alongside this script.
Output: results_synthetic/summary_synthetic_backup.csv (16 columns).
"""

# ============================================================
# 1. Imports and Configuration
# ============================================================
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd
from seamless_robustness import (adjacency_to_networkx, connected_component_sizes,
                                 copy_adjacency, order_seamless,
                                 read_integer_edgelist, remove_node_inplace)

BASE_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
BENCHMARKS_DIR = BASE_DIR / 'benchmarks'
OUTPUT_FILE = BASE_DIR / 'results_synthetic' / 'summary_synthetic_backup.csv'
MASTER_SEED, N_ATTACKS, SEAMLESS_M = 12345, 100, 4
TYPES = ('BA', 'ER', 'HSBM4', 'LFR', 'RGG', 'SBM4', 'SBM8', 'SBM16', 'SF2', 'WS')
KEYS = ['label', 'network_type', 'network_replicate', 'protocol', 'm', 'removed', 'p']
COLUMNS = KEYS + ['mean_S1_over_N', 'sd_S1_over_N', 'sem_S1_over_N',
                  'mean_S2_over_N', 'sd_S2_over_N', 'sem_S2_over_N',
                  'n', 'removed_node_label', 'seed_node_labels']

# ============================================================
# 2. Network File Discovery
# ============================================================
def network_specs():
    for kind in TYPES:
        for rep in range(1, 21):
            label = f'{kind}_N256_{kind}_r{rep}'
            path = BENCHMARKS_DIR / f'{kind}_N256' / f'{kind}_r{rep}.edges'
            if not path.exists():
                matches = [p for p in path.parent.glob('*')
                           if p.is_file() and p.stem.lower() == path.stem.lower()]
                if len(matches) != 1:
                    raise FileNotFoundError(f'Cannot uniquely locate {label}: {path}')
                path = matches[0]
            yield kind, rep, label, path

# ============================================================
# 3. Record Seeds Without Changing the Core Algorithm
# ============================================================
class SeedRecordingRNG:
    def __init__(self, generator):
        self.generator, self.seed_history = generator, []

    def choice(self, a, size=None, replace=True, p=None, axis=0, shuffle=True):
        result = self.generator.choice(a, size=size, replace=replace,
                                       p=p, axis=axis, shuffle=shuffle)
        if size is not None and replace is False:
            self.seed_history.append([int(v) for v in result])
        return result

    def __getattr__(self, name):
        return getattr(self.generator, name)

# ============================================================
# 4. Robustness Curves and Summary
# ============================================================
def robustness_curve(adj0, order, seeds):
    n = len(adj0)
    if len(order) != n or len(seeds) != n or len(set(order)) != n:
        raise ValueError('Invalid removal order or seed history')
    adj, rows = copy_adjacency(adj0), []
    for step in range(n + 1):
        sizes = connected_component_sizes(adj) if adj else []
        rows.append({'removed': step, 'p': step / n,
                     'S1_over_N': sizes[0] / n if sizes else 0.0,
                     'S2_over_N': sizes[1] / n if len(sizes) > 1 else 0.0,
                     'removed_node': int(order[step - 1]) if step else None,
                     'seed_nodes': seeds[step - 1] if step else None})
        if step < n:
            remove_node_inplace(adj, order[step])
    return pd.DataFrame(rows)


def sem(x):
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def summarize(raw):
    summary = raw.groupby(KEYS, dropna=False, as_index=False).agg(
        mean_S1_over_N=('S1_over_N', 'mean'),
        sd_S1_over_N=('S1_over_N', 'std'),
        sem_S1_over_N=('S1_over_N', sem),
        mean_S2_over_N=('S2_over_N', 'mean'),
        sd_S2_over_N=('S2_over_N', 'std'),
        sem_S2_over_N=('S2_over_N', sem), n=('S1_over_N', 'size'))
    history = []
    for key, group in raw.groupby(KEYS, dropna=False, sort=False):
        step = int(key[5])
        ordered = group.sort_values('stochastic_id')
        removed = '' if not step else ';'.join(
            f'{int(r.stochastic_id)}:{int(r.removed_node)}'
            for r in ordered.itertuples(index=False))
        seeds = '' if not step else ';'.join(
            f'{int(r.stochastic_id)}:[{",".join(map(str, r.seed_nodes))}]'
            for r in ordered.itertuples(index=False))
        history.append(dict(zip(KEYS, key)) | {
            'removed_node_label': removed, 'seed_node_labels': seeds})
    return summary.merge(pd.DataFrame(history), on=KEYS,
                         how='left', validate='one_to_one')[COLUMNS]

# ============================================================
# 5. Per-Network Analysis
# ============================================================
def process_network(spec, rng_master):
    kind, rep, label, path = spec
    observed = set()
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            parts = line.strip().split()
            if parts and not parts[0].startswith('#'):
                if len(parts) < 2:
                    raise ValueError(f'Malformed edge list line in {path}: {line!r}')
                observed.update((int(parts[0]), int(parts[1])))
    if not observed:
        raise ValueError(f'No nodes in {path}')
    adj = read_integer_edgelist(path, include_missing_integer_nodes=True,
                                zero_indexed=0 in observed)
    lcc = max(nx.connected_components(adjacency_to_networkx(adj)), key=len)
    adj = {u: {v for v in adj[u] if v in lcc} for u in lcc}
    frames = []
    for attack in range(1, N_ATTACKS + 1):
        seed = int(rng_master.integers(0, 2**32 - 1))
        rng = SeedRecordingRNG(np.random.default_rng(seed))
        order = order_seamless(adj, rng, SEAMLESS_M)
        frames.append(robustness_curve(adj, order, rng.seed_history).assign(
            label=label, network_type=kind, network_replicate=rep,
            protocol='seamless', m=SEAMLESS_M, stochastic_id=attack))
    return summarize(pd.concat(frames, ignore_index=True)), len(adj)

# ============================================================
# 6. Serial Execution and Output
# ============================================================
def main():
    specs = list(network_specs())
    rng_master, summaries = np.random.default_rng(MASTER_SEED), []
    for i, spec in enumerate(specs, 1):
        summary, n = process_network(spec, rng_master)
        summaries.append(summary)
        print(f'Completed {i}/{len(specs)}: {spec[2]} (N={n})', flush=True)
    combined = pd.concat(summaries, ignore_index=True)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f'Saved {len(combined):,} rows to {OUTPUT_FILE}')


if __name__ == '__main__':
    main()
