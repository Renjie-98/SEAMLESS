"""
SEAMLESS synthetic20: Compute once, plot three times.

10 network families × 20 realizations × 100 attacks (m = 4).
Analyze Global/Explored mechanisms, APs, and network fragmentation.
Bootstrap: 2,000 shared network-resampling draws for 95% confidence intervals.

Cache first, compute only what's missing. No SEAMLESS reruns!
Output: Three SVG figures (network_group1–3.svg).
"""
from __future__ import annotations
import ast
import math
import hashlib
import json
import sys
from concurrent.futures import as_completed
from pathlib import Path
from tqdm import tqdm
from IPython.display import SVG, display
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
try:
    import fastparquet
except ImportError:
    fastparquet = None
try:
    from joblib.externals.loky import ProcessPoolExecutor as LokyProcessPoolExecutor
except ImportError:
    LokyProcessPoolExecutor = None
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

# ============================================================
# 1. Configuration
# ============================================================
BASE_DIR = Path.cwd()
SUMMARY_FILE = BASE_DIR / 'results_synthetic' / 'summary_synthetic.csv'
CACHE_FILE = BASE_DIR / 'results_synthetic' / 'summary_synthetic.parquet'
CACHE_MANIFEST = CACHE_FILE.with_suffix('.inputs.json')
H_CACHE_FILE = CACHE_FILE.with_name('summary_synthetic_cluster.parquet')
SVG_DIR = BASE_DIR / 'seamless_figures'
NETWORK_GROUPS = (('BA', 'ER'), ('HSBM4', 'LFR', 'RGG', 'SBM4'), ('SBM8', 'SBM16', 'SF2', 'WS'))
SYNTHETIC_TYPES = [kind for group in NETWORK_GROUPS for kind in group]
NETWORK_REPLICATES = list(range(1, 21))
SYNTHETIC_LABELS_BY_TYPE = {network_type: [f'{network_type}_N256_{network_type}_r{replicate}' for replicate in NETWORK_REPLICATES] for network_type in SYNTHETIC_TYPES}
N_ATTACKS = 100
SEAMLESS_M = 4
SYNTHETIC_LABELS = [label for labels in SYNTHETIC_LABELS_BY_TYPE.values() for label in labels]
N_BOOT = 2000
BOOTSTRAP_SEED = 42
FORCE_RECOMPUTE = False
MECHANISM_VERSION = 2
N_WORKERS = 20
PARQUET_COMPRESSION = 'SNAPPY'
CACHE_WRITE_BATCH_REALIZATIONS = 10
TEXT_SIZE = 26
REFERENCE_SINGLE_ROW_FIGURE_HEIGHT_IN = 7.6
REFERENCE_GRID_BOTTOM = 0.3
REFERENCE_BOTTOM_MARGIN_IN = REFERENCE_GRID_BOTTOM * REFERENCE_SINGLE_ROW_FIGURE_HEIGHT_IN
REFERENCE_TOP_MARGIN_IN = REFERENCE_SINGLE_ROW_FIGURE_HEIGHT_IN - REFERENCE_BOTTOM_MARGIN_IN - 4.6315
FIGURE_WIDTH_IN = 27.0
SUMMARY_DATA = pd.DataFrame()
PANEL_WIDTH_IN = 4.6315
PANEL_GAP_IN = 1.2913
REFERENCE_ROW_GAP_IN = 0.9
THREE_PANEL_WIDTH_IN = 3.0 * PANEL_WIDTH_IN + 2.0 * PANEL_GAP_IN
FRACTION_PAD = 0.04
FRACTION_TICKS = np.linspace(0.0, 1.0, 6)
NETWORK_FILES = {label: BASE_DIR / 'benchmarks' / f'{kind}_N256' / f'{kind}_r{rep}.edges' for kind, labels in SYNTHETIC_LABELS_BY_TYPE.items() for rep, label in zip(NETWORK_REPLICATES, labels)}
FEATURE_ORDER = ['degree', 'betweenness', 'k_core', 'clustering_coefficient', 'local_efficiency']
CACHE_COLUMNS = ['label', 'scope', 'stochastic_id', 'step', 'ap_count', 'selected_is_ap', 'degree', 'betweenness', 'k_core', 'clustering_coefficient', 'local_efficiency', 'degree_percentile', 'betweenness_percentile', 'k_core_percentile', 'clustering_coefficient_percentile', 'local_efficiency_percentile']
NETWORK_NODE_COUNTS = {}
COMPONENT_COLOR = '#d02623'
S1_LINESTYLE = '--'
S2_LINESTYLE = '-'
COMPONENT_LINEWIDTH = 2.2
CLUSTER_MOMENT_KEYS = ['mean_s', 'mean_s2', 'mean_s2_over_mean_s_sq', 'mean_s2_over_mean_s']
H_COLUMNS = ['label', 'stochastic_id', 'step', 'f', 'n_components', *CLUSTER_MOMENT_KEYS, 'S1', 'S2', 'S2_over_S1']


# ============================================================
# 2. Data Loading and Validation
# ============================================================
def read_integer_graph(path: Path) -> nx.Graph:
    edges: list[tuple[int, int]] = []
    observed_nodes: set[int] = set()
    with path.open('r', encoding='utf-8') as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f'Malformed edge-list line {line_no} in {path}: {line!r}')
            try:
                u = int(parts[0])
                v = int(parts[1])
            except ValueError as exc:
                raise ValueError(f'Node labels must be integers. Bad line {line_no} in {path}: {line!r}') from exc
            observed_nodes.update((u, v))
            if u != v:
                edges.append((u, v))
    if not observed_nodes:
        raise ValueError(f'No nodes found in edge list: {path}')
    graph = nx.Graph()
    graph.add_nodes_from(range(max(observed_nodes) + 1))
    graph.add_edges_from(edges)
    largest_cc = max(nx.connected_components(graph), key=len)
    return graph.subgraph(largest_cc).copy()

def validate_summary(summary):
    required = {'label', 'network_type', 'network_replicate', 'protocol', 'm', 'removed', 'p', 'removed_node_label', 'seed_node_labels', 'mean_S1_over_N', 'mean_S2_over_N'}
    missing = required - set(summary)
    if missing:
        raise ValueError(f'Missing summary columns: {sorted(missing)}')
    rows = summary[summary.label.isin(SYNTHETIC_LABELS)]
    if rows.empty or not rows.protocol.eq('seamless').all() or (not rows.m.eq(SEAMLESS_M).all()):
        raise ValueError('Expected SEAMLESS m=4 data for the requested networks.')

def requested_labels_by_type(summary):
    available = set(summary.label.astype(str))
    missing = set(SYNTHETIC_LABELS) - available
    files = [str(NETWORK_FILES[label]) for label in SYNTHETIC_LABELS if not NETWORK_FILES[label].exists()]
    if missing:
        raise ValueError(f'Missing network summaries: {sorted(missing)}')
    if files:
        raise FileNotFoundError('Missing edge lists:\\n' + '\\n'.join(files))
    for kind, labels in SYNTHETIC_LABELS_BY_TYPE.items():
        rows = summary[summary.label.isin(labels)]
        if set(rows.network_type) != {kind} or set(rows.network_replicate.astype(int)) != set(NETWORK_REPLICATES):
            raise ValueError(f'Incorrect metadata for {kind}')
    return SYNTHETIC_LABELS_BY_TYPE


# ============================================================
# 3. SEAMLESS Mechanism Analysis
# ============================================================
def parse_removed_node_field(value):
    return {int(k): int(v) for token in str(value).split(';') if ':' in token for k, v in [token.split(':', 1)]}

def parse_seed_node_field(value):
    return {int(k): [int(x) for x in ast.literal_eval(v)] for token in str(value).split(';') if ':' in token for k, v in [token.split(':', 1)]}

def build_step_history(network_summary):
    return {int(row.removed): {'removed': parse_removed_node_field(row.removed_node_label), 'seeds': parse_seed_node_field(row.seed_node_labels)} for row in network_summary.itertuples(index=False)}

def candidate_set_from_seeds(graph, seeds):
    if not set(seeds) <= set(graph):
        raise ValueError('Seed absent from residual graph.')
    return {neighbor for seed in seeds for neighbor in graph.neighbors(seed)}

def build_explored_network(graph, seeds, candidate_set):
    queried = set(candidate_set)
    queried.update((v for node in candidate_set for v in graph.neighbors(node)))
    explored = nx.Graph()
    explored.add_nodes_from(seeds)
    explored.add_nodes_from(queried)
    explored.add_edges_from(((u, v) for u in queried for v in graph.neighbors(u)))
    return explored

def node_local_efficiency_values(graph):
    return {node: nx.global_efficiency(graph.subgraph(list(graph.neighbors(node)))) if graph.degree(node) >= 2 else 0.0 for node in graph}

def compute_all_node_features(graph):
    if not graph:
        return {key: {} for key in FEATURE_ORDER}
    if len(graph) == 1:
        return {key: {next(iter(graph)): 0.0} for key in FEATURE_ORDER}
    return {'degree': dict(graph.degree()), 'betweenness': nx.betweenness_centrality(graph), 'k_core': nx.core_number(graph), 'clustering_coefficient': nx.clustering(graph), 'local_efficiency': node_local_efficiency_values(graph)}

def percentile_rank(values, selected_node):
    if selected_node not in values:
        return np.nan
    arr = np.asarray(list(values.values()), dtype=float)
    val = values[selected_node]
    # Exact midrank: strict lower values and exact ties are disjoint.
    if not np.isfinite(arr).all() or not np.isfinite(val):
        return np.nan
    return float((np.count_nonzero(arr < val) + 0.5 * np.count_nonzero(arr == val)) / len(arr))

def selected_node_features(graph, selected_node):
    result = {}
    for key, values in compute_all_node_features(graph).items():
        result[key] = float(values.get(selected_node, np.nan))
        result[f'{key}_percentile'] = percentile_rank(values, selected_node)
    return result

def compute_realization(label: str, initial_graph: nx.Graph, history: dict[int, dict], stochastic_id: int) -> pd.DataFrame:
    graph = initial_graph.copy()
    n_initial = graph.number_of_nodes()
    rows: list[dict] = []
    for step in range(1, n_initial + 1):
        removed_map = history[step]['removed']
        seed_map = history[step]['seeds']
        if stochastic_id not in removed_map:
            raise ValueError(f'{label}: missing removed node for realization {stochastic_id}, step {step}.')
        if stochastic_id not in seed_map:
            raise ValueError(f'{label}: missing seed nodes for realization {stochastic_id}, step {step}.')
        selected_node = int(removed_map[stochastic_id])
        seeds = [int(node) for node in seed_map[stochastic_id]]
        if selected_node not in graph:
            raise ValueError(f'{label}: selected node {selected_node} is absent before step {step}, realization {stochastic_id}.')
        candidate_set = candidate_set_from_seeds(graph, seeds)
        if candidate_set:
            if selected_node not in candidate_set:
                raise ValueError(f'{label}: selected node {selected_node} is not in the candidate set at step {step}, realization {stochastic_id}.')
        elif selected_node not in seeds:
            raise ValueError(f'{label}: empty candidate set but selected node {selected_node} is not one of the seeds at step {step}, realization {stochastic_id}.')
        explored_graph = build_explored_network(graph, seeds, candidate_set)
        for scope, scope_graph in (('Global', graph), ('Explored', explored_graph)):
            articulation_points = set(nx.articulation_points(scope_graph))
            ap_count = len(articulation_points)
            row = {'label': label, 'scope': scope, 'stochastic_id': stochastic_id, 'step': step, 'ap_count': ap_count, 'selected_is_ap': int(selected_node in articulation_points)}
            row.update(selected_node_features(scope_graph, selected_node))
            rows.append(row)
        graph.remove_node(selected_node)
    if graph.number_of_nodes() != 0:
        raise AssertionError(f'{label}: residual graph is not empty after the final removal.')
    return pd.DataFrame(rows)


# ============================================================
# 4. Parallel Computing and Parquet Cache
# ============================================================
def require_parallel_backend() -> None:
    if N_WORKERS > 1 and LokyProcessPoolExecutor is None:
        raise ImportError('20-worker parallel execution requires joblib/loky.\nInstall it once in the notebook with:\n\n    %pip install joblib\n\nThen restart the kernel and run this script again.')

def _compute_attack_task(label, initial_graph, history, stochastic_id, need_mechanism, need_h):
    """One independent attack; compute only missing outputs in a persistent worker."""
    mechanism = compute_realization(label, initial_graph, history, stochastic_id) if need_mechanism else None
    cluster = None
    if need_h:
        cluster = compute_cluster_moment_trajectory(initial_graph, history, [stochastic_id])
        cluster.insert(0, 'label', label)
    return (stochastic_id, mechanism, cluster)

def _flush_attack_batches(mechanism_batch, h_batch, completed, completed_h, mechanism_keys, h_keys):
    """Only the main process writes Parquet files."""
    if mechanism_batch:
        append_cache(pd.concat(mechanism_batch, ignore_index=True))
        completed.update(mechanism_keys)
        mechanism_batch.clear()
        mechanism_keys.clear()
    if h_batch:
        append_h_cache(pd.concat(h_batch, ignore_index=True))
        completed_h.update(h_keys)
        h_batch.clear()
        h_keys.clear()

def compute_missing_realizations_for_network(*, label, initial_graph, history, completed, completed_h, executor):
    tasks = [(sid, (label, sid) not in completed, (label, sid) not in completed_h) for sid in range(1, N_ATTACKS + 1)]
    tasks = [task for task in tasks if task[1] or task[2]]
    if not tasks:
        return
    mechanism_batch, h_batch, mechanism_keys, h_keys = ([], [], [], [])

    def save_result(sid, mechanism, cluster):
        if mechanism is not None:
            mechanism_batch.append(mechanism)
            mechanism_keys.append((label, sid))
        if cluster is not None:
            h_batch.append(cluster)
            h_keys.append((label, sid))
        if max(len(mechanism_batch), len(h_batch)) >= CACHE_WRITE_BATCH_REALIZATIONS:
            _flush_attack_batches(mechanism_batch, h_batch, completed, completed_h, mechanism_keys, h_keys)
    if executor is None:
        for sid, need_mechanism, need_h in tasks:
            save_result(*_compute_attack_task(label, initial_graph, history, sid, need_mechanism, need_h))
    else:
        futures = {executor.submit(_compute_attack_task, label, initial_graph, history, sid, need_mechanism, need_h): sid for sid, need_mechanism, need_h in tasks}
        try:
            for future in as_completed(futures):
                expected_sid = futures[future]
                sid, mechanism, cluster = future.result()
                if sid != expected_sid:
                    raise RuntimeError(f'Worker returned attack {sid}, expected {expected_sid}.')
                save_result(sid, mechanism, cluster)
        except BaseException:
            _flush_attack_batches(mechanism_batch, h_batch, completed, completed_h, mechanism_keys, h_keys)
            for future in futures:
                future.cancel()
            raise
    _flush_attack_batches(mechanism_batch, h_batch, completed, completed_h, mechanism_keys, h_keys)

def input_fingerprint() -> dict:
    """Fingerprint CSV and all 200 network files to avoid stale cached results."""

    def digest(path: Path) -> str:
        sha = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                sha.update(chunk)
        return sha.hexdigest()
    return {'version': 1, 'mechanism_version': MECHANISM_VERSION, 'm': SEAMLESS_M, 'n_attacks': N_ATTACKS, 'summary_sha256': digest(SUMMARY_FILE), 'network_sha256': {label: digest(NETWORK_FILES[label]) for label in SYNTHETIC_LABELS}}

def ensure_cache_matches_inputs() -> None:
    """Reuse matching caches; reset old or stale caches before any computation.

    Write the manifest before computation so interrupted runs can resume.
    """
    current = input_fingerprint()
    previous = None
    if CACHE_MANIFEST.exists():
        try:
            previous = json.loads(CACHE_MANIFEST.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            pass
    if FORCE_RECOMPUTE or previous != current:
        if CACHE_FILE.exists() or H_CACHE_FILE.exists():
            print('Cache inputs changed (or forced): rebuilding stale caches.')
        CACHE_FILE.unlink(missing_ok=True)
        H_CACHE_FILE.unlink(missing_ok=True)
    if FORCE_RECOMPUTE or previous != current or (not CACHE_MANIFEST.exists()):
        CACHE_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        temporary = CACHE_MANIFEST.with_suffix('.tmp')
        temporary.write_text(json.dumps(current, indent=2), encoding='utf-8')
        temporary.replace(CACHE_MANIFEST)

def require_fastparquet() -> None:
    if fastparquet is None:
        raise ImportError('This Figure_synthetic20 script uses a single appendable Parquet cache. Install fastparquet first, for example:\n\n    pip install fastparquet\n')

def validate_existing_cache_schema() -> None:
    if not CACHE_FILE.exists():
        return
    try:
        columns = list(fastparquet.ParquetFile(str(CACHE_FILE)).columns)
    except Exception as exc:
        raise RuntimeError(f'Cannot read mechanism cache: {CACHE_FILE}') from exc
    if columns != CACHE_COLUMNS:
        raise ValueError(f'Unexpected Parquet schema in {CACHE_FILE}. This version requires exactly the 16 columns in CACHE_COLUMNS. Rename the incompatible file before running.')

def load_cache_keys() -> pd.DataFrame:
    if not CACHE_FILE.exists():
        return pd.DataFrame(columns=['label', 'scope', 'stochastic_id', 'step'])
    return fastparquet.ParquetFile(str(CACHE_FILE)).to_pandas(columns=['label', 'scope', 'stochastic_id', 'step'])

def completed_realizations(cache_keys: pd.DataFrame, network_sizes: dict[str, int]) -> set[tuple[str, int]]:
    completed: set[tuple[str, int]] = set()
    if cache_keys.empty:
        return completed
    cache_keys = cache_keys.copy()
    cache_keys['label'] = cache_keys['label'].astype(str)
    cache_keys['scope'] = cache_keys['scope'].astype(str)
    if not set(cache_keys['scope'].unique()).issubset({'Global', 'Explored'}):
        raise ValueError('Cache contains outdated scope names. Rename the old Parquet file or set FORCE_RECOMPUTE=True to rebuild it.')
    duplicated = cache_keys.duplicated(subset=['label', 'stochastic_id', 'scope', 'step'], keep=False)
    duplicate_attacks = set(((str(label), int(sid)) for label, sid in cache_keys.loc[duplicated, ['label', 'stochastic_id']].itertuples(index=False, name=None)))
    for (label, stochastic_id), group in cache_keys.groupby(['label', 'stochastic_id'], sort=False):
        label = str(label)
        if label not in network_sizes:
            continue
        n_nodes = int(network_sizes[label])
        global_steps = set(group.loc[group['scope'] == 'Global', 'step'].astype(int).tolist())
        explored_steps = set(group.loc[group['scope'] == 'Explored', 'step'].astype(int).tolist())
        expected_steps = set(range(1, n_nodes + 1))
        if (label, int(stochastic_id)) not in duplicate_attacks and global_steps == expected_steps and (explored_steps == expected_steps):
            completed.add((label, int(stochastic_id)))
    return completed

def invalid_cache_mask(frame: pd.DataFrame) -> pd.Series:
    """Shared validation for cached values, used both before and after computing."""
    invalid = ~frame['scope'].isin(('Global', 'Explored')) | ~frame['selected_is_ap'].isin((0, 1))
    ap = pd.to_numeric(frame['ap_count'], errors='coerce')
    invalid |= ~np.isfinite(ap) | (ap < 0)
    for metric in FEATURE_ORDER:
        values = pd.to_numeric(frame[metric], errors='coerce')
        rank = pd.to_numeric(frame[f'{metric}_percentile'], errors='coerce')
        invalid |= ~np.isfinite(values) | ~np.isfinite(rank) | ~rank.between(0, 1)
    return invalid

def append_cache(dataframe: pd.DataFrame) -> None:
    if dataframe.empty:
        return
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    fastparquet.write(str(CACHE_FILE), dataframe[CACHE_COLUMNS], append=CACHE_FILE.exists(), compression=PARQUET_COMPRESSION, write_index=False, stats=True)

def load_mechanism_cache(requested_labels=None) -> pd.DataFrame:
    """Load and validate the completed group without requiring later groups."""
    if not CACHE_FILE.exists():
        raise FileNotFoundError(f'Mechanism cache not found: {CACHE_FILE}')
    dataframe = fastparquet.ParquetFile(str(CACHE_FILE)).to_pandas(columns=CACHE_COLUMNS)
    if requested_labels is not None:
        dataframe = dataframe.loc[dataframe["label"].isin(requested_labels)].copy()
    key_columns = ['label', 'stochastic_id', 'scope', 'step']
    if dataframe.duplicated(subset=key_columns).any():
        raise ValueError('Duplicate mechanism cache rows found. Repair the cache before plotting.')
    if invalid_cache_mask(dataframe).any():
        raise ValueError('Mechanism cache contains invalid values; existing cache will not be modified automatically.')
    node_counts = dataframe['label'].map(NETWORK_NODE_COUNTS)
    if node_counts.isna().any():
        raise ValueError('Missing initial component sizes for p_before reconstruction.')
    dataframe['p_before'] = (dataframe['step'].astype(int) - 1) / node_counts
    return dataframe.reset_index(drop=True)

def append_h_cache(dataframe: pd.DataFrame) -> None:
    if dataframe.empty:
        return
    H_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    fastparquet.write(str(H_CACHE_FILE), dataframe[H_COLUMNS], append=H_CACHE_FILE.exists(), compression=PARQUET_COMPRESSION, write_index=False, stats=True)

def prepare_h_cache(network_sizes):
    """Read completed H-cache attacks without modifying existing cache files.

    Missing attacks are computed normally. Existing incomplete/invalid attacks
    cause a clear error rather than silently repairing or overwriting data.
    """
    if not H_CACHE_FILE.exists():
        return set()
    try:
        frame = fastparquet.ParquetFile(str(H_CACHE_FILE)).to_pandas()
    except Exception as exc:
        raise RuntimeError(
            f'Cannot read H cache: {H_CACHE_FILE}. '
            'Back up or rename the damaged file before rerunning; '
            'the mechanism cache will be preserved.'
        ) from exc
    if list(frame.columns) != H_COLUMNS:
        raise ValueError(f'Unexpected H cache columns: {H_CACHE_FILE}')
    complete = set()
    for (label, sid), group in frame.groupby(['label', 'stochastic_id'], sort=False):
        label, sid = str(label), int(sid)
        if label not in network_sizes:
            continue
        n = int(network_sizes[label])
        steps = pd.to_numeric(group['step'], errors='coerce')
        fractions = pd.to_numeric(group['f'], errors='coerce')
        metrics = group[['n_components', *CLUSTER_MOMENT_KEYS, 'S1', 'S2', 'S2_over_S1']]
        # At step == n the graph is empty: three ratio metrics are
        # mathematically undefined (NaN). This is expected, not corruption.
        ratio_columns = ['mean_s2_over_mean_s_sq', 'mean_s2_over_mean_s', 'S2_over_S1']
        ordinary_columns = ['n_components', 'mean_s', 'mean_s2', 'S1', 'S2']
        valid_metrics = np.isfinite(metrics[ordinary_columns].to_numpy(dtype=float)).all()
        valid_ratios = np.isfinite(
            metrics.loc[steps < n, ratio_columns].to_numpy(dtype=float)
        ).all()
        terminal_ratios = metrics.loc[steps == n, ratio_columns].to_numpy(dtype=float)
        valid_terminal = terminal_ratios.shape == (1, len(ratio_columns)) and np.isnan(terminal_ratios).all()
        valid = (
            len(group) == n + 1
            and steps.notna().all()
            and set(steps.astype(int)) == set(range(n + 1))
            and np.allclose(fractions.to_numpy(dtype=float), steps.to_numpy(dtype=float) / n)
            and valid_metrics and valid_ratios and valid_terminal
        )
        if not valid:
            raise ValueError(
                f'Incomplete or invalid H cache for {label}, attack {sid}. '
                'Existing caches are not modified automatically.'
            )
        complete.add((label, sid))
    return complete

def load_h_cache(requested_labels=None):
    if not H_CACHE_FILE.exists():
        raise FileNotFoundError(H_CACHE_FILE)
    frame = fastparquet.ParquetFile(str(H_CACHE_FILE)).to_pandas(columns=H_COLUMNS)
    if requested_labels is not None:
        frame = frame.loc[frame["label"].isin(requested_labels)].copy()
    return frame


# ============================================================
# 5. Statistical Aggregation
# ============================================================
def aggregate_ap(network_cache):
    return network_cache.groupby(['scope', 'step', 'p_before'], as_index=False).agg(mean_ap_count=('ap_count', 'mean'), selected_ap_probability=('selected_is_ap', 'mean'))

def aggregate_global_explored_differences(network_cache):
    keys = ['label', 'stochastic_id', 'step', 'p_before']
    pivot = network_cache.pivot_table(index=keys, columns='scope', values=[f'{key}_percentile' for key in FEATURE_ORDER])
    rows = []
    for feature in FEATURE_ORDER:
        pair = pivot[f'{feature}_percentile']
        delta = (pair['Global'] - pair['Explored']).abs().rename('mean_difference')
        data = delta.reset_index().groupby(['step', 'p_before'], as_index=False).mean(numeric_only=True)
        data['feature'] = feature
        rows.append(data)
    return pd.concat(rows, ignore_index=True)

def cluster_size_metrics(graph):
    sizes = sorted((len(c) for c in nx.connected_components(graph)), reverse=True)
    if not sizes:
        return dict(n_components=0, mean_s=0.0, mean_s2=0.0, mean_s2_over_mean_s_sq=np.nan, mean_s2_over_mean_s=np.nan, S1=0.0, S2=0.0, S2_over_S1=np.nan)
    mean, second = (np.mean(sizes), np.mean(np.square(sizes)))
    s1, s2 = (sizes[0], sizes[1] if len(sizes) > 1 else 0)
    return dict(n_components=len(sizes), mean_s=mean, mean_s2=second, mean_s2_over_mean_s_sq=second / mean ** 2, mean_s2_over_mean_s=second / mean, S1=s1, S2=s2, S2_over_S1=s2 / s1)

def compute_cluster_moment_trajectory(initial_graph, history, stochastic_ids):
    n = len(initial_graph)
    rows = []
    for sid in stochastic_ids:
        graph = initial_graph.copy()
        rows.append(dict(stochastic_id=sid, step=0, f=0.0, **cluster_size_metrics(graph)))
        for step in range(1, n + 1):
            node = history[step]['removed'][sid]
            if node not in graph:
                raise ValueError(f'Removed node missing at attack {sid}, step {step}')
            graph.remove_node(node)
            rows.append(dict(stochastic_id=sid, step=step, f=step / n, **cluster_size_metrics(graph)))
    return pd.DataFrame(rows)

def aggregate_cluster_moment_trajectory(trajectory):
    metrics = ['n_components', *CLUSTER_MOMENT_KEYS, 'S1', 'S2', 'S2_over_S1']
    result = trajectory.groupby(['step', 'f'], as_index=False)[metrics].mean()
    return result.rename(columns={key: f'mean_{key}' for key in metrics})

def component_curve_from_summary(network_summary):
    return network_summary[['p', 'mean_S1_over_N', 'mean_S2_over_N']].dropna().rename(columns={'mean_S1_over_N': 'S1_over_N', 'mean_S2_over_N': 'S2_over_N'}).sort_values('p').drop_duplicates('p').reset_index(drop=True)

def _interp_to_grid(x, y, grid):
    x, y, grid = map(lambda v: np.asarray(v, dtype=float), (x, y, grid))
    good = np.isfinite(x) & np.isfinite(y)
    if not good.any():
        return np.full_like(grid, np.nan)
    series = pd.Series(y[good], index=x[good]).groupby(level=0).mean().sort_index()
    result = np.full_like(grid, np.nan)
    if len(series) == 1:
        result[np.argmin(np.abs(grid - series.index[0]))] = series.iloc[0]
    else:
        inside = (grid >= series.index[0]) & (grid <= series.index[-1])
        result[inside] = np.interp(grid[inside], series.index, series.values)
    return result

def _mean_bootstrap_ci_across_replicates(matrix, bootstrap_indices):
    """Pointwise 95% CI; resample the same whole-network rows for every metric."""
    data = np.asarray(matrix, dtype=float)
    if data.ndim != 2 or data.shape[0] < 2:
        raise ValueError('Bootstrap requires at least two independent network curves.')
    if bootstrap_indices.shape != (N_BOOT, data.shape[0]):
        raise ValueError('Bootstrap indices do not match the network ensemble.')
    count = np.sum(np.isfinite(data), axis=0)
    sums = np.nansum(data, axis=0)
    mean = np.divide(sums, count, out=np.full(data.shape[1], np.nan), where=count > 0)
    sampled = data[bootstrap_indices]
    boot_counts = np.sum(np.isfinite(sampled), axis=1)
    boot_means = np.divide(np.nansum(sampled, axis=1), boot_counts, out=np.full((N_BOOT, data.shape[1]), np.nan), where=boot_counts > 0)
    low = np.full(data.shape[1], np.nan)
    high = np.full(data.shape[1], np.nan)
    for j in np.flatnonzero(count >= 2):
        valid = boot_means[:, j]
        valid = valid[np.isfinite(valid)]
        if valid.size >= 2:
            low[j], high[j] = np.percentile(valid, [2.5, 97.5])
    return (mean, low, high, count)

def common_fraction_grid(labels: list[str], network_sizes: dict[str, int]) -> np.ndarray:
    max_nodes = max((int(network_sizes[label]) for label in labels))
    return np.linspace(0.0, 1.0, max_nodes + 1)

def bootstrap_network_curves(curves, labels, x_column, y_column, grid, bootstrap_indices, subset=None):
    matrix = []
    for label in labels:
        frame = curves[label]
        if subset is not None:
            frame = frame.loc[subset(frame)]
        matrix.append(_interp_to_grid(frame[x_column], frame[y_column], grid))
    matrix = np.vstack(matrix)
    return _mean_bootstrap_ci_across_replicates(matrix, bootstrap_indices)[:3]

def aggregate_ap_ensemble(type_cache, labels, f_grid_before, rng):
    curves = {label: aggregate_ap(type_cache[type_cache['label'] == label]) for label in labels}
    rows = []
    for scope in ('Global', 'Explored'):
        output = pd.DataFrame({'scope': scope, 'step': np.arange(1, len(f_grid_before) + 1), 'p_before': f_grid_before})
        for metric in ('mean_ap_count', 'selected_ap_probability'):
            output[metric], output[f'ci_low_{metric}'], output[f'ci_high_{metric}'] = bootstrap_network_curves(curves, labels, 'p_before', metric, f_grid_before, rng, subset=lambda frame: frame['scope'] == scope)
        rows.append(output)
    return pd.concat(rows, ignore_index=True)

def aggregate_global_explored_differences_ensemble(type_cache, labels, f_grid_before, rng):
    curves = {label: aggregate_global_explored_differences(type_cache[type_cache['label'] == label]) for label in labels}
    rows = []
    for feature in FEATURE_ORDER:
        mean, low, high = bootstrap_network_curves(curves, labels, 'p_before', 'mean_difference', f_grid_before, rng, subset=lambda frame: frame['feature'] == feature)
        rows.append(pd.DataFrame({'feature': feature, 'p_before': f_grid_before, 'mean_difference': mean, 'ci_low_difference': low, 'ci_high_difference': high}))
    return pd.concat(rows, ignore_index=True)

def aggregate_cluster_moments_ensemble(labels: list[str], h_cache: pd.DataFrame, f_grid: np.ndarray, rng) -> pd.DataFrame:
    replicate_summaries = {}
    for label in labels:
        trajectory = h_cache.loc[h_cache['label'] == label]
        if trajectory['stochastic_id'].nunique() != N_ATTACKS:
            raise RuntimeError(f'{label}: incomplete cluster cache.')
        replicate_summaries[label] = aggregate_cluster_moment_trajectory(trajectory)
    output = pd.DataFrame({'step': np.arange(len(f_grid), dtype=int), 'f': f_grid})
    source_mean_columns = ['mean_n_components', *[f'mean_{metric}' for metric in CLUSTER_MOMENT_KEYS], 'mean_S1', 'mean_S2', 'mean_S2_over_S1']
    for mean_column in source_mean_columns:
        mean_values, low_values, high_values = bootstrap_network_curves(replicate_summaries, labels, 'f', mean_column, f_grid, rng)
        output[mean_column] = mean_values
        suffix = mean_column.removeprefix('mean_')
        output[f'ci_low_{suffix}'] = low_values
        output[f'ci_high_{suffix}'] = high_values
    return output

def component_curve_from_summary_ensemble(summary, labels, f_grid, rng):
    curves = {label: component_curve_from_summary(summary[summary.label == label]) for label in labels}
    output = pd.DataFrame({'p': f_grid})
    for metric in ('S1_over_N', 'S2_over_N'):
        output[metric], output[f'ci_low_{metric}'], output[f'ci_high_{metric}'] = bootstrap_network_curves(curves, labels, 'p', metric, f_grid, rng)
    return output


# ============================================================
# 6. Plotting Functions
# ============================================================
def configure_matplotlib():
    plt.rcParams.update({'font.family': 'Times New Roman', 'mathtext.fontset': 'stix', 'text.usetex': False, 'svg.fonttype': 'path', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.unicode_minus': False})

def fraction_tick_formatter(x, pos):
    if np.isclose(x, 0.0):
        return '0'
    return f'{x:.1f}'

def numeric_tick_formatter(x, pos):
    if np.isclose(x, round(x)):
        return f'{int(round(x))}'
    return f'{x:.1f}'

def set_fontsizes(ax):
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        label.set_fontfamily('Times New Roman')
        label.set_fontsize(TEXT_SIZE)

def nice_upper(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    maximum = max(0.0, float(np.max(values))) if values.size else 0.0
    if maximum <= 1:
        return 1.0
    if maximum <= 5:
        return float(np.ceil(maximum))
    if maximum <= 10:
        return float(np.ceil(maximum / 2) * 2)
    if maximum <= 20:
        return float(np.ceil(maximum / 5) * 5)
    if maximum <= 50:
        return float(np.ceil(maximum / 10) * 10)
    scale = 10 ** np.floor(np.log10(maximum))
    return float(np.ceil(maximum / scale) * scale)

def positive_padded_limit(values):
    upper = nice_upper(values)
    return (-upper * FRACTION_PAD, upper * (1 + FRACTION_PAD), np.linspace(0, upper, 6))

def style_axis(ax, *, y_probability: bool) -> None:
    ax.set_box_aspect(1)
    ax.grid(True, color='#D9D9D9', linewidth=0.8, alpha=0.9)
    ax.tick_params(axis='both', which='major', labelsize=TEXT_SIZE, length=5, width=0.9, direction='out')
    for spine in ax.spines.values():
        spine.set_color('#444444')
        spine.set_linewidth(0.9)
    ax.set_xlim(-FRACTION_PAD, 1.0 + FRACTION_PAD)
    ax.set_xticks(FRACTION_TICKS)
    ax.xaxis.set_major_formatter(FuncFormatter(fraction_tick_formatter))
    if y_probability:
        ax.set_ylim(-FRACTION_PAD, 1.0 + FRACTION_PAD)
        ax.set_yticks(FRACTION_TICKS)
        ax.yaxis.set_major_formatter(FuncFormatter(fraction_tick_formatter))
    set_fontsizes(ax)

def bounded_fill_ci(ax, x, low, high, *, color, alpha, lower_bound=None, upper_bound=None):
    low, high = (np.asarray(low, dtype=float), np.asarray(high, dtype=float))
    if lower_bound is not None:
        low = np.maximum(low, lower_bound)
    if upper_bound is not None:
        high = np.minimum(high, upper_bound)
    ax.fill_between(x, low, high, color=color, alpha=alpha, linewidth=0, zorder=1)
FEATURE_COLORS = {'degree': '#a3c7da', 'betweenness': '#2072aa', 'k_core': '#acc88a', 'clustering_coefficient': '#339a39', 'local_efficiency': '#f0e89c'}
AP_PROBABILITY_COLOR = '#e17927'
BALANCE_COLOR = '#ccba73'
AP_COUNT_COLOR = '#623f88'
HETEROGENEITY_COLOR = '#e49698'

def plot_scope_curve(ax, x, y, *, scope: str, color: str, linewidth: float=2.0) -> None:
    ax.plot(x, y, linewidth=linewidth, color=color, linestyle='--' if scope == 'Global' else '-', marker=None, zorder=7 if scope == 'Global' else 5)

def plot_ap_availability(ax, ap_summary):
    """Plot only global and explored articulation-point counts."""
    count_values = []
    for scope in ('Explored', 'Global'):
        data = ap_summary[ap_summary['scope'] == scope].sort_values('p_before')
        plot_scope_curve(ax, data['p_before'], data['mean_ap_count'], scope=scope, color=AP_COUNT_COLOR)
        bounded_fill_ci(ax, data['p_before'], data['ci_low_mean_ap_count'], data['ci_high_mean_ap_count'], color=AP_COUNT_COLOR, alpha=0.04 if scope == 'Explored' else 0.065, lower_bound=0.0)
        count_values.extend(data['mean_ap_count'].tolist())
    style_axis(ax, y_probability=False)
    y_low, y_high, y_ticks = positive_padded_limit(count_values)
    ax.set_ylim(y_low, y_high)
    ax.set_yticks(y_ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(numeric_tick_formatter))
    ax.set_ylabel('Number of APs', fontsize=TEXT_SIZE)

def plot_structural_difference(ax, difference_summary: pd.DataFrame, ap_summary: pd.DataFrame) -> None:
    for feature in FEATURE_ORDER:
        color = FEATURE_COLORS[feature]
        data = difference_summary[difference_summary['feature'] == feature].sort_values('p_before').reset_index(drop=True)
        x = data['p_before'].to_numpy(dtype=float)
        y = data['mean_difference'].to_numpy(dtype=float)
        low = data['ci_low_difference'].to_numpy(dtype=float)
        high = data['ci_high_difference'].to_numpy(dtype=float)
        ax.plot(x, y, color=color, linestyle='-', linewidth=1.9, zorder=5)
        bounded_fill_ci(ax, x, low, high, color=color, alpha=0.04, lower_bound=0.0, upper_bound=1.0)
    for scope in ['Explored', 'Global']:
        data = ap_summary[ap_summary['scope'] == scope].sort_values('p_before').reset_index(drop=True)
        ax.plot(data['p_before'], data['selected_ap_probability'], linewidth=2.0, color=AP_PROBABILITY_COLOR, linestyle='--' if scope == 'Global' else '-', marker=None, zorder=2.0 if scope == 'Global' else 1.8)
        p_x = data['p_before'].to_numpy(dtype=float)
        p_mean = data['selected_ap_probability'].to_numpy(dtype=float)
        p_lower = np.maximum(0.0, data['ci_low_selected_ap_probability'].to_numpy(dtype=float))
        p_upper = np.minimum(1.0, data['ci_high_selected_ap_probability'].to_numpy(dtype=float))
        ax.fill_between(p_x, p_lower, p_upper, color=AP_PROBABILITY_COLOR, alpha=0.028 if scope == 'Explored' else 0.045, linewidth=0, zorder=0.5)
    style_axis(ax, y_probability=True)
    ax.set_ylabel('Value', fontsize=TEXT_SIZE)

def plot_component_reference_curves(ax, component_curve: pd.DataFrame, *, visible_axis: bool=True):
    target_ax = ax if visible_axis else ax.twinx()
    if not visible_axis:
        target_ax.set_facecolor('none')
        target_ax.patch.set_alpha(0.0)
        target_ax.set_ylim(-FRACTION_PAD, 1.0 + FRACTION_PAD)
        target_ax.set_yticks(np.linspace(0.0, 1.0, 6))
        target_ax.tick_params(axis='y', left=False, right=False, labelleft=False, labelright=False)
        for spine in target_ax.spines.values():
            spine.set_visible(False)
    target_ax.plot(component_curve['p'], component_curve['S1_over_N'], color=COMPONENT_COLOR, linestyle=S1_LINESTYLE, linewidth=COMPONENT_LINEWIDTH, marker=None, zorder=8)
    target_ax.plot(component_curve['p'], component_curve['S2_over_N'], color=COMPONENT_COLOR, linestyle=S2_LINESTYLE, linewidth=COMPONENT_LINEWIDTH, marker=None, zorder=9)
    for mean_key, color in (('S1_over_N', COMPONENT_COLOR), ('S2_over_N', COMPONENT_COLOR)):
        bounded_fill_ci(target_ax, component_curve['p'], component_curve[f'ci_low_{mean_key}'], component_curve[f'ci_high_{mean_key}'], color=color, alpha=0.11, lower_bound=0.0, upper_bound=1.0)
    return target_ax

def add_components_and_balance_to_panel1(ax, cluster_summary):
    """Show S2/S1 only; S1/N and S2/N are reserved for column three."""
    data = cluster_summary[['f', 'mean_S2_over_S1', 'ci_low_S2_over_S1', 'ci_high_S2_over_S1']].sort_values('f').reset_index(drop=True)
    x = data['f'].to_numpy(dtype=float)
    y = data['mean_S2_over_S1'].to_numpy(dtype=float)
    low = data['ci_low_S2_over_S1'].to_numpy(dtype=float)
    high = data['ci_high_S2_over_S1'].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    xv = x[valid]
    yv = y[valid]
    lv, hv = (low[valid], high[valid])
    ax.fill_between(xv, np.clip(lv, 0.0, 1.0), np.clip(hv, 0.0, 1.0), color=BALANCE_COLOR, alpha=0.08, linewidth=0.0, zorder=1)
    ax.plot(xv, yv, color=BALANCE_COLOR, linewidth=2.0, zorder=7)

def plot_combined_cluster_statistics(ax, cluster_summary, component_curve):
    """Cluster-size heterogeneity (left), S1/N and S2/N (right)."""
    metric = 'mean_s2_over_mean_s_sq'
    style_axis(ax, y_probability=False)
    data = cluster_summary[['f', f'mean_{metric}']].sort_values('f')
    x = data['f'].to_numpy(dtype=float)
    y = data[f'mean_{metric}'].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    ax.plot(x[valid], y[valid], color=HETEROGENEITY_COLOR, linestyle='-', linewidth=2.1, zorder=6)
    low = cluster_summary.sort_values('f')[f'ci_low_{metric}'].to_numpy(dtype=float)
    high = cluster_summary.sort_values('f')[f'ci_high_{metric}'].to_numpy(dtype=float)
    bounded_fill_ci(ax, x, low, high, color=HETEROGENEITY_COLOR, alpha=0.12, lower_bound=0.0)
    finite_y = y[np.isfinite(y)]
    if finite_y.size:
        y_max = max(0.0, float(np.max(finite_y)))
        step = max(1, int(math.ceil(y_max / 5.0)))
        ticks = np.arange(6, dtype=float) * step
        pad = FRACTION_PAD * (ticks[-1] - ticks[0])
        ax.set_ylim(ticks[0] - pad, ticks[-1] + pad)
        ax.set_yticks(ticks)
        ax.set_yticklabels([str(int(v)) for v in ticks], fontname='Times New Roman', fontsize=TEXT_SIZE)
    ax.set_ylabel('Cluster-size heterogeneity', fontsize=TEXT_SIZE)
    set_fontsizes(ax)
    right = plot_component_reference_curves(ax, component_curve, visible_axis=False)
    right.set_ylim(-FRACTION_PAD, 1.0 + FRACTION_PAD)
    right.set_yticks(FRACTION_TICKS)
    right.tick_params(axis='y', which='major', right=True, labelright=True, left=False, labelleft=False, length=5, width=0.9, direction='in', pad=6, labelsize=TEXT_SIZE)
    right.spines['right'].set_visible(True)
    right.spines['right'].set_color('#444444')
    right.spines['right'].set_linewidth(0.9)
    right.yaxis.set_ticks_position('right')
    right.yaxis.set_label_position('right')
    right.set_ylabel('Component size', fontsize=TEXT_SIZE, rotation=90)
    set_fontsizes(right)
    return right

def legend_row_handles():
    """Legend handles grouped by figure column, with consistent metric colors."""
    names = {'degree': 'Degree', 'betweenness': 'Betweenness', 'k_core': 'K-core', 'clustering_coefficient': 'CC', 'local_efficiency': 'LE'}
    first = [Line2D([0], [0], color=FEATURE_COLORS[key], linewidth=2.0, linestyle='-', label=names[key]) for key in FEATURE_ORDER]
    first += [Line2D([0], [0], color=AP_PROBABILITY_COLOR, linestyle='--', linewidth=2, label='$P_{\\mathrm{Global}}$'), Line2D([0], [0], color=AP_PROBABILITY_COLOR, linestyle='-', linewidth=2, label='$P_{\\mathrm{Explored}}$'), Line2D([0], [0], color=BALANCE_COLOR, linestyle='-', linewidth=2, label='$S_2/S_1$')]
    second = [Line2D([0], [0], color=AP_COUNT_COLOR, linestyle='--', linewidth=2, label='$|AP|_{\\mathrm{Global}}$'), Line2D([0], [0], color=AP_COUNT_COLOR, linestyle='-', linewidth=2, label='$|AP|_{\\mathrm{Explored}}$')]
    third = [Line2D([0], [0], color=HETEROGENEITY_COLOR, linestyle='-', linewidth=2, label='$\\langle s^2\\rangle/\\langle s\\rangle^2$'), Line2D([0], [0], color=COMPONENT_COLOR, linestyle='--', linewidth=2, label='$S_1/N$'), Line2D([0], [0], color=COMPONENT_COLOR, linestyle='-', linewidth=2, label='$S_2/N$')]
    return (first, second, third)

def plot_network_row(axes, network_type: str, labels: list[str], type_cache: pd.DataFrame, h_cache: pd.DataFrame, network_sizes: dict[str, int], panel_labels: tuple[str, str, str], show_xlabel: bool) -> None:
    if len(labels) != len(NETWORK_REPLICATES):
        raise ValueError(f'{network_type}: expected {len(NETWORK_REPLICATES)} network replicates, found {len(labels)}.')
    bootstrap_indices = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(labels), size=(N_BOOT, len(labels)))
    if len({network_sizes[label] for label in labels}) != 1:
        print(f'Note: {network_type} has varying initial component sizes; curves are interpolated.')
    full_grid = common_fraction_grid(labels, network_sizes)
    before_grid = full_grid[:-1]
    ap_summary = aggregate_ap_ensemble(type_cache=type_cache, labels=labels, f_grid_before=before_grid, rng=bootstrap_indices)
    difference_summary = aggregate_global_explored_differences_ensemble(type_cache=type_cache, labels=labels, f_grid_before=before_grid, rng=bootstrap_indices)
    cluster_summary = aggregate_cluster_moments_ensemble(labels=labels, h_cache=h_cache, f_grid=full_grid, rng=bootstrap_indices)
    component_curve = component_curve_from_summary_ensemble(summary=SUMMARY_DATA, labels=labels, f_grid=full_grid, rng=bootstrap_indices)
    ax1, ax2, ax3 = axes
    plot_structural_difference(ax1, difference_summary, ap_summary)
    add_components_and_balance_to_panel1(ax1, cluster_summary)
    plot_ap_availability(ax2, ap_summary)
    ax3_right = plot_combined_cluster_statistics(ax3, cluster_summary, component_curve)
    for panel_ax in (ax1, ax2, ax3):
        if show_xlabel:
            panel_ax.set_xlabel('Fraction of nodes removed, $f$', fontsize=TEXT_SIZE)
            panel_ax.xaxis.set_label_coords(0.5, -0.13)
        else:
            panel_ax.set_xlabel('')
        panel_ax.set_box_aspect(1)
        set_fontsizes(panel_ax)
    ax3_right.set_box_aspect(1)
    for panel_label, panel_ax in zip(panel_labels, (ax1, ax2, ax3)):
        panel_ax.text(0.93, 0.93, panel_label, transform=panel_ax.transAxes, fontsize=TEXT_SIZE, fontweight='bold', fontname='Times New Roman', ha='right', va='top', clip_on=True, zorder=120)
    ax1.text(-0.25, 0.5, network_type, transform=ax1.transAxes, fontsize=TEXT_SIZE, fontweight='bold', fontname='Times New Roman', ha='right', va='center', rotation=0, clip_on=False, zorder=130)

def plot_network_group(group_index: int, network_types: list[str], labels_by_type: dict[str, list[str]], h_cache: pd.DataFrame, network_sizes: dict[str, int], mechanism_cache: pd.DataFrame) -> None:
    configure_matplotlib()
    nrows = len(network_types)
    if nrows not in (2, 4):
        raise ValueError('Grouped Figure_synthetic20 must contain either 2 or 4 network families.')
    figure_width_in = FIGURE_WIDTH_IN
    figure_height_in = REFERENCE_TOP_MARGIN_IN + nrows * PANEL_WIDTH_IN + (nrows - 1) * REFERENCE_ROW_GAP_IN + REFERENCE_BOTTOM_MARGIN_IN
    fig = plt.figure(figsize=(figure_width_in, figure_height_in), facecolor='none')
    fig.patch.set_alpha(0.0)
    side_margin_in = (figure_width_in - THREE_PANEL_WIDTH_IN) / 2.0
    if side_margin_in < 0:
        raise RuntimeError('Figure width is too narrow for three square panels.')
    grid_left = side_margin_in / figure_width_in
    grid_right = 1.0 - grid_left
    grid_wspace = PANEL_GAP_IN / PANEL_WIDTH_IN
    grid_bottom = REFERENCE_BOTTOM_MARGIN_IN / figure_height_in
    grid_top = 1.0 - REFERENCE_TOP_MARGIN_IN / figure_height_in
    gs = fig.add_gridspec(nrows, 3, left=grid_left, right=grid_right, bottom=grid_bottom, top=grid_top, wspace=grid_wspace, hspace=REFERENCE_ROW_GAP_IN / PANEL_WIDTH_IN)
    for row, network_type in enumerate(network_types):
        type_labels = labels_by_type[network_type]
        type_cache = mechanism_cache.loc[mechanism_cache['label'].isin(type_labels)]
        axes = [fig.add_subplot(gs[row, col]) for col in range(3)]
        panel_start = 3 * row
        panel_labels = tuple((chr(ord('A') + panel_start + offset) for offset in range(3)))
        plot_network_row(axes=axes, network_type=network_type, labels=type_labels, type_cache=type_cache, h_cache=h_cache, network_sizes=network_sizes, panel_labels=panel_labels, show_xlabel=row == nrows - 1)
    first, second, third = legend_row_handles()
    reading_order = first + second + third
    assert len(reading_order) == 13
    top_row = reading_order[:7]
    bottom_row = reading_order[7:]
    legend_common = dict(loc='center', bbox_transform=fig.transFigure, frameon=False, fontsize=TEXT_SIZE, handlelength=1.65, handletextpad=0.35, columnspacing=0.65, labelspacing=0.5, borderaxespad=0.0)
    for handles, columns, height in ((top_row, 7, 0.85), (bottom_row, 6, 0.28)):
        legend = fig.legend(handles=handles, ncol=columns, bbox_to_anchor=(0.5, height / figure_height_in), **legend_common)
        legend.set_zorder(100)
    SVG_DIR.mkdir(parents=True, exist_ok=True)
    svg_file = SVG_DIR / f'network_group{group_index}.svg'
    fig.savefig(svg_file, format='svg', bbox_inches='tight', transparent=True)
    plt.close(fig)
    # Show each completed figure immediately; keep earlier figures visible.
    print(f'Saved: {svg_file.name}', flush=True)
    display(SVG(filename=str(svg_file)))
    sys.stdout.flush()
    sys.stderr.flush()


# ============================================================
# 7. Main Workflow
# ============================================================
def print_heading(title: str) -> None:
    print('=' * 92)
    print(title)
    print('=' * 92)

def main():
    global SUMMARY_DATA
    require_fastparquet()
    print_heading('SEAMLESS synthetic20')
    if not SUMMARY_FILE.exists():
        raise FileNotFoundError(SUMMARY_FILE)
    summary_columns = ['label', 'network_type', 'network_replicate', 'protocol', 'm', 'removed', 'p', 'mean_S1_over_N', 'mean_S2_over_N', 'removed_node_label', 'seed_node_labels']
    SUMMARY_DATA = pd.read_csv(SUMMARY_FILE, usecols=summary_columns)
    validate_summary(SUMMARY_DATA)
    labels_by_type = requested_labels_by_type(SUMMARY_DATA)
    summary_by_label = dict(tuple(SUMMARY_DATA.groupby('label', sort=False)))
    graphs, sizes, histories = ({}, {}, {})
    for kind, labels in labels_by_type.items():
        for label in labels:
            graph = read_integer_graph(NETWORK_FILES[label])
            graphs[label], sizes[label] = (graph, len(graph))
            NETWORK_NODE_COUNTS[label] = len(graph)
            rows = summary_by_label[label]
            histories[label] = build_step_history(rows)
            if set(histories[label]) != set(range(len(graph) + 1)):
                raise ValueError(f'Incomplete removal history: {label}')
    ensure_cache_matches_inputs()
    validate_existing_cache_schema()
    keys = load_cache_keys() if CACHE_FILE.exists() else pd.DataFrame(columns=['label', 'scope', 'stochastic_id', 'step'])
    completed = completed_realizations(keys, sizes)
    # Completed attacks are reused; missing attacks are computed normally.
    # Existing incomplete mechanism attacks are not automatically repaired.
    existing_attacks = {
        (str(label), int(sid))
        for label, sid in keys[['label', 'stochastic_id']].itertuples(index=False, name=None)
    }
    incomplete_attacks = existing_attacks - completed
    if incomplete_attacks:
        sample = sorted(incomplete_attacks)[:5]
        raise ValueError(
            f'Mechanism cache contains {len(incomplete_attacks)} incomplete attacks '
            f'(examples: {sample}). Existing cache is not modified automatically.'
        )
    completed_h = prepare_h_cache(sizes)
    require_parallel_backend()
    env = {key: '1' for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}
    executor = LokyProcessPoolExecutor(max_workers=N_WORKERS, env=env) if N_WORKERS > 1 else None
    # Plot each group as soon as its last family finishes, without waiting
    # for computations belonging to later groups.
    family_to_group = {kind: group_index for group_index, kinds in enumerate(NETWORK_GROUPS, 1) for kind in kinds}
    try:
        for family_index, (kind, labels) in enumerate(labels_by_type.items(), start=1):
            print()
            print_heading(f'Family {family_index}/{len(labels_by_type)} | {kind} | r1-r20')
            with tqdm(total=len(labels), desc=f'{kind}', leave=True, ascii=True, ncols=52, bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}') as progress:
                for label in labels:
                    compute_missing_realizations_for_network(label=label, initial_graph=graphs[label], history=histories[label], completed=completed, completed_h=completed_h, executor=executor)
                    if any(((label, sid) not in completed or (label, sid) not in completed_h for sid in range(1, N_ATTACKS + 1))):
                        raise RuntimeError(f'Incomplete cache: {label}')
                    progress.update(1)
            group_index = family_to_group[kind]
            group_kinds = NETWORK_GROUPS[group_index - 1]
            if kind == group_kinds[-1]:
                group_labels = [label for group_kind in group_kinds for label in labels_by_type[group_kind]]
                print_heading(f'Group {group_index}/3 complete | Rendering network_group{group_index}.svg')
                mechanism_cache = load_mechanism_cache(group_labels)
                h_cache = load_h_cache(group_labels)
                plot_network_group(group_index, group_kinds, labels_by_type, h_cache, sizes, mechanism_cache)
                print(f'Saved and displayed: network_group{group_index}.svg', flush=True)
                del mechanism_cache, h_cache
    finally:
        if executor is not None:
            executor.shutdown(wait=True, kill_workers=False)
if __name__ == '__main__':
    main()
