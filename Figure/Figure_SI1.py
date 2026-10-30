"""SEAMLESS Figure SI1: performance, empirical runtime, and computational complexity.
Data
----
A: Dismantling performance across ten synthetic network models.
B: SEAMLESS performance for m = 1-20.
C: Empirical runtime versus performance (m=4).
D: Theoretical computational cost versus performance.
E: Theoretical complexity scaling with network size.
Input
-----
Place this script in results_new/.
    results_synthetic/{network_type}/*/summary.csv
    results_synthetic/runtime.csv
    benchmarks/{network_type}_N256/*.edges
Output
------
    seamless_figures/cost.svg
"""


# ============================================================
# 1. Imports and Configuration
# ============================================================
from __future__ import annotations
import math
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import LogFormatterMathtext, NullFormatter
from matplotlib.transforms import Bbox
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()
RESULTS_DIR = BASE_DIR / 'results_synthetic'
BENCHMARK_DIR = BASE_DIR / 'benchmarks'
RUNTIME_CSV = RESULTS_DIR / 'runtime.csv'
# This order also determines the vertical entries in each legend column.
NETWORK_TYPES = ['BA', 'ER', 'HSBM4', 'LFR', 'RGG', 'SBM4', 'SBM8', 'SBM16', 'SF2', 'WS']
METHOD_ORDER = ['Degree', 'Adaptive degree', 'Betweenness', 'Adaptive betweenness', 'SEAMLESS']
NON_SEAMLESS_METHODS = ['Degree', 'Adaptive degree', 'Betweenness', 'Adaptive betweenness']
METHOD_TICK_LABELS = ['Degree', 'Adaptive\ndegree', 'Betweenness', 'Adaptive\nbetweenness', 'SEAMLESS']
PROTOCOL_TO_METHOD = {'degree': 'Degree', 'adap_degree': 'Adaptive degree', 'betweenness': 'Betweenness', 'adap_betweenness': 'Adaptive betweenness', 'seamless': 'SEAMLESS'}
METHOD_MARKERS = {'Degree': 'v', 'Adaptive degree': '^', 'Betweenness': 'D', 'Adaptive betweenness': '*', 'SEAMLESS': 'o'}
FIXED_M_FOR_PANEL3 = 4
PANEL5_SEAMLESS_M_VALUES = [4, 16, 64, 256]
PANEL5_N_MIN = 10.0
PANEL5_N_MAX = 1000000.0
PANEL5_N_POINTS = 241
TEXT_SIZE = 26
SCATTER_SIZE = 150
POINT_MARKER_SIZE = math.sqrt(SCATTER_SIZE)
X_GAP_RATIO = 0.04
NETWORK_COLORS = {'BA': '#88CCEE', 'ER': '#882255', 'HSBM4': '#999933', 'LFR': '#5B8E7D', 'RGG': '#AA4499', 'SBM4': '#332288', 'SBM8': '#CC6677', 'SBM16': '#6F8FA6', 'SF2': '#117733', 'WS': '#A66F5B'}
PANEL_WIDTH_IN = 4.6315
PANEL_GAP_IN = 1.2913
REFERENCE_XLABEL_Y = -0.13
SEAMLESS_SCALING_COLORS = {4: '#3B2E5A', 16: '#684B8A', 64: '#9A6AB4', 256: '#C990D1'}
SCALING_MARKEVERY = {'Degree': (0, 48), 'Adaptive degree': (8, 48), 'Betweenness': (24, 48), 'Adaptive betweenness': (16, 48), 'SEAMLESS ($m=4$)': (4, 48), 'SEAMLESS ($m=16$)': (12, 48), 'SEAMLESS ($m=64$)': (28, 48), 'SEAMLESS ($m=256$)': (36, 48)}


# ============================================================
# 2. Data Loading
# ============================================================
def compute_one_minus_auc(curve_df: pd.DataFrame) -> float:
    x = pd.to_numeric(curve_df['p'], errors='coerce').to_numpy(dtype=float)
    y = pd.to_numeric(curve_df['mean_S1_over_N'], errors='coerce').to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 2:
        return float('nan')
    order = np.argsort(x)
    integrate = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    return 1.0 - float(integrate(y[order], x[order]))


def parse_instance_name(result_dir_name: str, network_type: str) -> str:
    prefix = f'results_{network_type}_N256_'
    if result_dir_name.startswith(prefix):
        return result_dir_name[len(prefix):]
    parts = result_dir_name.split('_')
    if len(parts) >= 2:
        return '_'.join(parts[-2:])
    return result_dir_name


def read_lcc_stats(edge_path: Path) -> dict[str, float]:
    try:
        graph = nx.read_edgelist(edge_path, nodetype=int, data=False, create_using=nx.Graph())
    except Exception as exc:
        raise RuntimeError(f'Failed to read edge file: {edge_path}') from exc
    graph.remove_edges_from(nx.selfloop_edges(graph))
    if graph.number_of_nodes() == 0:
        raise RuntimeError(f'Empty graph: {edge_path}')
    largest_cc_nodes = max(nx.connected_components(graph), key=len)
    lcc = graph.subgraph(largest_cc_nodes).copy()
    return {'N': float(lcc.number_of_nodes()), 'M': float(lcc.number_of_edges())}


def collect_instance_results() -> pd.DataFrame:
    records = []
    if not RESULTS_DIR.exists():
        raise FileNotFoundError(RESULTS_DIR)

    for network in NETWORK_TYPES:
        root = RESULTS_DIR / network
        if not root.exists():
            warnings.warn(f'Missing directory: {root}')
            continue
        for folder in sorted(p for p in root.iterdir() if p.is_dir()):
            file = folder / 'summary.csv'
            if not file.exists():
                warnings.warn(f'Missing summary.csv: {file}')
                continue
            try:
                df = pd.read_csv(file)
            except Exception as exc:
                warnings.warn(f'Cannot read {file}: {exc}')
                continue
            if not {'protocol', 'p', 'mean_S1_over_N'}.issubset(df):
                warnings.warn(f'Missing required columns: {file}')
                continue

            instance = parse_instance_name(folder.name, network)
            df = df[df['protocol'].isin(PROTOCOL_TO_METHOD)].copy()
            df['m'] = pd.to_numeric(df['m'], errors='coerce') if 'm' in df else np.nan
            df.loc[df['protocol'] != 'seamless', 'm'] = np.nan
            for (protocol, m), group in df.groupby(['protocol', 'm'], dropna=False):
                if protocol == 'seamless' and not np.isfinite(m):
                    continue
                score = compute_one_minus_auc(group)
                if np.isfinite(score):
                    records.append(dict(network_type=network, instance=instance,
                                        protocol=protocol, method=PROTOCOL_TO_METHOD[protocol],
                                        m=int(m) if np.isfinite(m) else np.nan,
                                        one_minus_auc=score))

    if not records:
        raise RuntimeError('No valid result data were collected.')
    return pd.DataFrame(records)

def collect_graph_stats() -> pd.DataFrame:
    records: list[dict] = []
    if not BENCHMARK_DIR.exists():
        raise FileNotFoundError(f'Missing directory: {BENCHMARK_DIR}')
    for network_type in NETWORK_TYPES:
        type_dir = BENCHMARK_DIR / f'{network_type}_N256'
        if not type_dir.exists():
            warnings.warn(f'Missing benchmark directory: {type_dir}')
            continue
        for edge_path in sorted(type_dir.glob('*.edges')):
            try:
                stats = read_lcc_stats(edge_path)
            except Exception as exc:
                warnings.warn(str(exc))
                continue
            records.append({'network_type': network_type, 'instance': edge_path.stem, 'N': stats['N'], 'M': stats['M']})
    if not records:
        raise RuntimeError('No valid benchmark networks were collected.')
    return pd.DataFrame(records)


# ============================================================
# 3. Data Processing
# ============================================================
def summarize_method_performance(instance_df: pd.DataFrame) -> pd.DataFrame:
    non_seamless = instance_df[instance_df['method'] != 'SEAMLESS'].groupby(['network_type', 'method'], as_index=False)['one_minus_auc'].mean()
    seamless = instance_df[(instance_df['method'] == 'SEAMLESS') & instance_df['m'].between(1, 20)].groupby(['network_type', 'm'], as_index=False)['one_minus_auc'].mean().groupby('network_type', as_index=False)['one_minus_auc'].mean()
    seamless['method'] = 'SEAMLESS'
    return pd.concat([non_seamless, seamless], ignore_index=True)


def summarize_seamless_by_m(instance_df: pd.DataFrame) -> pd.DataFrame:
    seamless_df = instance_df[(instance_df['method'] == 'SEAMLESS') & instance_df['m'].between(1, 20, inclusive='both')]
    return seamless_df.groupby(['network_type', 'm'], as_index=False)['one_minus_auc'].mean().sort_values(['network_type', 'm'])


def select_fixed_m(instance_df: pd.DataFrame) -> pd.DataFrame:
    return instance_df[(instance_df['method'] != 'SEAMLESS') |
                       (instance_df['m'] == FIXED_M_FOR_PANEL3)]


def require_100_instances(grouped, label: str) -> None:
    expected = pd.MultiIndex.from_product(
        [NETWORK_TYPES, METHOD_ORDER], names=['network_type', 'method'])
    if not grouped['instance'].nunique().reindex(expected).eq(100).all():
        raise ValueError(f'{label} requires exactly 100 instances per network/method.')


def summarize_runtime_vs_performance(instance_df: pd.DataFrame) -> pd.DataFrame:
    selected = select_fixed_m(instance_df)
    performance = selected.groupby(['network_type', 'method'], as_index=False)['one_minus_auc'].mean()

    if not RUNTIME_CSV.exists():
        raise FileNotFoundError(RUNTIME_CSV)
    runtime = pd.read_csv(RUNTIME_CSV)
    required = {'network_type', 'instance', 'method', 'mean_runtime_s'}
    if not required.issubset(runtime):
        raise ValueError(f'Runtime CSV missing columns: {required - set(runtime)}')
    runtime = runtime[runtime['network_type'].isin(NETWORK_TYPES) &
                      runtime['method'].isin(PROTOCOL_TO_METHOD)].copy()
    if 'm' in runtime:
        m = pd.to_numeric(runtime['m'], errors='coerce')
        runtime = runtime[(runtime['method'] != 'seamless') | (m == FIXED_M_FOR_PANEL3)]
    runtime['method'] = runtime['method'].map(PROTOCOL_TO_METHOD)
    runtime['mean_runtime_s'] = pd.to_numeric(runtime['mean_runtime_s'], errors='coerce')
    runtime = runtime.dropna(subset=['network_type', 'instance', 'method', 'mean_runtime_s'])
    keys = ['network_type', 'instance', 'method']
    if runtime.duplicated(keys).any():
        raise ValueError('Duplicate runtime rows for network_type × instance × method.')

    grouped = runtime.groupby(['network_type', 'method'])
    require_100_instances(grouped, 'Panel C')
    totals = grouped['mean_runtime_s'].sum().rename('estimated_runtime_100_s').reset_index()
    totals['n_runtime_networks'] = 100
    return performance.merge(totals, on=['network_type', 'method'], how='inner')

def summarize_theoretical_cost(instance_df: pd.DataFrame, graph_stats_df: pd.DataFrame) -> pd.DataFrame:
    attack = select_fixed_m(instance_df).copy()
    if attack.empty:
        raise RuntimeError('No attack results for panel D.')
    required = {'network_type', 'instance', 'N', 'M'}
    if not required.issubset(graph_stats_df):
        raise ValueError(f'Graph statistics missing columns: {required - set(graph_stats_df)}')
    graph = graph_stats_df.copy()
    graph[['N', 'M']] = graph[['N', 'M']].apply(pd.to_numeric, errors='coerce')
    graph = graph.dropna(subset=list(required))
    if graph.duplicated(['network_type', 'instance']).any():
        raise ValueError('Duplicate benchmark network statistics.')
    merged = attack.merge(graph, on=['network_type', 'instance'], how='left', validate='many_to_one')
    if merged[['N', 'M']].isna().any().any():
        raise ValueError('Some attack instances have no matching benchmark network.')

    n, m = merged['N'], merged['M']
    formulas = {'Degree': n + m, 'Adaptive degree': n**2 + m,
                'Betweenness': n*m, 'Adaptive betweenness': n**2*m,
                'SEAMLESS': FIXED_M_FOR_PANEL3*m**3/n**2}
    merged['theoretical_cost'] = np.select(
        [merged['method'].eq(method) for method in METHOD_ORDER],
        [formulas[method] for method in METHOD_ORDER], default=np.nan)
    if not np.isfinite(merged['theoretical_cost']).all():
        raise ValueError('Non-finite theoretical cost.')

    grouped = merged.groupby(['network_type', 'method'])
    require_100_instances(grouped, 'Panel D')
    result = grouped.agg(theoretical_cost_100=('theoretical_cost', 'sum'),
                         one_minus_auc=('one_minus_auc', 'mean')).reset_index()
    result['n_networks'] = 100
    return result

def theoretical_scaling_curves() -> pd.DataFrame:
    n = np.logspace(math.log10(PANEL5_N_MIN), math.log10(PANEL5_N_MAX), PANEL5_N_POINTS)
    formulas = {'Degree': n, 'Adaptive degree': n**2,
                'Betweenness': n**2, 'Adaptive betweenness': n**3}
    series = [(method, method, np.nan, values) for method, values in formulas.items()]
    series += [('SEAMLESS', f'SEAMLESS ($m={m}$)', m, m*n)
               for m in PANEL5_SEAMLESS_M_VALUES]
    return pd.concat([pd.DataFrame({'method': method, 'series_label': label,
                                    'm': m, 'N': n, 'theoretical_complexity': values})
                      for method, label, m, values in series], ignore_index=True)

# ============================================================
# 4. Plot Styling
# ============================================================
def configure_matplotlib() -> None:
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['text.usetex'] = False
    plt.rcParams['svg.fonttype'] = 'path'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['axes.unicode_minus'] = False


def style_axis(ax: plt.Axes) -> None:
    ax.set_box_aspect(1)
    ax.set_facecolor('none')
    ax.patch.set_alpha(0.0)
    ax.grid(True, color='#D9D9D9', linewidth=0.8, alpha=0.9)
    ax.tick_params(axis='both', which='major', labelsize=TEXT_SIZE, length=5, width=0.9, direction='out')
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontname('Times New Roman')
        tick_label.set_fontsize(TEXT_SIZE)
    for spine in ax.spines.values():
        spine.set_color('#444444')
        spine.set_linewidth(0.9)


def set_right_x_gap(ax: plt.Axes, last_tick: float, ratio: float=X_GAP_RATIO, log: bool=False) -> None:
    left = ax.get_xlim()[0]
    if log:
        left, last_tick = (math.log10(left), math.log10(last_tick))
        right = (last_tick - ratio * left) / (1.0 - ratio)
        ax.set_xlim(10.0 ** left, 10.0 ** right)
    else:
        right = (last_tick - ratio * left) / (1.0 - ratio)
        ax.set_xlim(left, right)


def set_symmetric_x_gap(ax: plt.Axes, first_tick: float, last_tick: float, ratio: float=X_GAP_RATIO) -> None:
    pad = ratio * (last_tick - first_tick) / (1.0 - 2.0 * ratio)
    ax.set_xlim(first_tick - pad, last_tick + pad)


def apply_upper_reference_performance_yaxis(ax: plt.Axes) -> None:
    ticks = np.linspace(0.5, 0.8, 4)
    half_step = (ticks[1] - ticks[0]) / 2
    ax.set_ylim(ticks[0] - half_step, ticks[-1] + half_step)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f'{value:.3f}'.rstrip('0').rstrip('.') for value in ticks])


def apply_upper_reference_runtime_xaxis(ax: plt.Axes, seamless_e_df: pd.DataFrame) -> None:
    ax.set_xscale('log')
    finite_runtimes = pd.to_numeric(seamless_e_df['estimated_runtime_100_s'], errors='coerce').to_numpy(dtype=float)
    finite_runtimes = finite_runtimes[np.isfinite(finite_runtimes) & (finite_runtimes > 0)]
    if finite_runtimes.size > 0:
        x_min = float(np.nanmin(finite_runtimes))
        log_min = math.log10(x_min)
        ax.set_xlim(10 ** (log_min - 0.2), 10 ** 3.16)
    ax.set_xticks([0.1, 1.0, 10.0, 100.0, 1000.0])
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis='x', which='minor', bottom=False, top=False)


def apply_lower_log10_scaling_axes(ax: plt.Axes, cost_b_df: pd.DataFrame) -> None:
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    set_symmetric_x_gap(ax, 1.0, 6.0)
    finite_costs = pd.to_numeric(cost_b_df['theoretical_complexity'], errors='coerce').to_numpy(dtype=float)
    finite_costs = finite_costs[np.isfinite(finite_costs) & (finite_costs > 0)]
    if finite_costs.size == 0:
        return
    log_costs = np.log10(finite_costs)
    data_min = float(np.nanmin(log_costs))
    data_max = float(np.nanmax(log_costs))
    first_tick = int(math.floor(data_min))
    last_needed = int(math.ceil(data_max))
    step = max(1, int(math.ceil((last_needed - first_tick) / 3.0)))
    y_ticks = np.arange(first_tick, first_tick + 4 * step, step, dtype=float)
    half_step = float(step) / 2
    ax.set_ylim(float(y_ticks[0] - half_step), float(y_ticks[-1] + half_step))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(int(value)) for value in y_ticks], fontname='Times New Roman', fontsize=TEXT_SIZE)


def add_seamless_m_legend_inside(ax: plt.Axes):
    from matplotlib.lines import Line2D
    handles = []
    for m_value in PANEL5_SEAMLESS_M_VALUES:
        color = SEAMLESS_SCALING_COLORS[m_value]
        handles.append(Line2D([0], [0], color=color, linewidth=1.8, linestyle='-', marker=METHOD_MARKERS['SEAMLESS'], markersize=POINT_MARKER_SIZE, markerfacecolor=color, markeredgecolor=color, markeredgewidth=1.5, label=f'$m={m_value}$'))
    legend = ax.legend(handles=handles, loc='upper left', frameon=False, fontsize=TEXT_SIZE - 3, handlelength=1.15, handletextpad=0.4, borderaxespad=0.55, labelspacing=0.25, ncol=1)
    return legend


# ============================================================
# 5. Figure Panels
# ============================================================
def draw_summary_bar_panel(ax: plt.Axes, summary: pd.DataFrame,
                           color_map: dict[str, object]) -> None:
    """Draw network contributions for each method in a stacked horizontal bar."""
    values = (summary.pivot(index='method', columns='network_type',
                            values='one_minus_auc')
              .reindex(index=METHOD_ORDER, columns=NETWORK_TYPES).fillna(0.0))
    y_positions = np.arange(len(METHOD_ORDER))
    cumulative_left = np.zeros(len(METHOD_ORDER), dtype=float)

    for network_type in NETWORK_TYPES:
        widths = values[network_type].to_numpy(dtype=float)
        ax.barh(y_positions, widths, left=cumulative_left, height=0.56,
                color=color_map[network_type], edgecolor='none', zorder=3)
        cumulative_left += widths

    ax.set_yticks(y_positions)
    ax.set_yticklabels(METHOD_TICK_LABELS, fontname='Times New Roman',
                       fontsize=TEXT_SIZE, multialignment='center')
    ax.invert_yaxis()
    ax.set_xlim(0.0, 8.32)
    ax.set_xticks([2, 4, 6, 8])
    ax.set_xlabel('$1-\\mathrm{AUC}\\,(S_1/N)$', fontsize=TEXT_SIZE, labelpad=8)

    for i, method in enumerate(METHOD_ORDER):
        ax.plot([cumulative_left[i] + 0.18], [y_positions[i]],
                marker=METHOD_MARKERS[method], markersize=POINT_MARKER_SIZE,
                markerfacecolor='none', markeredgecolor='black',
                markeredgewidth=1.5, linestyle='', clip_on=False, zorder=5)


def draw_m_panel(ax: plt.Axes, seamless_d_df: pd.DataFrame, color_map: dict[str, object]) -> None:
    for network_type, sub_df in seamless_d_df.groupby('network_type', sort=False):
        sub_df = sub_df.sort_values('m')
        ax.plot(sub_df['m'], sub_df['one_minus_auc'], marker='o', markersize=POINT_MARKER_SIZE, linewidth=1.5, color=color_map[network_type], markerfacecolor=color_map[network_type], markeredgecolor=color_map[network_type], zorder=3)
    ax.set_xlim(0.0, 20.8)
    ax.set_xticks([4, 8, 12, 16, 20])
    apply_upper_reference_performance_yaxis(ax)
    ax.set_xlabel('$m$', fontsize=TEXT_SIZE, labelpad=8)
    ax.set_ylabel('$1-\\mathrm{AUC}\\,(S_1/N)$', fontsize=TEXT_SIZE, labelpad=8)


def draw_method_scatter(ax: plt.Axes, df: pd.DataFrame, x_col: str,
                        color_map: dict[str, object], log10_x: bool = False) -> None:
    """Draw method-shaped, network-colored points for panels C and D."""
    for row in df.itertuples(index=False):
        x = getattr(row, x_col)
        if not np.isfinite(x) or x <= 0:
            continue

        x_position = math.log10(x) if log10_x else x
        ax.scatter(x_position, row.one_minus_auc, s=SCATTER_SIZE,
                   marker=METHOD_MARKERS[row.method],
                   color=color_map[row.network_type], edgecolors='none', zorder=3)


def draw_scaling_panel(ax: plt.Axes, cost_b_df: pd.DataFrame) -> None:
    for label, group in cost_b_df.groupby('series_label', sort=False):
        group = group.sort_values('N')
        is_seamless = label.startswith('SEAMLESS')
        method = 'SEAMLESS' if is_seamless else label
        color = SEAMLESS_SCALING_COLORS[int(group['m'].iloc[0])] if is_seamless else 'black'
        ax.plot(np.log10(group['N']), np.log10(group['theoretical_complexity']),
                linewidth=1.8, linestyle='-', marker=METHOD_MARKERS[method],
                markersize=POINT_MARKER_SIZE, markevery=SCALING_MARKEVERY[label],
                color=color, markerfacecolor=color if is_seamless else 'none',
                markeredgecolor=color, markeredgewidth=1.5,
                label=label, zorder=4 if is_seamless else 3)
    apply_lower_log10_scaling_axes(ax, cost_b_df)
    ax.set_xlabel(r'$\log_{10}(N)$', fontsize=TEXT_SIZE, labelpad=8)
    ax.set_ylabel(r'$\log_{10}(\mathcal{C})$', fontsize=TEXT_SIZE, labelpad=8)

# ============================================================
# 6. Figure Assembly and Export
# ============================================================
def build_cost_abcde_figure(seamless_c_df, seamless_d_df, seamless_e_df, cost_a_df, cost_b_df, color_map):
    fig = plt.figure(figsize=(18.0, 12.35), facecolor='none')
    fig.patch.set_alpha(0.0)
    panel_w = PANEL_WIDTH_IN
    panel_gap = PANEL_GAP_IN
    width = fig.get_figwidth()
    upper_total = 3 * panel_w + 2 * panel_gap
    upper_x = [(width - upper_total) / 2 + i * (panel_w + panel_gap) for i in range(3)]
    lower_x = upper_x[:2]
    legend_x = upper_x[2]
    y_lower = 0.95
    y_upper = 6.9
    def make_axes(xs, y):
        return [fig.add_axes([x / width, y / fig.get_figheight(),
                              panel_w / width, panel_w / fig.get_figheight()]) for x in xs]

    upper_axes, lower_axes = make_axes(upper_x, y_upper), make_axes(lower_x, y_lower)
    for ax in upper_axes + lower_axes:
        style_axis(ax)
        ax.xaxis.set_label_coords(0.5, REFERENCE_XLABEL_Y)
    ax_a, ax_b, ax_c = upper_axes
    draw_summary_bar_panel(ax_a, seamless_c_df, color_map)
    draw_m_panel(ax_b, seamless_d_df, color_map)
    draw_method_scatter(ax_c, seamless_e_df, 'estimated_runtime_100_s', color_map)
    ax_c.set_xlabel('Empirical runtime (s)', fontsize=TEXT_SIZE, labelpad=8)
    ax_c.set_ylabel('$1-\\mathrm{AUC}\\,(S_1/N)$', fontsize=TEXT_SIZE, labelpad=8)
    apply_upper_reference_performance_yaxis(ax_c)
    apply_upper_reference_runtime_xaxis(ax_c, seamless_e_df)
    set_right_x_gap(ax_a, 8.0)
    set_right_x_gap(ax_b, 20.0)
    set_right_x_gap(ax_c, 1000.0, log=True)
    ax_d, ax_e = lower_axes
    draw_method_scatter(ax_d, cost_a_df, 'theoretical_cost_100', color_map, log10_x=True)
    ax_d.set_xticks([5, 6, 7, 8, 9, 10])
    set_symmetric_x_gap(ax_d, 5.0, 10.0)
    apply_upper_reference_performance_yaxis(ax_d)
    ax_d.set_xlabel('$\\log_{10}(\\mathcal{C})$', fontsize=TEXT_SIZE, labelpad=8)
    ax_d.set_ylabel('$1-\\mathrm{AUC}\\,(S_1/N)$', fontsize=TEXT_SIZE, labelpad=8)
    draw_scaling_panel(ax_e, cost_b_df)
    add_seamless_m_legend_inside(ax_e)
    legend_ax = fig.add_axes([legend_x / width, y_lower / fig.get_figheight(), panel_w / width, panel_w / fig.get_figheight()])
    legend_ax.set_axis_off()
    handles = [Patch(facecolor=color_map[name], edgecolor='none', label=name)
               for name in NETWORK_TYPES]
    legend_ax.legend(handles=handles, loc='center', bbox_to_anchor=(0.5, 0.5), bbox_transform=legend_ax.transAxes, frameon=False, fontsize=TEXT_SIZE, ncol=2, columnspacing=1.5, handlelength=0.95, handleheight=0.85, handletextpad=0.4, labelspacing=0.35, borderaxespad=0.0)
    for letter, ax in zip('ABCDE', upper_axes + lower_axes):
        box = ax.get_position()
        fig.text(box.x0 + 0.96 * box.width, box.y0 + 0.035 * box.height, letter, ha='right', va='bottom', fontsize=TEXT_SIZE, fontweight='bold', fontname='Times New Roman', zorder=10)
    return fig


# ============================================================
# 7. Main
# ============================================================
def main() -> None:
    configure_matplotlib()
    instance_df = collect_instance_results()
    seamless_c_df = summarize_method_performance(instance_df)
    seamless_d_df = summarize_seamless_by_m(instance_df)
    seamless_e_df = summarize_runtime_vs_performance(instance_df)
    graph_stats_df = collect_graph_stats()
    cost_a_df = summarize_theoretical_cost(instance_df, graph_stats_df)
    cost_b_df = theoretical_scaling_curves()
    color_map = {name: NETWORK_COLORS[name] for name in NETWORK_TYPES}
    cost_fig = build_cost_abcde_figure(seamless_c_df, seamless_d_df, seamless_e_df, cost_a_df, cost_b_df, color_map)
    output_dir = BASE_DIR / 'seamless_figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    # Include axes, legends, and panel labels in the exported SVG bounds.
    cost_fig.patch.set_alpha(0)
    cost_fig.canvas.draw()
    renderer = cost_fig.canvas.get_renderer()
    boxes = []
    for ax in cost_fig.axes:
        if ax.axison:
            boxes.append(ax.get_tightbbox(renderer))
        else:
            boxes.extend(child.get_window_extent(renderer) for child in ax.get_children()
                         if isinstance(child, plt.matplotlib.legend.Legend) and child.get_visible())
    boxes.extend(artist.get_window_extent(renderer) for artist in cost_fig.texts + cost_fig.legends
                 if artist.get_visible())
    bbox = Bbox.union([box for box in boxes if box is not None])
    bbox = bbox.transformed(cost_fig.dpi_scale_trans.inverted())
    cost_fig.savefig(output_dir / 'cost.svg', format='svg', bbox_inches=bbox,
                     pad_inches=0, transparent=True)
    plt.show()


if __name__ == "__main__":
    main()
