
"""
Figure_Metabolic: Metabolic network adaptive fragility and biological associations.

Read precomputed attack curves for five strategies (SEAMLESS: m = 4).
Analyze 1 - AUC(S2/N), biological features, and regression statistics.
Plot six panels (A-F), including representative species and group insets.

Read existing results only; never rerun SEAMLESS attacks.
Output: seamless_figures/Metabolic_seamless.svg (Panels A-F).
"""
from __future__ import annotations
import json
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from scipy.stats import linregress


# ============================================================
# 1. Configuration
# ============================================================
BASE_DIR = Path.cwd()
RESULTS_DIR = BASE_DIR / 'results_metabolic'
FEATURE_CANDIDATES = [BASE_DIR / 'benchmarks' / 'Metabolic_SI' / 'metabolomes_KEGG_feature.csv', BASE_DIR / 'metabolomes_KEGG_feature.csv']
NODE_SIZE_MIN = 90.0
NODE_SIZE_MAX = 260.0
NETWORK_N_COL = 'Network_N'
MARKER_SIZE_COL = '__marker_size'
STAT_PROTOCOLS = ['seamless', 'degree', 'adap_degree', 'betweenness', 'adap_betweenness']
PROTOCOL_DISPLAY_NAMES = {'degree': 'Degree', 'adap_degree': 'Adaptive degree', 'betweenness': 'Betweenness', 'adap_betweenness': 'Adaptive betweenness', 'seamless': 'SEAMLESS'}
SEAMLESS_M = 4.0
RANDOM_SEED = 11111
METRIC_COL = 'One_minus_AUC_S2'
METRIC_LABEL = '$1-\\mathrm{AUC}(S_2/N)$'
FEATURES = [('Protein_coding_genes', 'Protein-coding genes', True), ('Unique_KEGG_Orthology_groups', 'Unique KOs', True), ('Unique_Enzyme_Commission_numbers', 'Unique EC numbers', True), ('KEGG_pathways', 'KEGG pathways', False), ('KEGG_modules', 'KEGG modules', False)]
TEXT_SIZE = 26
LEGEND_TEXT_SIZE = 23
INSET_TEXT_SIZE = 16
INSET_STATS_TEXT_SIZE = 22
PANEL_WIDTH_IN = 4.6315
PANEL_GAP_IN = 1.2913
REFERENCE_VISIBLE_PANEL_GAP_IN = 0.55
ROW_GAP_IN = 1.8
BOTTOM_MARGIN_IN = 2.28
TOP_MARGIN_IN = 0.6885
FIGSIZE = (18.5, TOP_MARGIN_IN + 2 * PANEL_WIDTH_IN + ROW_GAP_IN + BOTTOM_MARGIN_IN)
SCATTER_GRAY = '#8A8A8A'
SCATTER_SIZE = 180
Y_AXIS_PAD_FRACTION = 0.04
NORMALIZED_Y_TICKS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
BLUE = '#2072AA'
GREEN = '#339A39'
ORANGE = '#D02623'
PANEL1_FILL_COLORS = {BLUE: '#A3C7DA', GREEN: '#AECC8A', ORANGE: '#E49698'}
REPRESENTATIVE_COLOR_BY_GENUS_INITIAL = {'C': ORANGE, 'B': BLUE, 'S': GREEN}
GROUP_INSET_LEFT = 0.06
GROUP_INSET_WIDTH = 0.29
GROUP_INSET_HEIGHT = 0.2
SEAMLESS_X_TICKS = np.array([0.984, 0.987, 0.99, 0.993, 0.996])
SCATTER_COMMON_XLIM = (0.98352, 0.99648)
PROKARYOTE_EDGE_COLOR = 'white'
PROKARYOTE_EDGE_WIDTH = 0.35
EUKARYOTE_EDGE_COLOR = '#333333'
EUKARYOTE_EDGE_WIDTH = 2
INSET_POINT_SIZE = 28
EUKARYOTE_INSET_POINT_SIZE = 24
INSET_LINEWIDTH = 1.15
EUKARYOTE_Y_PAD_FRACTION = 0.1


# ============================================================
# 2. Data Loading and Processing
# ============================================================
def normalize_col_name(value) -> str:
    return re.sub('[^a-z0-9]+', '', str(value).strip().lower())

def find_column(df: pd.DataFrame, candidates: list[str], file_label: str) -> str:
    normalized = {normalize_col_name(column): column for column in df.columns}
    for candidate in candidates:
        key = normalize_col_name(candidate)
        if key in normalized:
            return normalized[key]
    raise ValueError(f'Cannot find expected column in {file_label}: {candidates}\nAvailable columns: {df.columns.tolist()}')

def normalize_p(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size and np.nanmax(values) > 1.5:
        values = values / 100.0
    return values

def trapezoid_auc(x_values: np.ndarray, y_values: np.ndarray) -> float:
    if hasattr(np, 'trapezoid'):
        return float(np.trapezoid(y_values, x_values))
    return float(np.trapz(y_values, x_values))

def find_feature_file() -> Path:
    for path in FEATURE_CANDIDATES:
        if path.exists():
            return path
    searched = '\n'.join((f'  - {path}' for path in FEATURE_CANDIDATES))
    raise FileNotFoundError(f'Cannot find metabolomes_KEGG_feature.csv.\nSearched:\n{searched}')

def load_results():
    """Read each network's config.json and summary.csv once."""
    if not RESULTS_DIR.is_dir():
        raise FileNotFoundError(f'Results directory not found: {RESULTS_DIR}')
    folders = sorted((p for p in RESULTS_DIR.iterdir() if p.is_dir() and p.name.lower().startswith('results_metabolic_')), key=lambda p: p.name.lower())
    records, curves, node_records = [], {}, []
    for folder in folders:
        kegg_id = folder.name[len('results_Metabolic_'):].strip().lower()
        config_path, summary_path = folder / 'config.json', folder / 'summary.csv'
        if not config_path.is_file():
            raise FileNotFoundError(f'Missing config.json: {config_path}')
        with config_path.open(encoding='utf-8') as handle:
            n_nodes = int(json.load(handle)['n_nodes'])
        if n_nodes <= 0:
            raise ValueError(f'Invalid n_nodes in {config_path}: {n_nodes}')
        node_records.append({'KEGG_ID': kegg_id, NETWORK_N_COL: n_nodes})
        if not summary_path.is_file():
            continue
        try:
            summary = pd.read_csv(summary_path)
            summary.columns = [str(c).strip() for c in summary.columns]
            protocol_col = find_column(summary, ['protocol'], str(summary_path))
            p_col = find_column(summary, ['p', 'fraction removed', 'fraction_removed'], str(summary_path))
            s1_col = find_column(summary, ['mean_S1_over_N', 'S1_over_N', 'mean_S1/N', 'S1/N', 'meanS1overN'], str(summary_path))
            s2_col = find_column(summary, ['mean_S2_over_N', 'S2_over_N', 'mean_S2/N', 'S2/N', 'meanS2overN'], str(summary_path))
            protocol_names = summary[protocol_col].astype(str).str.strip().str.lower()
            for protocol in STAT_PROTOCOLS:
                part = summary.loc[protocol_names == protocol].copy()
                if protocol == 'seamless':
                    m_col = find_column(summary, ['m'], str(summary_path))
                    part = part[np.isclose(pd.to_numeric(part[m_col], errors='coerce'), SEAMLESS_M, equal_nan=False)]
                if part.empty:
                    continue
                for col in [p_col, s1_col, s2_col]:
                    part[col] = pd.to_numeric(part[col], errors='coerce')
                part = part.dropna(subset=[p_col, s1_col, s2_col])
                if len(part) < 2:
                    continue
                curve = part.groupby(p_col, as_index=False)[[s1_col, s2_col]].mean().sort_values(p_col)
                curve = curve.rename(columns={p_col: 'p', s1_col: 'S1_over_N', s2_col: 'S2_over_N'})
                curve['p'] = normalize_p(curve['p'].to_numpy(dtype=float))
                curve = curve.sort_values('p').drop_duplicates('p').reset_index(drop=True)
                if len(curve) < 2:
                    continue
                auc = trapezoid_auc(curve['p'].to_numpy(dtype=float), curve['S2_over_N'].to_numpy(dtype=float))
                records.append({'KEGG_ID': kegg_id, 'protocol': protocol, NETWORK_N_COL: n_nodes,
                                'AUC_S2': auc, METRIC_COL: 1.0 - auc})
                if protocol == 'seamless':
                    curves[kegg_id] = curve
        except Exception as exc:
            print(f'[WARNING] Failed: {summary_path}: {exc}')
    if not records:
        raise RuntimeError(f'No valid attack results found under {RESULTS_DIR}')
    results = pd.DataFrame(records).drop_duplicates(['KEGG_ID', 'protocol'], keep='first')
    node_counts = pd.DataFrame(node_records).drop_duplicates('KEGG_ID').copy()
    lo, hi = node_counts[NETWORK_N_COL].min(), node_counts[NETWORK_N_COL].max()
    if hi > lo:
        node_counts[MARKER_SIZE_COL] = NODE_SIZE_MIN + (node_counts[NETWORK_N_COL] - lo) / (hi - lo) * (NODE_SIZE_MAX - NODE_SIZE_MIN)
    else:
        node_counts[MARKER_SIZE_COL] = SCATTER_SIZE
    results = results.drop(columns=NETWORK_N_COL).merge(node_counts, on='KEGG_ID', how='left')
    return results, curves


def prepare_features(feature_file: Path) -> pd.DataFrame:
    features = pd.read_csv(feature_file)
    features.columns = [str(column).strip() for column in features.columns]
    required_columns = ['KEGG_ID', 'Species_name', 'Cellular_group'] + [col for col, _, _ in FEATURES]
    missing_columns = [col for col in required_columns if col not in features.columns]
    if missing_columns:
        raise ValueError('Missing required columns in metabolomes_KEGG_feature.csv:\n' + '\n'.join((f'  - {column}' for column in missing_columns)) + '\n\nAvailable columns:\n' + '\n'.join((f'  - {column}' for column in features.columns)))
    features['KEGG_ID'] = features['KEGG_ID'].astype(str).str.strip().str.lower()
    features['Species_name'] = features['Species_name'].astype(str).str.strip()
    features['Cellular_group'] = features['Cellular_group'].astype(str).str.strip()
    for feature_col, _, use_log10 in FEATURES:
        features[feature_col] = pd.to_numeric(features[feature_col], errors='coerce')
        raw = features[feature_col]
        if use_log10:
            transformed = np.log10(raw.where(raw > 0))
        else:
            transformed = raw.astype(float)
        finite = transformed[np.isfinite(transformed)]
        normalized_col = f'__plot_{feature_col}'
        if finite.empty:
            features[normalized_col] = np.nan
            continue
        y_min = float(finite.min())
        y_max = float(finite.max())
        if y_max > y_min:
            features[normalized_col] = (transformed - y_min) / (y_max - y_min)
        else:
            features[normalized_col] = np.where(np.isfinite(transformed), 0.5, np.nan)
    return features


# ============================================================
# 3. Statistical Analysis
# ============================================================
def format_p_value(p_value: float) -> str:
    if pd.isna(p_value):
        return ''
    if p_value == 0:
        return '$p<10^{-300}$'
    if p_value < 0.01:
        exponent = int(np.floor(np.log10(p_value)))
        mantissa = p_value / 10 ** exponent
        return f'$p={mantissa:.1f}\\times10^{{{exponent}}}$'
    return f'$p={p_value:.2f}$'

def linear_regression_statistics(df: pd.DataFrame, x_col: str, y_col: str):
    plot_data = df[[x_col, y_col]].copy()
    plot_data[x_col] = pd.to_numeric(plot_data[x_col], errors='coerce')
    plot_data[y_col] = pd.to_numeric(plot_data[y_col], errors='coerce')
    plot_data = plot_data.dropna()
    if len(plot_data) < 3:
        return None
    x = plot_data[x_col].to_numpy(dtype=float)
    y = plot_data[y_col].to_numpy(dtype=float)
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return None
    try:
        result = linregress(x, y)
    except Exception as exc:
        print(f'[WARNING] Linear regression failed for {y_col}: {exc}')
        return None
    fitted = result.intercept + result.slope * x
    residuals = y - fitted
    rss = float(np.sum(residuals ** 2))
    n = int(len(plot_data))
    rss_for_likelihood = max(rss, np.finfo(float).tiny)
    sigma2_mle = rss_for_likelihood / n
    log_likelihood = -0.5 * n * (np.log(2.0 * np.pi) + 1.0 + np.log(sigma2_mle))
    k = 3
    aic = 2.0 * k - 2.0 * log_likelihood
    bic = np.log(n) * k - 2.0 * log_likelihood
    return {'slope': float(result.slope), 'intercept': float(result.intercept), 'r_squared': float(result.rvalue ** 2), 'p_value': float(result.pvalue), 'n': n, 'aic': float(aic), 'bic': float(bic)}

def feature_panel_data(merged: pd.DataFrame, feature_col: str, extra_columns=()) -> tuple[pd.DataFrame, str]:
    plot_col = f'__plot_{feature_col}'
    columns = list(dict.fromkeys(['KEGG_ID', 'Cellular_group', METRIC_COL, plot_col, MARKER_SIZE_COL, *extra_columns]))
    data = merged[columns].copy()
    return data.dropna(subset=['KEGG_ID', METRIC_COL, plot_col, MARKER_SIZE_COL]), plot_col

def build_panel_group_model_statistics(merged: pd.DataFrame, *, protocol: str, panel_index: int, feature_col: str, feature_label: str, use_log10: bool) -> pd.DataFrame:
    plot_data, plot_col = feature_panel_data(merged, feature_col)
    group_labels = plot_data['Cellular_group'].astype(str).str.strip().str.lower()
    groups = [('Overall', plot_data), ('Prokaryotes', plot_data[group_labels == 'prokaryotes']), ('Eukaryotes', plot_data[group_labels == 'eukaryotes'])]
    display_name = PROTOCOL_DISPLAY_NAMES.get(protocol, protocol)
    transform_name = 'log10 + global Min-Max' if use_log10 else 'raw + global Min-Max'
    rows = []
    for group_name, group_data in groups:
        stats = linear_regression_statistics(group_data, METRIC_COL, plot_col)
        if stats is None:
            rows.append({'Attack strategy': display_name, 'Protocol': protocol, 'Panel': panel_index, 'Biological feature': feature_label, 'Group': group_name, 'Transform': transform_name, 'n': np.nan, 'R_squared': np.nan, 'p_value': np.nan, 'AIC': np.nan, 'BIC': np.nan})
            continue
        rows.append({'Attack strategy': display_name, 'Protocol': protocol, 'Panel': panel_index, 'Biological feature': feature_label, 'Group': group_name, 'Transform': transform_name, 'n': stats['n'], 'R_squared': stats['r_squared'], 'p_value': stats['p_value'], 'AIC': stats['aic'], 'BIC': stats['bic']})
    return pd.DataFrame(rows)

def collect_attack_comparison_statistics(results: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for protocol in STAT_PROTOCOLS:
        merged = results.loc[results['protocol'] == protocol]
        if merged.empty:
            raise RuntimeError(f'No valid metabolic results for protocol={protocol!r}.')
        for panel_index, (feature_col, feature_label, use_log10) in enumerate(FEATURES, start=2):
            frames.append(build_panel_group_model_statistics(
                merged, protocol=protocol, panel_index=panel_index,
                feature_col=feature_col, feature_label=feature_label, use_log10=use_log10))
    return pd.concat(frames, ignore_index=True)

# ============================================================
# 4. Plot Settings and Helpers
# ============================================================
def get_main_panel_ylim() -> tuple[float, float]:
    pad = Y_AXIS_PAD_FRACTION * (NORMALIZED_Y_TICKS[-1] - NORMALIZED_Y_TICKS[0])
    return float(NORMALIZED_Y_TICKS[0] - pad), float(NORMALIZED_Y_TICKS[-1] + pad)

def get_group_inset_positions() -> dict[str, list[float]]:
    y_low, y_high = get_main_panel_ylim()
    position = lambda y: (y - y_low) / (y_high - y_low)
    return {
        'prokaryotes': [GROUP_INSET_LEFT, position(float(NORMALIZED_Y_TICKS[3])) - GROUP_INSET_HEIGHT, GROUP_INSET_WIDTH, GROUP_INSET_HEIGHT],
        'eukaryotes': [GROUP_INSET_LEFT, position(float(NORMALIZED_Y_TICKS[1])), GROUP_INSET_WIDTH, GROUP_INSET_HEIGHT],
    }

def configure_matplotlib() -> None:
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['text.usetex'] = False
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['axes.unicode_minus'] = False

def fraction_tick_formatter(x, pos):
    if np.isclose(x, 0.0):
        return '0'
    if np.isclose(x, 1.0):
        return '1.0'
    return f'{x:.1f}'

def zero_or_two_decimal_tick_formatter(x, pos):
    if np.isclose(x, 0.0):
        return '0'
    return f'{x:.2f}'

def normalized_y_tick_formatter(x, pos):
    if np.isclose(x, 0.0):
        return '0'
    if np.isclose(x, 1.0):
        return '1.00'
    return f'{x:.2f}'

def style_axis(ax, *, x_fraction=False, y_fraction=False, grid=True) -> None:
    if grid:
        ax.grid(True, color='#D9D9D9', linewidth=0.8, alpha=0.9, zorder=0)
    ax.tick_params(axis='both', which='major', labelsize=TEXT_SIZE, length=5, width=0.9, direction='out')
    for spine in ax.spines.values():
        spine.set_color('#444444')
        spine.set_linewidth(0.9)
    if x_fraction:
        ax.set_xlim(-0.04, 1.04)
        ax.set_xticks(np.linspace(0.0, 1.0, 6))
        ax.xaxis.set_major_formatter(FuncFormatter(fraction_tick_formatter))
    if y_fraction:
        ax.set_ylim(-0.04, 1.04)
        ax.set_yticks(np.linspace(0.0, 1.0, 6))
        ax.yaxis.set_major_formatter(FuncFormatter(fraction_tick_formatter))
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontsize(TEXT_SIZE)

def set_five_y_ticks_with_reference_padding(ax, ticks, *, formatter=None, pad_fraction=Y_AXIS_PAD_FRACTION) -> None:
    ticks = np.asarray(ticks, dtype=float)
    if len(ticks) != 5:
        raise ValueError(f'Exactly five y ticks are required; received {len(ticks)}.')
    span = float(ticks[-1] - ticks[0])
    pad = span * pad_fraction if span > 0 else pad_fraction
    ax.set_ylim(float(ticks[0] - pad), float(ticks[-1] + pad))
    ax.set_yticks(ticks)
    if formatter is not None:
        ax.yaxis.set_major_formatter(FuncFormatter(formatter))
    for label in ax.get_yticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontsize(TEXT_SIZE)

def apply_metabolic_panel_geometry(ax) -> None:
    ax.set_box_aspect(1.0)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontname('Times New Roman')
        tick_label.set_fontsize(TEXT_SIZE)
    ax.xaxis.set_label_coords(0.5, -0.13)

def align_visible_panel_gaps(fig, rows) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    target_gap = REFERENCE_VISIBLE_PANEL_GAP_IN * fig.dpi

    def visible_gap(left_ax, right_ax) -> float:
        visible_ticks = [tick for tick in left_ax.xaxis.get_major_ticks() if left_ax.get_xlim()[0] <= tick.get_loc() <= left_ax.get_xlim()[1] and tick.tick1line.get_visible()]
        if not visible_ticks:
            raise RuntimeError('Cannot determine panel spacing: no visible x ticks.')
        last_tick = max(visible_ticks, key=lambda tick: tick.get_loc())
        tick_right = last_tick.tick1line.get_window_extent(renderer).x1
        ylabel_left = right_ax.yaxis.label.get_window_extent(renderer).x0
        return ylabel_left - tick_right

    def shift_axis_x(ax, pixels: float) -> None:
        position = ax.get_position()
        ax.set_position([position.x0 + pixels / fig.bbox.width, position.y0, position.width, position.height])
    for row in rows:
        original_left = row[0].get_position().x0
        original_right = row[-1].get_position().x1
        original_center = (original_left + original_right) / 2.0
        for left_ax, right_ax in zip(row, row[1:]):
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            shift_axis_x(right_ax, target_gap - visible_gap(left_ax, right_ax))
        new_left = row[0].get_position().x0
        new_right = row[-1].get_position().x1
        new_center = (new_left + new_right) / 2.0
        center_shift_pixels = (original_center - new_center) * fig.bbox.width
        for ax in row:
            shift_axis_x(ax, center_shift_pixels)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

def simplify_species_name(name: object) -> str:
    name_parts = str(name).strip().split()
    if len(name_parts) >= 2:
        return f'{name_parts[0][0]}. {name_parts[1]}'
    return name_parts[0] if name_parts else ''


# ============================================================
# 5. Plotting
# ============================================================
def plot_panel_1(ax, selected_species, curve_store, selected_color_map) -> None:
    species_handles = []
    for row in selected_species.itertuples(index=False):
        kegg_id = str(row.KEGG_ID)
        species_name = simplify_species_name(row.Species_name)
        color = selected_color_map[kegg_id]
        curve = curve_store[kegg_id].sort_values('p').copy()
        x_values = pd.to_numeric(curve['p'], errors='coerce').to_numpy(dtype=float)
        y_values = pd.to_numeric(curve['S2_over_N'], errors='coerce').to_numpy(dtype=float)
        valid = np.isfinite(x_values) & np.isfinite(y_values)
        x_values, y_values = (x_values[valid], y_values[valid])
        if not x_values.size:
            continue
        ax.fill_between(x_values, 0.0, y_values, facecolor=PANEL1_FILL_COLORS.get(color, color), alpha=0.28, linewidth=0.0, zorder=2)
        ax.plot(x_values, y_values, color=color, linewidth=2.4, zorder=5)
        species_handles.append(Line2D([0], [0], color=color, linewidth=3.0, label=species_name))
    ordered_handles = []
    for preferred_color in [ORANGE, BLUE, GREEN]:
        ordered_handles.extend((handle for handle in species_handles if handle.get_color() == preferred_color))
    if ordered_handles:
        legend = ax.legend(handles=ordered_handles, loc='upper right', frameon=False, fontsize=LEGEND_TEXT_SIZE, handlelength=1.6, labelspacing=0.45, borderpad=0.55)
        legend.set_zorder(20)
        for item in legend.get_texts():
            item.set_fontstyle('italic')
            item.set_fontname('Times New Roman')
    style_axis(ax, x_fraction=True, y_fraction=False)
    ax.set_xlim(-0.016, 0.416)
    ax.set_xticks(np.arange(0.0, 0.41, 0.1))
    ax.xaxis.set_major_formatter(FuncFormatter(fraction_tick_formatter))
    ax.set_ylim(-0.004, 0.104)
    ax.set_yticks([0.0, 0.05, 0.1])
    ax.yaxis.set_major_formatter(FuncFormatter(zero_or_two_decimal_tick_formatter))
    ax.set_xlabel('Fraction of nodes removed, $f$', fontsize=TEXT_SIZE)
    ax.set_ylabel('$S_2/N$', fontsize=TEXT_SIZE)

def add_group_inset(parent_ax, group_data: pd.DataFrame, *, plot_col: str, group_name: str, bounds, selected_color_map: dict[str, str], is_eukaryote: bool, panel_index: int, common_xlim: tuple[float, float]) -> None:
    inset_ax = parent_ax.inset_axes(bounds)
    inset_ax.set_facecolor('white')
    inset_ax.set_zorder(15)
    data = group_data
    selected_ids = set(selected_color_map)
    ordinary = data[~data['KEGG_ID'].isin(selected_ids)]
    highlighted = data[data['KEGG_ID'].isin(selected_ids)]
    if is_eukaryote:
        ordinary_edgecolor = EUKARYOTE_EDGE_COLOR
        ordinary_linewidth = EUKARYOTE_EDGE_WIDTH
        size_scale = EUKARYOTE_INSET_POINT_SIZE / SCATTER_SIZE
    else:
        ordinary_edgecolor = PROKARYOTE_EDGE_COLOR
        ordinary_linewidth = PROKARYOTE_EDGE_WIDTH
        size_scale = INSET_POINT_SIZE / SCATTER_SIZE
    if not ordinary.empty:
        inset_sizes = ordinary[MARKER_SIZE_COL].to_numpy(dtype=float) * size_scale
        inset_ax.scatter(ordinary[METRIC_COL], ordinary[plot_col], s=inset_sizes, color=SCATTER_GRAY, alpha=0.78, edgecolor=ordinary_edgecolor, linewidth=ordinary_linewidth, zorder=3)
    for _, row in highlighted.iterrows():
        kegg_id = str(row['KEGG_ID'])
        highlighted_inset_size = float(row[MARKER_SIZE_COL]) * size_scale * 1.35
        inset_ax.scatter(float(row[METRIC_COL]), float(row[plot_col]), s=highlighted_inset_size, color=selected_color_map[kegg_id], alpha=1.0, edgecolor='none', linewidth=0.0, zorder=7)
    stats = linear_regression_statistics(data, METRIC_COL, plot_col)
    if stats is not None:
        x_values = data[METRIC_COL].to_numpy(dtype=float)
        x_min = float(np.nanmin(x_values))
        x_max = float(np.nanmax(x_values))
        if np.isfinite(x_min) and np.isfinite(x_max) and (x_max > x_min):
            x_line = np.linspace(x_min, x_max, 100)
            y_line = stats['intercept'] + stats['slope'] * x_line
            inset_ax.plot(x_line, y_line, color='black', linewidth=INSET_LINEWIDTH, zorder=5)
        inset_r2_text = f"$R^2={stats['r_squared']:.3f}$"
        inset_p_text = format_p_value(stats['p_value'])
    else:
        inset_r2_text = '$R^2=\\mathrm{NA}$'
        inset_p_text = '$p=\\mathrm{NA}$'
    inset_stats_text = inset_r2_text + '\n' + inset_p_text
    inset_ax.text(0.04, 0.96, group_name, transform=inset_ax.transAxes, ha='left', va='top', fontsize=INSET_TEXT_SIZE, fontname='Times New Roman', clip_on=True, zorder=20)
    if panel_index == 2:
        if is_eukaryote:
            exemplar_edgecolor = EUKARYOTE_EDGE_COLOR
            exemplar_linewidth = EUKARYOTE_EDGE_WIDTH
        else:
            exemplar_edgecolor = PROKARYOTE_EDGE_COLOR
            exemplar_linewidth = PROKARYOTE_EDGE_WIDTH
        inset_ax.scatter([0.11], [0.6], s=SCATTER_SIZE, color=SCATTER_GRAY, alpha=0.68, edgecolor=exemplar_edgecolor, linewidth=exemplar_linewidth, transform=inset_ax.transAxes, clip_on=True, zorder=25)
    stats_x = 1.02
    stats_y_r2 = 1.0
    inset_ax.text(stats_x, stats_y_r2, inset_stats_text, transform=inset_ax.transAxes, ha='left', va='top', fontsize=INSET_STATS_TEXT_SIZE, fontname='Times New Roman', linespacing=1.2, clip_on=False, zorder=20)
    inset_ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    for spine in inset_ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(0.75)
    inset_ax.set_xlim(*common_xlim)
    if is_eukaryote and (not data.empty):
        y_values = pd.to_numeric(data[plot_col], errors='coerce').to_numpy(dtype=float)
        y_values = y_values[np.isfinite(y_values)]
        if y_values.size > 0:
            y_min = float(np.min(y_values))
            y_max = float(np.max(y_values))
            y_span = y_max - y_min
            if y_span > 0:
                y_pad = EUKARYOTE_Y_PAD_FRACTION * y_span
            else:
                parent_low, parent_high = get_main_panel_ylim()
                y_pad = 0.05 * (parent_high - parent_low)
            inset_ax.set_ylim(y_min - y_pad, y_max + y_pad)
        else:
            inset_ax.set_ylim(*get_main_panel_ylim())
    else:
        inset_ax.set_ylim(*get_main_panel_ylim())

def add_prokaryote_eukaryote_insets(ax, plot_data: pd.DataFrame, *, plot_col: str, panel_index: int, selected_color_map: dict[str, str], common_xlim: tuple[float, float]) -> None:
    positions = get_group_inset_positions()
    group_labels = plot_data['Cellular_group'].astype(str).str.strip().str.lower()
    prok_data = plot_data[group_labels == 'prokaryotes']
    euk_data = plot_data[group_labels == 'eukaryotes']
    add_group_inset(ax, prok_data, plot_col=plot_col, group_name='Prokaryotes', bounds=positions['prokaryotes'], selected_color_map=selected_color_map, is_eukaryote=False, panel_index=panel_index, common_xlim=common_xlim)
    add_group_inset(ax, euk_data, plot_col=plot_col, group_name='Eukaryotes', bounds=positions['eukaryotes'], selected_color_map=selected_color_map, is_eukaryote=True, panel_index=panel_index, common_xlim=common_xlim)

def plot_scatter_panel(ax, merged: pd.DataFrame, *, panel_index: int, feature_col: str, feature_label: str, use_log10: bool, selected_color_map: dict[str, str], common_xlim: tuple[float, float]):
    plot_data, plot_col = feature_panel_data(merged, feature_col, extra_columns=['Species_name', NETWORK_N_COL])
    plot_data[NETWORK_N_COL] = pd.to_numeric(plot_data[NETWORK_N_COL], errors='coerce')
    selected_ids = set(selected_color_map)
    ordinary = plot_data[~plot_data['KEGG_ID'].astype(str).isin(selected_ids)]
    highlighted = plot_data[plot_data['KEGG_ID'].astype(str).isin(selected_ids)]
    cellular_group = ordinary['Cellular_group'].astype(str).str.strip().str.lower()
    ordinary_prok = ordinary[cellular_group == 'prokaryotes']
    ordinary_euk = ordinary[cellular_group == 'eukaryotes']
    ordinary_other = ordinary[~cellular_group.isin(['prokaryotes', 'eukaryotes'])]

    def _sizes(frame: pd.DataFrame):
        return frame[MARKER_SIZE_COL].to_numpy(dtype=float)
    if not ordinary_prok.empty:
        ax.scatter(ordinary_prok[METRIC_COL], ordinary_prok[plot_col], s=_sizes(ordinary_prok), color=SCATTER_GRAY, alpha=0.68, edgecolor=PROKARYOTE_EDGE_COLOR, linewidth=PROKARYOTE_EDGE_WIDTH, zorder=3)
    if not ordinary_euk.empty:
        ax.scatter(ordinary_euk[METRIC_COL], ordinary_euk[plot_col], s=_sizes(ordinary_euk), color=SCATTER_GRAY, alpha=0.68, edgecolor=EUKARYOTE_EDGE_COLOR, linewidth=EUKARYOTE_EDGE_WIDTH, zorder=4)
    if not ordinary_other.empty:
        ax.scatter(ordinary_other[METRIC_COL], ordinary_other[plot_col], s=_sizes(ordinary_other), color=SCATTER_GRAY, alpha=0.68, edgecolor=PROKARYOTE_EDGE_COLOR, linewidth=PROKARYOTE_EDGE_WIDTH, zorder=3)
    for _, row in highlighted.iterrows():
        kegg_id = str(row['KEGG_ID'])
        highlighted_size = float(row[MARKER_SIZE_COL])
        ax.scatter(float(row[METRIC_COL]), float(row[plot_col]), s=highlighted_size, color=selected_color_map[kegg_id], alpha=1.0, edgecolor='none', linewidth=0.0, zorder=6)
    stats = linear_regression_statistics(plot_data, METRIC_COL, plot_col)
    if stats is not None:
        stats_text = f"$R^2={stats['r_squared']:.3f}$" + '\n' + format_p_value(stats['p_value'])
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, ha='left', va='top', fontsize=TEXT_SIZE, fontname='Times New Roman')
    add_prokaryote_eukaryote_insets(ax, plot_data, plot_col=plot_col, panel_index=panel_index, selected_color_map=selected_color_map, common_xlim=common_xlim)
    style_axis(ax, x_fraction=False, y_fraction=False)
    ax.set_xlabel(METRIC_LABEL, fontsize=TEXT_SIZE)
    y_title = f'Log10({feature_label})' if use_log10 else feature_label
    ax.set_ylabel(y_title, fontsize=TEXT_SIZE, fontname='Times New Roman', fontweight='normal', rotation=90)
    return stats


# ============================================================
# 6. Figure Generation
# ============================================================
def build_figure(merged: pd.DataFrame, curve_store: dict, output_dir: Path):
    complete_columns = ['KEGG_ID', 'Species_name', METRIC_COL, 'Protein_coding_genes', 'Unique_KEGG_Orthology_groups', 'Unique_Enzyme_Commission_numbers', NETWORK_N_COL, MARKER_SIZE_COL, 'KEGG_pathways', 'KEGG_modules']
    complete_pool = merged[complete_columns].copy().dropna().drop_duplicates(subset=['KEGG_ID'], keep='first').sort_values('KEGG_ID').reset_index(drop=True)
    positive_mask = (complete_pool['Protein_coding_genes'] > 0) & (complete_pool['Unique_KEGG_Orthology_groups'] > 0) & (complete_pool['Unique_Enzyme_Commission_numbers'] > 0) & (complete_pool['KEGG_pathways'] > 0) & (complete_pool['KEGG_modules'] > 0)
    complete_pool = complete_pool[positive_mask].reset_index(drop=True)
    if len(complete_pool) < 3:
        raise RuntimeError('Fewer than three species have complete data for SEAMLESS.')
    selected_species = complete_pool.sample(n=3, random_state=RANDOM_SEED, replace=False).copy()
    selected_species = selected_species.sort_values(METRIC_COL, ascending=False).reset_index(drop=True)
    selected_color_map = {}
    for _, row in selected_species.iterrows():
        species_name = str(row['Species_name']).strip()
        if not species_name:
            raise ValueError(f"Missing species name for KEGG_ID={row['KEGG_ID']}.")
        genus_initial = species_name[0].upper()
        if genus_initial not in REPRESENTATIVE_COLOR_BY_GENUS_INITIAL:
            raise ValueError(f"No representative color is configured for species {species_name!r} (KEGG_ID={row['KEGG_ID']}).")
        selected_color_map[str(row['KEGG_ID'])] = REPRESENTATIVE_COLOR_BY_GENUS_INITIAL[genus_initial]
    panel_species_sets = []
    panel_sample_counts = []
    for panel_index, (feature_col, feature_label, use_log10) in enumerate(FEATURES, start=2):
        panel_data, _ = feature_panel_data(merged, feature_col)
        panel_sample_counts.append((panel_index, feature_label, panel_data['KEGG_ID'].nunique()))
        panel_species_sets.append(set(panel_data['KEGG_ID'].astype(str)))
    print('\n========== Panel sample sizes ==========')
    print(f"Panel A selected species                : {selected_species['KEGG_ID'].nunique()}")
    for panel_index, feature_label, count in panel_sample_counts:
        print(f"Panel {chr(ord('A') + panel_index - 1)} - {feature_label:<30}: {count}")
    union_species = set(selected_species['KEGG_ID'].astype(str)).union(*panel_species_sets)
    print(f'Panels A-F union species               : {len(union_species)}')
    fig = plt.figure(figsize=FIGSIZE, facecolor='none')
    occupied_width_in = 3 * PANEL_WIDTH_IN + 2 * PANEL_GAP_IN
    side_margin_in = (FIGSIZE[0] - occupied_width_in) / 2
    grid = fig.add_gridspec(nrows=2, ncols=3, left=side_margin_in / FIGSIZE[0], right=1 - side_margin_in / FIGSIZE[0], bottom=BOTTOM_MARGIN_IN / FIGSIZE[1], top=1 - TOP_MARGIN_IN / FIGSIZE[1], wspace=PANEL_GAP_IN / PANEL_WIDTH_IN, hspace=ROW_GAP_IN / PANEL_WIDTH_IN)
    ax1 = fig.add_subplot(grid[0, 0])
    scatter_axes = [fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[0, 2]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1]), fig.add_subplot(grid[1, 2])]
    plot_panel_1(ax1, selected_species, curve_store, selected_color_map)
    for panel_index, (ax, (feature_col, feature_label, use_log10)) in enumerate(zip(scatter_axes, FEATURES), start=2):
        plot_scatter_panel(ax, merged, panel_index=panel_index, feature_col=feature_col, feature_label=feature_label, use_log10=use_log10, selected_color_map=selected_color_map, common_xlim=SCATTER_COMMON_XLIM)
    for ax in scatter_axes:
        ax.set_xlim(*SCATTER_COMMON_XLIM)
        ax.set_xticks(SEAMLESS_X_TICKS)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.3f}'))
        set_five_y_ticks_with_reference_padding(ax, NORMALIZED_Y_TICKS, formatter=normalized_y_tick_formatter)
    for ax in [ax1, *scatter_axes]:
        apply_metabolic_panel_geometry(ax)
    for ax, letter in zip([ax1, *scatter_axes], 'ABCDEF'):
        ax.text(0.98, 0.98, letter, transform=ax.transAxes, ha='right', va='top', fontsize=TEXT_SIZE, fontname='Times New Roman', fontweight='bold', color='black', zorder=30)
    align_visible_panel_gaps(fig, rows=[[ax1, scatter_axes[0], scatter_axes[1]], scatter_axes[2:5]])
    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / 'Metabolic_seamless.svg'
    fig.patch.set_alpha(0)
    for axis in fig.axes:
        axis.set_facecolor('none')
    fig.savefig(svg_path, format='svg', bbox_inches='tight', transparent=True)
    plt.show()
    plt.close(fig)


# ============================================================
# 7. Main
# ============================================================
def main():
    configure_matplotlib()
    feature_file = find_feature_file()
    features = prepare_features(feature_file)
    metrics, curve_store = load_results()
    merged = metrics.merge(features, on='KEGG_ID', how='inner')
    output_dir = BASE_DIR / 'seamless_figures'
    build_figure(merged.loc[merged['protocol'] == 'seamless'].copy(), curve_store, output_dir)
    all_panel_group_model_statistics = collect_attack_comparison_statistics(merged)
    print('\n' + '=' * 110)
    print('PANELS B-F LINEAR REGRESSION MODEL STATISTICS (AIC/BIC/ΔAIC/ΔBIC)')
    print('=' * 110)
    if all_panel_group_model_statistics.empty:
        print('No valid regression statistics were available.')
    else:
        stats = all_panel_group_model_statistics.copy()
        for column in ['n', 'R_squared', 'p_value', 'AIC', 'BIC']:
            stats[column] = pd.to_numeric(stats[column], errors='coerce')
        keys = ['Biological feature', 'Group']
        stats['Delta_AIC'] = stats['AIC'] - stats.groupby(keys)['AIC'].transform('min')
        stats['Delta_BIC'] = stats['BIC'] - stats.groupby(keys)['BIC'].transform('min')
        stats['Rank'] = stats.groupby(keys)['AIC'].rank(method='min', ascending=True, na_option='bottom')
        feature_order = [label for _, label, _ in FEATURES]
        feature_panel = {label: chr(ord('A') + index + 1) for index, label in enumerate(feature_order)}
        group_order = ['Overall', 'Prokaryotes', 'Eukaryotes']
        feature_order_map = {name: index for index, name in enumerate(feature_order)}
        group_order_map = {name: index for index, name in enumerate(group_order)}
        stats['_feature_order'] = stats['Biological feature'].map(feature_order_map).fillna(999)
        stats['_group_order'] = stats['Group'].map(group_order_map).fillna(999)
        stats = stats.sort_values(['_feature_order', '_group_order', 'AIC', 'BIC'], na_position='last')
        for feature in feature_order:
            feature_stats = stats[stats['Biological feature'] == feature]
            if feature_stats.empty:
                continue
            panel_letter = feature_panel[feature]
            print(f'\nBiological feature: {feature}  (Panel {panel_letter})')
            print('-' * 110)
            for group in group_order:
                group_stats = feature_stats[feature_stats['Group'] == group]
                if group_stats.empty:
                    continue
                print(f'\nGroup: {group}')
                display = group_stats[['Rank', 'Attack strategy', 'n', 'R_squared', 'p_value', 'AIC', 'BIC', 'Delta_AIC', 'Delta_BIC']].rename(columns={'Delta_AIC': 'ΔAIC', 'Delta_BIC': 'ΔBIC'}).copy()
                display['Rank'] = display['Rank'].map(lambda x: '' if pd.isna(x) else str(int(x)))
                display['n'] = display['n'].map(lambda x: '' if pd.isna(x) else str(int(x)))
                for col in ['R_squared', 'AIC', 'BIC', 'ΔAIC', 'ΔBIC']:
                    display[col] = display[col].map(lambda x: '' if pd.isna(x) else f'{x:.3f}')
                display['p_value'] = display['p_value'].map(lambda x: '' if pd.isna(x) else f'{x:.3e}')
                print(display.to_string(index=False))
    print('=' * 110)
if __name__ == '__main__':
    main()
