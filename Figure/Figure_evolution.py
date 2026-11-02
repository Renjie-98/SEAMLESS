
"""
Figure_evolution: Network evolution, adaptive fragility, and integrated efficiency.
Figure_evolution_curve: Network dismantling dynamics across generations.

Read precomputed evolutionary metrics and SEAMLESS dismantling curves.
Analyze 1 - AUC(S2/N), integrated efficiency, and Pearson correlations.
Plot radial trees, scatter plots, probability distributions, and S1/N and S2/N curves.

Read existing results only; never rerun SEAMLESS attacks.
Outputs: seamless_figures/evolution.svg
         seamless_figures/evolution_curve.svg
"""
from __future__ import annotations
import hashlib
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import MaxNLocator, FuncFormatter
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from IPython.display import display

# ============================================================
# 1. Project settings
# ============================================================
BASE = Path.cwd()
OUTPUT_DIR = BASE / 'natural_selection_PPI_9669'
SAVE_DIR = BASE / 'seamless_figures'
RATES = (0.2, 0.25, 0.3)
FONT = 'Times New Roman'
TREE_COLORS = {0.2: '#2072AA', 0.25: '#339A39', 0.3: '#D02623'}
TREE_CMAPS = {0.2: 'Blues', 0.25: 'Greens', 0.3: 'Reds'}
CMAP_RANGE = (0.46, 0.96)
GEN_GROUPS = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 50)]
GEN_GROUP_LABELS = [f'Gen {lo}–{hi}' for lo, hi in GEN_GROUPS]
EFFICIENCY_TICKS = np.array([3.5, 5.0, 6.5, 8.0, 9.5])
FRAGILITY_TICKS = np.array([0.92, 0.94, 0.96, 0.98, 1.0])
PROB_BINS = 20
PROB_SMOOTHING = 2.0
REF_TEXT_SIZE = 26
EVOLUTION_PANEL_W = 4.6315
EVOLUTION_PANEL_GAP = 1.2913
EVOLUTION_ROW_GAP = 1.2
EVOLUTION_BOTTOM = 2.28
EVOLUTION_TOP = 0.45
PER_GENERATION = 10
SAMPLE_SEED = 42
CURVE_LINE_W = 0.65
CURVE_LINE_A = 0.3
EVOLUTION_CURVE_W = 20.5
EVOLUTION_CURVE_PANEL_W = EVOLUTION_PANEL_W
EVOLUTION_CURVE_PANEL_G = EVOLUTION_PANEL_GAP
EVOLUTION_CURVE_ROW_GAP = EVOLUTION_ROW_GAP
EVOLUTION_CURVE_BOTTOM = 1.5
EVOLUTION_CURVE_TOP = 1.1
TREE_ANGLE_START = 90.0
TREE_INNER_RADIUS_FRACTION = 0.16
TREE_FAMILY_GAP_RATIO = 2.0

# ============================================================
# 2. Data loading and validation
# ============================================================
def load_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, low_memory=False)
    required = {'Generation', 'Network_ID', 'Parent_Network_ID', 'N', 'Integrated_Efficiency', 'Mean_AUC(S2/N)'}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f'{path} missing columns: {sorted(missing)}')
    for col in ('Generation', 'N', 'Integrated_Efficiency', 'Mean_AUC(S2/N)'):
        data[col] = pd.to_numeric(data[col], errors='coerce')
    bad_n = ~np.isfinite(data['N']) | (data['N'] <= 0) | (data['N'] % 1 != 0)
    if bad_n.any():
        examples = (data.index[bad_n][:5] + 2).tolist()
        raise ValueError(f'{path}: {int(bad_n.sum())} invalid network sizes N (missing, non-finite, non-positive, or non-integer); example CSV lines: {examples}')
    data['Network_ID'] = data['Network_ID'].astype(str).str.strip()
    data['Parent_Network_ID'] = data['Parent_Network_ID'].fillna('').astype(str).str.strip()
    bad_gen = data['Generation'].isna() | ~np.isfinite(data['Generation']) | (data['Generation'] < 0) | (data['Generation'] % 1 != 0)
    bad_id = data['Network_ID'].str.casefold().isin({'', 'nan', 'none'})
    if (bad_gen | bad_id).any():
        raise ValueError(f'{path}: invalid structural records detected.')
    data['Generation'] = data['Generation'].astype(int)
    data = data.sort_values(['Generation', 'Network_ID'], kind='stable').reset_index(drop=True)
    if data['Network_ID'].duplicated().any():
        raise ValueError(f'Duplicate Network_ID in {path}')
    roots = data.loc[data['Generation'] == 0, 'Network_ID']
    if len(roots) != 1:
        raise ValueError(f'Expected one gen-0 ancestor; found {len(roots)}')
    gen_by_id = dict(zip(data['Network_ID'], data['Generation'])); ids = set(data['Network_ID'])
    for row in data.loc[data['Generation'] > 0].itertuples(index=False):
        p = row.Parent_Network_ID
        if p not in ids:
            raise ValueError(f'Missing parent {p!r}')
        if gen_by_id[p] != row.Generation - 1:
            raise ValueError(f'Generation mismatch for {row.Network_ID!r}')
    return data

# ============================================================
# 3. Shared plotting functions
# ============================================================
def style_axes(ax, font=REF_TEXT_SIZE):
    """Shared grid, tick, font, and spine appearance."""
    ax.grid(True, color='#D9D9D9', linewidth=0.8, alpha=0.9); ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', labelsize=font, length=5, width=0.9, direction='out')
    for spine in ax.spines.values():
        spine.set(visible=True, color='#444444', linewidth=0.9)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set(fontname=FONT, fontsize=font, color='black')
def palette(rate, count=None):
    """Shared color range; return a truncated map or sampled group colors."""
    base = plt.get_cmap(TREE_CMAPS[rate]); lo, hi = CMAP_RANGE
    if count is not None:
        return [base(f) for f in np.linspace(lo, hi, count)]
    return LinearSegmentedColormap.from_list(f'{TREE_CMAPS[rate]}_trunc', base(np.linspace(lo, hi, 256)))
def result_csv(rate: float) -> Path:
    for p in (OUTPUT_DIR / f'{rate:.0%}_50gen' / 'evolution_metrics.csv', OUTPUT_DIR / f'{rate:.0%}' / 'evolution_metrics.csv'):
        if p.is_file():
            return p
    raise FileNotFoundError(f'No metrics CSV for r={rate:.0%}')
def display_limits(values, ticks, padding=0.04):
    """Include all data and requested ticks, retaining a small margin."""
    v = np.asarray(values, dtype=float); v = v[np.isfinite(v)]
    lo = min(float(v.min()), float(ticks[0])); hi = max(float(v.max()), float(ticks[-1]))
    margin = padding * (hi - lo if hi > lo else 1.0)
    return (lo - margin, hi + margin)
def style_generation_colorbar(colorbar, title, labelpad, tickpad=None):
    """Apply the shared Generation colorbar typography and tick styling."""
    colorbar.set_label(title, fontsize=22, labelpad=labelpad); colorbar.ax.xaxis.set_label_position('top')
    colorbar.locator = MaxNLocator(nbins=5, integer=True); colorbar.update_ticks()
    kwargs = dict(labelsize=22, direction='out', length=3, width=0.8)
    if tickpad is not None:
        kwargs['pad'] = tickpad
    colorbar.ax.tick_params(**kwargs)
    for label in colorbar.ax.get_xticklabels():
        label.set(fontsize=22, fontname=FONT)
    colorbar.ax.xaxis.label.set_fontsize(22); colorbar.outline.set_linewidth(0.6)
def add_panel(fig, x, y, w, h):
    """Create a transparent panel using the original figure coordinates."""
    ax = fig.add_axes([x, y, w, h]); ax.set_facecolor('none')
    return ax
def add_generation_bar(fig, bounds, rate, max_generation, title, labelpad, tickpad=None):
    cax = fig.add_axes(bounds)
    bar = fig.colorbar(ScalarMappable(norm=Normalize(0, max(max_generation, 1)), cmap=palette(rate)), cax=cax, orientation='horizontal')
    style_generation_colorbar(bar, title, labelpad, tickpad)
    return bar
def numeric_formatter(decimals=2):
    return FuncFormatter(lambda value, _: '0' if np.isclose(value, 0) else f'{value:.{decimals}f}')
def save_svg(fig, filename):
    path = SAVE_DIR / filename; fig.savefig(path, format='svg', bbox_inches='tight', transparent=True)
    print(f'Saved: {path}'); plt.show()
def reference_p_text(p):
    if p <= 0:
        return '$p<10^{-300}$'
    if p < 0.01:
        exp = int(np.floor(np.log10(p)))
        return f'$p={p / 10.0 ** exp:.1f}\\times10^{{{exp}}}$'
    return f'$p={p:.2f}$'

# ============================================================
# 4. Figure_evolution: Trees, scatter plots, and distributions
# ============================================================
def radial_tree_layout(data: pd.DataFrame):
    ids = data.Network_ID.astype(str).tolist(); generation = dict(zip(ids, data.Generation.astype(int)))
    root = next((n for n in ids if generation[n] == 0)); children, parent_of = (defaultdict(list), {})
    for row in data.loc[data.Generation > 0].itertuples(index=False):
        p, c = (str(row.Parent_Network_ID), str(row.Network_ID)); children[p].append(c)
        parent_of[c] = p
    for kids in children.values():
        kids.sort()
    traversal, stack, seen = ([], [root], set())
    while stack:
        node = stack.pop()
        if node in seen:
            raise ValueError('Cycle detected.')
        seen.add(node); traversal.append(node)
        stack.extend(reversed(children.get(node, [])))
    levels = defaultdict(list)
    for node in traversal:
        levels[generation[node]].append(node)
    local_max = max(generation.values()); polar, xy = ({}, {})
    for gen in sorted(levels):
        if gen == 0:
            nodes = [root]; angles = np.asarray([np.deg2rad(TREE_ANGLE_START)])
        else:
            families = sorted({parent_of[n] for n in levels[gen]}, key=lambda p: (polar[p][1] % (2 * np.pi), p))
            nodes = [c for p in families for c in children[p]]; count = len(nodes)
            gap_units = np.asarray([1.0 if parent_of[nodes[i]] == parent_of[nodes[(i + 1) % count]] else TREE_FAMILY_GAP_RATIO for i in range(count)])
            step = 2 * np.pi / gap_units.sum(); slots = np.r_[0.0, np.cumsum(gap_units[:-1])] * step
            targets = np.asarray([polar[parent_of[n]][1] for n in nodes])
            vector = np.exp(1j * (targets - slots)).sum()
            initial = float(np.angle(vector)) if abs(vector) > 1e-10 else 0.0
            cands = initial + np.linspace(-np.pi, np.pi, 181, endpoint=False)
            def cost(r):
                d = (slots + r - targets + np.pi) % (2 * np.pi) - np.pi
                return float(np.mean(d * d))
            rotation = min(cands, key=cost)
            for _ in range(8):
                d = (slots + rotation - targets + np.pi) % (2 * np.pi) - np.pi; corr = float(np.mean(d))
                prop = rotation - corr
                if cost(prop) > cost(rotation) + 1e-12:
                    break
                rotation = prop
                if abs(corr) < 1e-10:
                    break
            angles = slots + rotation
        radius = TREE_INNER_RADIUS_FRACTION + (1 - TREE_INNER_RADIUS_FRACTION) * gen / max(local_max, 1)
        for node, angle in zip(nodes, angles):
            polar[node] = (radius, float(angle)); xy[node] = (radius * np.cos(angle), radius * np.sin(angle))
    return (xy, polar, dict(children))
def tree_arc(radius, a0, a1):
    steps = max(2, int(np.ceil(abs(a1 - a0) / np.deg2rad(0.6))) + 1); angles = np.linspace(a0, a1, steps)
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))
def plot_tree(ax, data, rate, global_max_gen):
    xy, polar, children = radial_tree_layout(data); cmap = palette(rate)
    norm = Normalize(0, max(global_max_gen, 1))
    generation = dict(zip(data.Network_ID.astype(str), data.Generation.astype(int)))
    crowding = float(np.clip(np.log10(max(len(data), 400) / 400) / 2, 0, 1)); ow = 0.36 - 0.24 * crowding
    rw = 0.85 - 0.35 * crowding; oa = 0.52 - 0.38 * crowding
    ra = 0.95 - 0.12 * crowding; ordinary, repro, drawn_edges = ([], [], set())
    ocols, rcols = ([], [])
    for parent, kids in children.items():
        r0, a0 = polar[parent]; child_angles = np.asarray([polar[c][1] for c in kids])
        if len(kids) > 1:
            wrapped = np.sort(np.mod(child_angles, 2 * np.pi))
            gaps = np.diff(np.r_[wrapped, wrapped[0] + 2 * np.pi])
            start = wrapped[(int(np.argmax(gaps)) + 1) % len(wrapped)]
        else:
            start = float(child_angles[0])
        def near(a):
            return start + (a - start) % (2 * np.pi)
        fa = [near(a) for a in child_angles]; mid = (min(fa) + max(fa)) / 2
        pa = mid + (a0 - mid + np.pi) % (2 * np.pi) - np.pi; ang = fa + [pa]
        if max(ang) - min(ang) > 1e-12:
            ordinary.append(tree_arc(r0, min(ang), max(ang))); ocols.append(cmap(norm(generation[kids[0]])))
        ba = [near(polar[c][1]) for c in kids if c in children]
        if ba:
            ba.append(pa)
            if max(ba) - min(ba) > 1e-12:
                repro.append(tree_arc(r0, min(ba), max(ba))); rcols.append(cmap(norm(generation[kids[0]])))
        for child in kids:
            drawn_edges.add((parent, child)); _, a1 = polar[child]
            seg = [(r0 * np.cos(a1), r0 * np.sin(a1)), xy[child]]
            (repro if child in children else ordinary).append(seg)
            (rcols if child in children else ocols).append(cmap(norm(generation[child])))
    for segs, cols, w, alpha, z in ((ordinary, ocols, ow, oa, 1), (repro, rcols, rw, ra, 2)):
        if segs:
            ax.add_collection(LineCollection(segs, colors=cols, linewidths=w, alpha=alpha, capstyle='round', joinstyle='round', zorder=z, rasterized=True))
    ids = data.Network_ID.astype(str).to_numpy(); generations = np.asarray([generation[n] for n in ids])
    points = np.asarray([xy[n] for n in ids]); parents = np.asarray([n in children for n in ids])
    ss = (1 - crowding) ** 2
    for mask, size, opacity, z in ((~parents, 0.025 + 1.8 * ss, 0.65 - 0.3 * crowding, 3), (parents, 0.3 + 10.0 * ss, 0.98, 4)):
        if mask.any():
            ax.scatter(points[mask, 0], points[mask, 1], s=size, c=generations[mask], cmap=cmap, norm=norm, marker='o', edgecolors='none', linewidths=0, alpha=opacity, zorder=z, rasterized=True)
    expected = {(str(r.Parent_Network_ID), str(r.Network_ID)) for r in data.loc[data.Generation > 0].itertuples(index=False)}
    if set(xy) != set(ids) or drawn_edges != expected:
        raise RuntimeError('Node/edge audit failed.')
    ax.set_xlim(-1.0, 1.0); ax.set_ylim(-1.0, 1.0)
    for art in list(ax.collections) + list(ax.lines):
        art.set_clip_on(False)
    ax.set_aspect('equal', adjustable='box'); ax.set_axis_off()
    ax.text(0, 0, f'{rate:.0%}', ha='center', va='center', fontsize=18, fontweight='bold', color=TREE_COLORS[rate])
def smooth_bin_probabilities(values, edges):
    """20-bin empirical probabilities, Gaussian-smoothed and sum-normalized."""
    values = np.asarray(values, dtype=float); values = values[np.isfinite(values)]
    if not len(values):
        return None
    counts, _ = np.histogram(values, bins=edges); probs = counts.astype(float) / len(values)
    smooth = gaussian_filter1d(probs, sigma=PROB_SMOOTHING, mode='constant'); total = smooth.sum()
    if total <= 0:
        return None
    smooth /= total
    return ((edges[:-1] + edges[1:]) / 2, smooth)
def style_probability_ax(ax, metric_label, metric_ticks, metric_lim, metric_formatter, show_legend, colors):
    """X: network metric; Y: smoothed 20-bin probability."""
    ax.set_xlim(*metric_lim); ax.set_xticks(metric_ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(metric_formatter)); ax.set_ylim(-0.03, 0.33)
    ax.set_yticks([0.0, 0.15, 0.3]); ax.yaxis.set_major_formatter(numeric_formatter())
    ax.text(0.045, 0.94, metric_label, transform=ax.transAxes, ha='left', va='top', fontsize=REF_TEXT_SIZE, fontname=FONT, color='black', zorder=12)
    style_axes(ax)
    if show_legend:
        handles = [plt.Line2D([0], [0], color=color, linewidth=2, linestyle='-', label=label) for color, label in zip(colors, GEN_GROUP_LABELS)]
        ax.legend(handles=handles, frameon=False, fontsize=22, loc='upper right', handlelength=1.5, bbox_to_anchor=(0.98, 1.08))
def plot_probability_pair(ax_top, ax_bot, data, rate, efficiency_edges, fragility_edges, efficiency_xlim, fragility_xlim):
    colors = palette(rate, len(GEN_GROUPS))
    for (g_lo, g_hi), color in zip(GEN_GROUPS, colors):
        sub = data.loc[data['Generation'].between(g_lo, g_hi)]
        distributions = ((ax_top, sub['Integrated_Efficiency'].to_numpy(float), efficiency_edges), (ax_bot, 1 - sub['Mean_AUC(S2/N)'].to_numpy(float), fragility_edges))
        for ax, values, edges in distributions:
            result = smooth_bin_probabilities(values, edges)
            if result is None:
                continue
            centers, probabilities = result
            if probabilities.max() > 0.33:
                metric_name = 'Integrated Efficiency' if ax is ax_top else '1-AUC(S2/N)'
                print(f'Warning: r={rate:.0%}, Gen {g_lo}-{g_hi}, {metric_name}: probability peak {probabilities.max():.4f} exceeds fixed y-axis maximum 0.33 (curve may be clipped).')
            ax.plot(centers, probabilities, color=color, linewidth=2.0, alpha=0.9, linestyle='-', zorder=6)
    style_probability_ax(ax_top, '$\\mathcal{I}$', EFFICIENCY_TICKS, efficiency_xlim, lambda v, pos: f'{v:.1f}', True, colors)
    style_probability_ax(ax_bot, '$1-\\mathrm{AUC}(S_2/N)$', FRAGILITY_TICKS, fragility_xlim, lambda v, pos: f'{v:.2f}', False, colors)
def plot_reference_relationship(ax, data, rate, n_min, n_max, xlim, ylim, max_generation):
    metric = 'Mean_AUC(S2/N)'; valid = np.isfinite(data[metric]) & np.isfinite(data['Integrated_Efficiency'])
    plotted = data.loc[valid]
    if plotted.empty:
        raise ValueError(f'No data for r={rate:.0%}')
    x = 1.0 - plotted[metric].to_numpy(float); y = plotted['Integrated_Efficiency'].to_numpy(float)
    nn = plotted['N'].to_numpy(float); good = np.isfinite(nn) & (nn > 0)
    nn = np.where(good, nn, float(np.median(nn[good])) if good.any() else n_min)
    sizes = 90.0 + 170.0 * np.clip((nn - n_min) / (n_max - n_min), 0, 1) if n_max > n_min else np.full(len(nn), 175.0)
    ax.scatter(x, y, s=sizes, c=plotted['Generation'].to_numpy(float), cmap=palette(rate), norm=Normalize(0, max(max_generation, 1)), alpha=0.68, edgecolor='white', linewidth=0.35, zorder=3, rasterized=True)
    if len(x) >= 3 and np.ptp(x) > 0 and (np.ptp(y) > 0):
        fit = stats.linregress(x, y)
        label = f'$R^2={fit.rvalue ** 2:.3f}$' + '\n' + reference_p_text(fit.pvalue)
        ax.text(0.05, 0.95, label, transform=ax.transAxes, ha='left', va='top', fontsize=REF_TEXT_SIZE - 3, fontname=FONT, zorder=8)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xticks(FRAGILITY_TICKS); ax.set_yticks(EFFICIENCY_TICKS)
    ax.xaxis.set_major_formatter(numeric_formatter())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f'{v:.1f}')); style_axes(ax)
    ax.set(xlabel='$1-\\mathrm{AUC}(S_2/N)$', ylabel='$\\mathcal{I}$')
    ax.xaxis.label.set_fontsize(REF_TEXT_SIZE); ax.yaxis.label.set_fontsize(REF_TEXT_SIZE)
    ax.xaxis.set_label_coords(0.5, -0.13); ax.set_box_aspect(1)
def build_figure_evolution(datasets, max_generation, n_min, n_max, efficiency_xlim, fragility_xlim, efficiency_edges, fragility_edges):
    width = 20.5; panel = EVOLUTION_PANEL_W
    kde_sub_h = panel * 0.48; kde_gap = panel * 0.18
    kde_total = 2 * kde_sub_h + kde_gap
    height = EVOLUTION_TOP + kde_total + panel + panel + 2 * EVOLUTION_ROW_GAP + EVOLUTION_BOTTOM
    fig = plt.figure(figsize=(width, height), facecolor='none'); occupied = 3 * panel + 2 * EVOLUTION_PANEL_GAP
    left = (width - occupied) / 2
    for column, (rate, data) in enumerate(datasets):
        col_x = (left + column * (panel + EVOLUTION_PANEL_GAP)) / width; pw = panel / width
        ph = panel / height; kde_bot_y = EVOLUTION_BOTTOM / height
        kde_top_y = kde_bot_y + (kde_sub_h + kde_gap) / height
        scat_bot_y = (EVOLUTION_BOTTOM + kde_total + EVOLUTION_ROW_GAP) / height
        tree_bot_y = (EVOLUTION_BOTTOM + kde_total + panel + 2 * EVOLUTION_ROW_GAP) / height
        kde_sh = kde_sub_h / height
        ax_tree, ax_scat, ax_kde_i, ax_kde_x = [add_panel(fig, col_x, y, pw, h) for y, h in ((tree_bot_y, ph), (scat_bot_y, ph), (kde_top_y, kde_sh), (kde_bot_y, kde_sh))]
        plot_tree(ax_tree, data, rate, max_generation)
        plot_reference_relationship(ax_scat, data, rate, n_min, n_max, fragility_xlim, efficiency_xlim, max_generation)
        if column > 0:
            ax_scat.set_ylabel(''); ax_scat.tick_params(axis='y', labelleft=False, left=False)
        plot_probability_pair(ax_kde_i, ax_kde_x, data, rate, efficiency_edges, fragility_edges, efficiency_xlim, fragility_xlim)
        if column != 0:
            for ax in (ax_kde_i, ax_kde_x):
                ax.tick_params(axis='y', labelleft=False, left=False)
        lower_top = ax_scat.get_position().y1; upper_bottom = ax_tree.get_position().y0
        bar_center = (lower_top + upper_bottom) / 2
        add_generation_bar(fig, [col_x, bar_center - 0.08 / height, pw, 0.13 / height], rate, max_generation, f'Generation (n={len(data):,})', 6, 2)
        for row_idx, ax in enumerate((ax_tree, ax_scat, ax_kde_i, ax_kde_x)):
            letter = chr(65 + row_idx * 3 + column) if row_idx < 3 else chr(65 + 9 + column)
            ax.text(0.98, 0.98, letter, transform=ax.transAxes, ha='right', va='top', fontsize=REF_TEXT_SIZE, fontname=FONT, fontweight='bold', zorder=30)
    centre_y = (kde_bot_y + kde_top_y + kde_sh) / 2
    fig.text(left / width - 0.047, centre_y, 'Probability', rotation=90, ha='center', va='center', fontsize=REF_TEXT_SIZE, fontname=FONT)
    return fig

# ============================================================
# 5. Figure_evolution_curve: Dismantling curves
# ============================================================
def sampling_score(identity):
    return hashlib.sha256(f'{SAMPLE_SEED}|curve-display|{identity}'.encode()).digest()
def sample_networks(data):
    selected = []
    for generation, group in data.groupby('Generation', sort=True):
        ranked = sorted(group.Network_ID, key=lambda x: (sampling_score(x), x))
        chosen = ranked if generation == 0 else ranked[:PER_GENERATION]
        selected.extend(((i, int(generation)) for i in chosen))
    return selected
def load_selected_curves(selected, curve_dir):
    curves = []
    for identity, generation in selected:
        path = curve_dir / f'{identity}.npz'
        try:
            with np.load(path, allow_pickle=False) as saved:
                x = np.asarray(saved['removed_fraction'], dtype=float)
                s1 = np.asarray(saved['mean_s1_over_n'], dtype=float)
                s2 = np.asarray(saved['mean_s2_over_n'], dtype=float)
        except (OSError, ValueError, KeyError, EOFError) as exc:
            raise ValueError(f'Cannot load {path}: {exc}') from exc
        if x.ndim != 1 or len(x) < 2 or s1.shape != x.shape or (s2.shape != x.shape) or (not np.isfinite(np.r_[x, s1, s2]).all()) or np.any(np.diff(x) <= 0) or (not np.isclose(x[0], 0)) or (not np.isclose(x[-1], 1)) or (x.min() < -1e-09) or (x.max() > 1 + 1e-09) or (min(s1.min(), s2.min()) < -1e-09) or (max(s1.max(), s2.max()) > 1 + 1e-09):
            raise ValueError(f'Invalid normalized curve in {path}')
        curves.append((identity, generation, x, s1, s2))
    return curves
def build_figure_evolution_curve(loaded, max_generation):
    width = EVOLUTION_CURVE_W
    height = EVOLUTION_CURVE_TOP + 2 * EVOLUTION_CURVE_PANEL_W + EVOLUTION_CURVE_ROW_GAP + EVOLUTION_CURVE_BOTTOM
    fig = plt.figure(figsize=(width, height), facecolor='none')
    occupied = 3 * EVOLUTION_CURVE_PANEL_W + 2 * EVOLUTION_CURVE_PANEL_G; left = (width - occupied) / 2
    norm = Normalize(0, max(max_generation, 1)); s2_ticks = np.array([0.0, 0.06, 0.12, 0.18, 0.24])
    y_ticks = (np.linspace(0, 1, 5), s2_ticks); limits = ((-0.04, 1.04), (-0.0096, 0.2496))
    for column, (rate, curves) in enumerate(loaded):
        cmap = palette(rate); descendants = [c for c in curves if c[1] > 0]
        ancestors = [c for c in curves if c[1] == 0]
        if len(ancestors) != 1:
            raise ValueError(f'Expected one ancestor curve for retention {rate:.0%}; found {len(ancestors)}')
        ancestor = ancestors[0]; colors = [cmap(norm(c[1])) for c in descendants]
        xpos = (left + column * (EVOLUTION_CURVE_PANEL_W + EVOLUTION_CURVE_PANEL_G)) / width
        for row in range(2):
            ypos = (EVOLUTION_CURVE_BOTTOM + (1 - row) * (EVOLUTION_CURVE_PANEL_W + EVOLUTION_CURVE_ROW_GAP)) / height
            ax = add_panel(fig, xpos, ypos, EVOLUTION_CURVE_PANEL_W / width, EVOLUTION_CURVE_PANEL_W / height)
            if descendants:
                segs = [np.column_stack((c[2], c[3 + row])) for c in descendants]
                ax.add_collection(LineCollection(segs, colors=colors, linewidths=CURVE_LINE_W, alpha=CURVE_LINE_A, zorder=2))
            ax.plot(ancestor[2], ancestor[3 + row], color='#333333', linestyle='--', linewidth=1.65, label='Ancestor', zorder=4)
            ax.set_xlim(-0.04, 1.04); ax.set_ylim(*limits[row])
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_yticks(y_ticks[row])
            ax.xaxis.set_major_formatter(numeric_formatter()); ax.yaxis.set_major_formatter(numeric_formatter())
            style_axes(ax); ax.set_box_aspect(1)
            if column == 0:
                ax.set_ylabel('$S_1/N$' if row == 0 else '$S_2/N$', fontsize=REF_TEXT_SIZE)
            else:
                ax.tick_params(axis='y', left=False, labelleft=False)
            if row == 1:
                ax.set_xlabel('Fraction of nodes removed, $f$', fontsize=REF_TEXT_SIZE)
                ax.xaxis.set_label_coords(0.5, -0.13)
            ax.text(0.98, 0.98, chr(65 + row * 3 + column), transform=ax.transAxes, ha='right', va='top', fontsize=REF_TEXT_SIZE, fontweight='bold')
            if row == 1 and column == 2:
                ax.legend(loc='upper right', frameon=False, fontsize=REF_TEXT_SIZE)
        top = (EVOLUTION_CURVE_BOTTOM + 2 * EVOLUTION_CURVE_PANEL_W + EVOLUTION_CURVE_ROW_GAP) / height
        add_generation_bar(fig, [xpos, top + 0.35 / height, EVOLUTION_CURVE_PANEL_W / width, 0.13 / height], rate, max_generation, 'Generation', 5)
    return fig
def show_grouped_correlations(datasets):
    """Notebook table: Pearson R² and p by retention and network size N."""
    records = []
    for rate, data in datasets:
        for network_size, group in data.groupby('N', sort=True, dropna=True):
            x = 1.0 - group['Mean_AUC(S2/N)'].to_numpy(float)
            y = group['Integrated_Efficiency'].to_numpy(float); valid = np.isfinite(x) & np.isfinite(y)
            x, y = (x[valid], y[valid]); n = len(x)
            r_squared, p_value, status = (np.nan, np.nan, 'OK')
            if n < 3:
                status = 'Insufficient data'
            elif np.ptp(x) == 0 or np.ptp(y) == 0:
                status = 'Constant values'
            else:
                fit = stats.linregress(x, y); r_squared, p_value = (fit.rvalue ** 2, fit.pvalue)
                if not (np.isfinite(r_squared) and np.isfinite(p_value)):
                    status = 'Invalid result'
            records.append({'Retention': f'{rate:.0%}', 'N': int(network_size), 'n': n, 'R²': f'{r_squared:.4f}' if np.isfinite(r_squared) else '—', 'p': ('<1e-300' if p_value < 1e-300 else f'{p_value:.3e}') if np.isfinite(p_value) else '—', 'Status': status})
    results_df = pd.DataFrame(records, columns=['Retention', 'N', 'n', 'R²', 'p', 'Status'])
    print('1-AUC(S2/N) vs integrated efficiency'); display(results_df)
    return results_df

# ============================================================
# 6. Main
# ============================================================
def main():
    plt.rcParams.update({'font.family': 'serif', 'font.serif': [FONT, 'DejaVu Serif'], 'mathtext.fontset': 'stix', 'axes.unicode_minus': False, 'svg.fonttype': 'path', 'pdf.fonttype': 42, 'ps.fonttype': 42})
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    datasets = [(rate, load_data(result_csv(rate))) for rate in RATES]
    efficiency = np.concatenate([d['Integrated_Efficiency'].to_numpy(float) for _, d in datasets])
    fragility = np.concatenate([1.0 - d['Mean_AUC(S2/N)'].to_numpy(float) for _, d in datasets])
    efficiency = efficiency[np.isfinite(efficiency)]; fragility = fragility[np.isfinite(fragility)]
    if not len(efficiency) or not len(fragility):
        raise ValueError('No finite efficiency or fragility observations available')
    efficiency_xlim = display_limits(efficiency, EFFICIENCY_TICKS)
    fragility_xlim = display_limits(fragility, FRAGILITY_TICKS)
    max_generation = max((int(d.Generation.max()) for _, d in datasets))
    all_n = np.concatenate([d['N'].to_numpy(float) for _, d in datasets])
    all_n = all_n[np.isfinite(all_n) & (all_n > 0)]
    if not len(all_n):
        raise ValueError('No valid network sizes found.')
    n_min, n_max = (float(all_n.min()), float(all_n.max()))
    def shared_edges(values):
        low, high = (float(np.min(values)), float(np.max(values)))
        if np.isclose(low, high):
            pad = max(abs(low) * 0.01, 1e-06)
        else:
            pad = (high - low) * 1e-09
        return np.linspace(low - pad, high + pad, PROB_BINS + 1)
    efficiency_edges = shared_edges(efficiency); fragility_edges = shared_edges(fragility)
    show_grouped_correlations(datasets)
    fig_evolution = build_figure_evolution(datasets, max_generation, n_min, n_max, efficiency_xlim, fragility_xlim, efficiency_edges, fragility_edges)
    save_svg(fig_evolution, 'evolution.svg'); loaded_evolution_curve = []
    for rate, data in datasets:
        selected = sample_networks(data)
        print(f'r={rate:.0%}: loading {len(selected):,}/{len(data):,} curve networks...')
        curves = load_selected_curves(selected, result_csv(rate).parent / 'mean_curves')
        loaded_evolution_curve.append((rate, curves))
    fig_evolution_curve = build_figure_evolution_curve(loaded_evolution_curve, max_generation)
    save_svg(fig_evolution_curve, 'evolution_curve.svg')
    return (fig_evolution, fig_evolution_curve)
if __name__ == '__main__':
    main()

