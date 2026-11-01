
"""
Figure_bio_curve: Component-size curves of biological networks under adaptive dismantling.

Read precomputed SEAMLESS attack curves (m = 4) for PPI and metabolic networks.
Plot individual S1/N and S2/N curves and their cross-network means.

Read existing results only; never rerun SEAMLESS attacks.
Output: seamless_figures/PPI_Metabolic_component_curves.svg (Panels A-B).
"""

# ============================================================
# 1. Configuration
# ============================================================
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

BASE = Path.cwd()
F = np.linspace(0, 1, 201)
FIGSIZE, PANEL_W, PANEL_GAP, VISIBLE_GAP_IN = (12, 7.6), 4.6315, 1.2913, 0.55
FONT, RED = 26, '#d02623'
S1_DASH = (0, (3.7, 1.6))
ALIASES = {
    'p': ['p', 'fraction removed', 'fraction_removed'],
    's1': ['mean_S1_over_N', 'S1_over_N', 'mean_S1/N', 'S1/N', 'meanS1overN'],
    's2': ['mean_S2_over_N', 'S2_over_N', 'mean_S2/N', 'S2/N', 'meanS2overN'],
}

# ============================================================
# 2. Data loading
# ============================================================
def load_networks(folder, kind, selected=None):
    """Read each network's summary.csv; average within network, not across networks."""
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    curves = {'s1': [], 's2': []}
    count = 0
    prefix = f'results_{kind}_'
    norm = lambda x: re.sub('[^a-z0-9]', '', str(x).lower())
    for sub in sorted(folder.iterdir()):
        if not sub.is_dir() or not sub.name.lower().startswith(prefix.lower()):
            continue
        key = re.sub(r'\.0$', '', sub.name[len(prefix):].strip())
        if selected is not None and key not in selected:
            continue
        files = sorted(sub.rglob('summary.csv'))
        if not files:
            continue
        try:
            df = pd.read_csv(files[0])
            cols = {norm(c): c for c in df.columns}
            find = lambda names: next((cols[norm(n)] for n in names if norm(n) in cols), None)
            p, protocol, m = find(ALIASES['p']), find(['protocol']), find(['m'])
            comp = {k: find(ALIASES[k]) for k in ('s1', 's2')}
            comp = {k: v for k, v in comp.items() if v is not None}
            if p is None or protocol is None or not comp:
                continue
            df = df[df[protocol].astype(str).str.strip().str.lower().eq('seamless')].copy()
            if m is not None:
                df = df[np.isclose(pd.to_numeric(df[m], errors='coerce'), 4)].copy()
            used = [p, *comp.values()]
            df[used] = df[used].apply(pd.to_numeric, errors='coerce')
            grouped = df.dropna(subset=used).groupby(p)[list(comp.values())].mean().sort_index()
            if len(grouped) < 2:
                continue
            x = grouped.index.to_numpy(dtype=float)
            if x.max() > 1.5:
                x = x / 100
            order = np.argsort(x)
            x = x[order]
            if np.any(np.diff(x) <= 0):
                continue
            inside = (F >= x[0]) & (F <= x[-1])
            for k, col in comp.items():
                y = np.full(F.shape, np.nan)
                y[inside] = np.interp(F[inside], x, grouped[col].to_numpy()[order])
                curves[k].append(y)
            count += 1
        except Exception:
            continue  # Preserve original skip-on-error behavior.
    return curves, count

# ============================================================
# 3. Plotting
# ============================================================
def plot_panel(ax, title, n, curves):
    for k, ls, label in [('s1', S1_DASH, r'$S_1/N$'), ('s2', '-', r'$S_2/N$')]:
        if not curves[k]:
            continue
        for curve in curves[k]:
            ax.plot(F, curve, color=RED, ls=ls, lw=0.8, alpha=0.15, zorder=2)
        mean = np.nanmean(np.asarray(curves[k], dtype=float), axis=0)
        ax.plot(F, mean, color=RED, ls=ls, lw=2.2, label=label, zorder=3)
    ax.set_title(f'{title} (n={n})', fontsize=FONT, pad=12)
    ax.set(xlim=(-0.04, 1.04), ylim=(-0.04, 1.04))
    ax.set_xlabel(r'Fraction of nodes removed, $f$', fontsize=FONT, labelpad=8)
    ax.set_ylabel('Component size', fontsize=FONT, labelpad=8)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    formatter = FuncFormatter(lambda x, _: '0' if np.isclose(x, 0) else f'{x:.1f}')
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.tick_params(labelsize=FONT, length=5, width=0.9, direction='out')
    ax.grid(True, color='#D9D9D9', linewidth=0.8, alpha=0.9)
    for spine in ax.spines.values():
        spine.set(color='#444444', linewidth=0.9)

# ============================================================
# 4. Main
# ============================================================
def main():
    ids_path = BASE / 'benchmarks/PPI_SI/species_ID.txt'
    ids = {re.sub(r'\.0$', '', s.strip()) for s in ids_path.read_text(encoding='utf-8-sig').splitlines() if s.strip()}
    ppi, n_ppi = load_networks(BASE / 'results_PPI', 'PPI', ids)
    met, n_met = load_networks(BASE / 'results_metabolic', 'Metabolic')
    if not any(ppi.values()) or not any(met.values()):
        raise ValueError('No valid PPI or metabolic SEAMLESS m=4 curves were found.')
    plt.rcParams.update({'font.family': 'Times New Roman', 'mathtext.fontset': 'stix'})
    fig = plt.figure(figsize=FIGSIZE, facecolor='white')
    left = (FIGSIZE[0] - (2 * PANEL_W + PANEL_GAP)) / 2
    bottom = (FIGSIZE[1] - PANEL_W) / 2
    axes = [fig.add_axes([(left + i * (PANEL_W + PANEL_GAP)) / FIGSIZE[0],
                          bottom / FIGSIZE[1], PANEL_W / FIGSIZE[0], PANEL_W / FIGSIZE[1]]) for i in range(2)]
    for ax, title, n, curves, label in zip(axes, ['PPI', 'Metabolic'], [n_ppi, n_met], [ppi, met], 'AB'):
        plot_panel(ax, title, n, curves)
        ax.text(0.98, 0.98, label, transform=ax.transAxes, ha='right', va='top', fontsize=FONT, fontweight='bold')
    axes[0].legend(frameon=False, fontsize=FONT - 4, loc='lower right',
                   bbox_to_anchor=(0.98, 0.08), borderaxespad=0,
                   handlelength=1.8, handletextpad=0.45, labelspacing=0.35)
    # Preserve original visible inter-panel gap (0.55 in) and overall centering.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ticks = [t for t in axes[0].xaxis.get_major_ticks() if t.tick1line.get_visible()]
    if ticks:
        tick_x = max(ticks, key=lambda t: t.get_loc()).tick1line.get_window_extent(renderer).x1
        label_x = axes[1].yaxis.label.get_window_extent(renderer).x0
        shift = (VISIBLE_GAP_IN * fig.dpi - (label_x - tick_x)) / fig.bbox.width
        center = (axes[0].get_position().x0 + axes[1].get_position().x1) / 2
        pos = axes[1].get_position()
        axes[1].set_position([pos.x0 + shift, pos.y0, pos.width, pos.height])
        correction = center - (axes[0].get_position().x0 + axes[1].get_position().x1) / 2
        for ax in axes:
            pos = ax.get_position()
            ax.set_position([pos.x0 + correction, pos.y0, pos.width, pos.height])
    out = BASE / 'seamless_figures/bio_curve.svg'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format='svg', bbox_inches='tight', transparent=True)
    plt.show()
    plt.close(fig)
    print(f'SVG saved to: {out}')

if __name__ == '__main__':
    main()
