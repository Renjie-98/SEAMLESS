"""
SEAMLESS Figure 1: toy-model illustration and existing PPI results.

Data
----
A-D: fixed 16-node schematic; no dismantling simulation.
E: snapshots from one reproducible SEAMLESS attack (seed from config.json).
F: saved S1/N and S2/N curves for PPI network 880447 (m=4).

Input
-----
Place this script in results_new/.
    benchmarks/PPI/880447.edges
    results_PPI/results_PPI_880447/config.json
    results_PPI/results_PPI_880447/summary.csv
Panel F shows saved means over 100 attacks; Panel E is one example attack.

Output
------
    seamless_figures/seamless.svg
"""

# ============================================================
# 1. Imports and Configuration
# ============================================================
from pathlib import Path
import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb

ROOT = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
RESULTS = ROOT / 'results_PPI' / 'results_PPI_880447'
EDGES = ROOT / 'benchmarks' / 'PPI' / '880447.edges'
M, FRACTIONS = 4, (0, 0.05, 0.10)
RED, GREEN, BLUE, GRAY = '#D02623', '#339a39', '#2072aa', '#D9DEE7'
CMAP = LinearSegmentedColormap.from_list('candidate', ['#FBE8E7', RED])
plt.rcParams.update({'font.family': 'Times New Roman', 'mathtext.fontset': 'stix',
                     'svg.fonttype': 'path', 'axes.unicode_minus': False})

# ============================================================
# 2. Toy Model (Panels A-D)
# ============================================================
def score(g, u):
    ns = set(g[u])
    terms = []
    for v in ns:
        a, b = ns - {v}, set(g[v]) - {u}
        terms.append(1 - len(a & b) / len(a | b) if a | b else 0)
    return np.mean(terms) * np.sqrt(len(ns)) if terms else 0


def toy_network():
    g = nx.Graph()
    g.add_nodes_from(range(1, 17))
    g.add_edges_from([(1,2),(1,5),(2,3),(2,6),(3,7),(4,8),(5,6),(5,9),
                      (6,7),(6,10),(6,11),(7,8),(9,10),(10,11),(10,14),
                      (11,12),(12,16),(13,14),(15,16)])
    pos = nx.spring_layout(g, seed=23, k=.65, iterations=500, scale=3)
    mapping = {4:'1',8:'2',3:'3',7:'4',15:'5',1:'6',2:'7',5:'8',6:'9',
               11:'10',12:'11',16:'12',9:'13',10:'14',14:'15',13:'16'}
    inv = {v:k for k,v in mapping.items()}
    for i, name in enumerate(('11','12','5'), 1):
        pos[inv[name]] = np.asarray(pos[inv['10']]) + i*np.array([.45,.72])
    # Branch-aware spatial numbering: finish the left branch before the right.
    # Only displayed labels change; node identities, positions, edges, and seeds stay fixed.
    # Left branch (top to bottom): 1–9; right branch (top to bottom): 10–16.
    label_order = (4, 8, 3, 7, 2, 1, 5, 6, 9,
                   15, 16, 12, 11, 10, 14, 13)
    labels = {node: str(i) for i, node in enumerate(label_order, 1)}
    return g, pos, labels, {inv['7'], inv['5']}


def draw_network(ax, g, pos, seeds, candidates, explored, queried,
                 scores=None, labels=None, stage=''):
    scores = scores or {}
    known = [(u,v) for u,v in g.edges if u in queried or v in queried]
    unknown = [(u,v) for u,v in g.edges if u not in queried and v not in queried]
    toy = labels is not None
    for edges, color, width, alpha in ((unknown,'#B8BEC7',1.3 if toy else .55,.75),
                                       (known,'#000000',1.65 if toy else 1.15,.9)):
        nx.draw_networkx_edges(g,pos,edgelist=edges,edge_color=color,width=width,
                               alpha=alpha,ax=ax)
   

    ranks = {}
    if stage == 'scored' and scores:
        unique_scores = sorted(set(scores.values()), reverse=True)
        rank_map = {s: i for i, s in enumerate(unique_scores)}
        ranks = {
            u: CMAP(1 - rank_map[s] / max(1, len(unique_scores) - 1))
            for u, s in scores.items()
        }
    sizes = (845,793,728,728) if toy else (86,86,70,70)
    colors = {}
    for nodes, color, size in ((set(g)-seeds-candidates-explored,GRAY,sizes[3]),
                               (explored,BLUE,sizes[2]),(candidates-seeds,RED,sizes[1]),
                               (seeds,GREEN,sizes[0])):
        if not nodes:
            continue
        palette = [ranks.get(u, RED) if color == RED and stage == 'scored' else color
                   for u in sorted(nodes)]
        nx.draw_networkx_nodes(g,pos,nodelist=sorted(nodes),node_color=palette,
                               node_size=size,linewidths=0,ax=ax)
        colors.update(zip(sorted(nodes),palette))
    if toy:
        for u in g:
            rgb = np.array(to_rgb(colors[u]))
            linear = np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
            white = linear @ np.array([.2126,.7152,.0722]) < .4 or (
                stage!='seeds' and labels[u] in {'4','8','11','7'})
            nx.draw_networkx_labels(g,pos,{u:labels[u]},font_size=18,
                                    font_weight='bold',font_color='white' if white else '#263238',ax=ax)
    ax.set_axis_off()


def neighbors(g, nodes):
    return set().union(*(set(g[u]) for u in nodes)) if nodes else set()


def draw_toy(ax, stage, letter, toy):
    g,pos,labels,seeds = toy
    candidates = neighbors(g, seeds) if stage!='seeds' else set()
    first = neighbors(g, candidates)
    second = neighbors(g, first)
    explored = ((first|second) if stage=='scored' else first if stage=='first_neighbors'
                else set())-seeds-candidates
    queried = (seeds|candidates|first if stage=='scored' else seeds|candidates
               if stage=='first_neighbors' else seeds if stage=='candidates' else set())
    draw_network(ax,g,pos,seeds,candidates,explored,queried,
                 {u:score(g,u) for u in candidates},labels,stage)
    xy = np.array(list(pos.values()))
    center = (xy.min(0)+xy.max(0))/2
    radius=np.ptp(xy,axis=0).max()/2+.4
    ax.set(xlim=(center[0]-radius,center[0]+radius),
           ylim=(center[1]-radius,center[1]+radius),aspect='equal')
    ax.text(.04,.96,letter,transform=ax.transAxes,va='top',fontsize=26,fontweight='bold')

# ============================================================
# 3. Load Existing PPI Results
# ============================================================
def load_results():
    summary_path = RESULTS / 'summary.csv'
    config_path = RESULTS / 'config.json'
    with config_path.open(encoding='utf-8') as f:
        config = json.load(f)
    df = pd.read_csv(summary_path)
    df = df[(df['protocol'].str.lower() == 'seamless') & (df['m'] == M)]
    history = (df.sort_values('removed')
               [['removed', 'mean_S1_over_N', 'mean_S2_over_N']]
               .to_numpy(float))

    g = nx.read_edgelist(EDGES, nodetype=int)
    g.remove_edges_from(nx.selfloop_edges(g))
    g = g.subgraph(max(nx.connected_components(g), key=len)).copy()
 
    # One reproducible attack solely to visualize Panel E.
    rng = np.random.default_rng(int(config.get('seed', 12345)))
    remaining = g.copy()
    targets = {round(len(g)*f): f for f in FRACTIONS}
    snapshots = {}
    for step in range(max(targets)+1):
        seeds = set(map(int, rng.choice(list(remaining),
                         size=min(M,len(remaining)), replace=False)))
        candidates = neighbors(remaining, seeds)
        scores = {u: score(remaining,u) for u in candidates}
        if step in targets:
            snapshots[targets[step]] = (remaining.copy(), seeds,
                                        candidates.copy())
        if step == max(targets):
            break
        pool = candidates if candidates else seeds
        best_score = max((scores.get(u, 0) for u in pool), default=0)
        best = sorted(u for u in pool if scores.get(u, 0) == best_score)
        remaining.remove_node(int(rng.choice(best)))
    return g, snapshots, history

# ============================================================
# 4. Figure Layout and SVG Export
# ============================================================
def main():
    g, snapshots, history = load_results()
    pos = nx.spring_layout(g, seed=7, k=2.3/np.sqrt(len(g)), iterations=1000)
    fig = plt.figure(figsize=(21, 11.5))

    # Panels A-D
    toy = toy_network()
    top = fig.add_gridspec(1, 4, left=.035, right=.965, bottom=.54, top=.96, wspace=.12)
    for i, stage in enumerate(('seeds', 'candidates', 'first_neighbors', 'scored')):
        draw_toy(fig.add_subplot(top[0, i]), stage, 'ABCD'[i], toy)

    # Panel E
    bottom = fig.add_gridspec(
        1, 7, left=.035, right=.965, bottom=.14, top=.49,
        width_ratios=(1, .12, 1, .12, 1, .20, 1.25), wspace=.02
    )
    axes = []
    for col, f in zip((0, 2, 4), FRACTIONS):
        ax = fig.add_subplot(bottom[0, col])
        sub, seeds, candidates = snapshots[f]
        first = neighbors(sub, candidates)
        explored = (first | neighbors(sub, first)) - seeds - candidates

        draw_network(
            ax, sub, pos, seeds, candidates, explored,
            seeds | candidates | first,
            {u: score(sub, u) for u in candidates},
            stage='scored'
        )
        ax.text(.5, .035, f'$f={f:.2f}$' if f > 0 else '$f=0$',
                transform=ax.transAxes, ha='center', fontsize=26)
        axes.append(ax)

    for col in (1, 3, 5):
        ax = fig.add_subplot(bottom[0, col])
        ax.axis('off')
        ax.text(.0, .5, '...', transform=ax.transAxes, ha='center', fontsize=26)

    box0, box2 = axes[0].get_position(), axes[-1].get_position()
    cax = fig.add_axes([
        box0.x0 + .06*(box2.x1-box0.x0), .085,
        .88*(box2.x1-box0.x0), .012
    ])
    cb = fig.colorbar(
        ScalarMappable(norm=Normalize(0, 1), cmap=CMAP),
        cax=cax, orientation='horizontal'
    )
    cb.set_ticks([])
    
    fig.text(box2.x1, .49, 'E', ha='right', va='top',
             fontsize=26, fontweight='bold')

    # Panel F
    ax = fig.add_subplot(bottom[0, 6])
    n = len(g)

    for y, ls, label in (
        (1, (0, (3.7, 1.6)), r'$S_1/N$'),
        (2, '-', r'$S_2/N$')
    ):
        ax.plot(history[:, 0], history[:, y], color=RED,
                ls=ls, lw=1.87, marker='o', ms=3, label=label)

    ax.set(
        xlabel='Number of removed nodes',
        ylabel='Component size',
        xlim=(-.04*n, 1.04*n),
        ylim=(-.04, 1.04)
    )
    ax.set_xticks(np.rint(np.linspace(0, n, 5)).astype(int))
    ax.set_yticks(
        np.linspace(0, 1, 6),
        labels=[f'{v:.1f}' if v > 0 else '0' for v in np.linspace(0, 1, 6)]
    )
    ax.set_box_aspect(1)
    ax.tick_params(labelsize=26)
    ax.grid(color='#D9D9D9')
    ax.xaxis.label.set_size(26)
    ax.yaxis.label.set_size(26)
    ax.legend(frameon=False, fontsize=22, loc='lower right')
    ax.text(.98, .98, 'F', transform=ax.transAxes,
            ha='right', va='top', fontsize=26, fontweight='bold')

    # SVG Export
    out = ROOT / 'seamless_figures'
    out.mkdir(exist_ok=True)
    fig.savefig(out / 'seamless.svg', bbox_inches='tight',
                pad_inches=0, transparent=True)
    plt.show()

if __name__=='__main__':
    main()
