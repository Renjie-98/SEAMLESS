
"""
Figure_PPI: PPI adaptive fragility and biological associations.

Read precomputed attack curves for six strategies (SEAMLESS: m = 4).
Analyze 1 - AUC(S2/N), biological features, and regression statistics.
Bootstrap: 2000 resamples for 95% confidence intervals.

Read existing results only; never rerun SEAMLESS attacks.
Output: seamless_figures/PPI.svg (Panels A-E).
"""

from __future__ import annotations
import json
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.ticker import FuncFormatter, MaxNLocator
from scipy.stats import linregress, mannwhitneyu

# ============================================================
# 1. Configuration
# ============================================================
BASE_DIR = Path.cwd()
PPI_SI_DIR = BASE_DIR / 'benchmarks' / 'PPI_SI'
RESULTS_DIR = BASE_DIR / 'results_PPI'
SPECIES_FILE = PPI_SI_DIR / 'treeoflife.species.tsv'
REGULATORY_FILE = PPI_SI_DIR / 'treeoflife.species.regulatorygenes.xls'
ENVIRONMENT_FILE = PPI_SI_DIR / 'treeoflife.species.environments.xls'
PROTOCOLS = ['seamless', 'random', 'degree', 'adap_degree', 'betweenness', 'adap_betweenness']
PROTOCOL_DISPLAY_NAMES = {
    'seamless': 'SEAMLESS',
    'random': 'Random',
    'degree': 'Degree',
    'adap_degree': 'Adaptive degree',
    'betweenness': 'Betweenness',
    'adap_betweenness': 'Adaptive betweenness',
}
SEAMLESS_M = 4.0
RANDOM_SEED = 4567
BOOTSTRAP_N = 2000
BOOTSTRAP_CI = 95
BOOTSTRAP_SEED = 4567
TEXT_SIZE = 26
PANEL_WIDTH_IN = 4.6315
PANEL_GAP_IN = 1.2913
ROW_GAP_IN = 1.8
BOTTOM_MARGIN_IN = 2.28
TOP_MARGIN_IN = 0.6885
FIGSIZE = (20.5, TOP_MARGIN_IN + 2 * PANEL_WIDTH_IN + ROW_GAP_IN + BOTTOM_MARGIN_IN)
SHOW_FIGURE = True
SCATTER_GRAY = '#8A8A8A'
NODE_SIZE_MIN = 90.0
NODE_SIZE_MAX = 260.0
NETWORK_N_COL = 'Network_N'
MARKER_SIZE_COL = 'Marker_size'
SELECTED_SPECIES_COLORS = ['#2072AA', '#339A39', '#D02623']
SELECTED_SPECIES_FILL_COLORS = ['#A3C7DA', '#AECC8A', '#E49698']
SELECTED_SPECIES_FILL_MAP = dict(zip(SELECTED_SPECIES_COLORS, SELECTED_SPECIES_FILL_COLORS))
HABITAT_COLOR = '#E17927'
HABITAT_FACE = '#ECBA73'
HABITAT_BG = '#FDF6EB'
OXYGEN_COLOR = '#623F88'
OXYGEN_FACE = '#C4B1D1'
OXYGEN_BG = '#F7F3FA'
UNKNOWN_CATEGORY_VALUES = {'', 'unknown', 'unk', 'not known', 'not-known', 'not available', 'not applicable', 'na', 'n/a', 'nan', 'none', 'null', '-'}
METRIC_COL = 'One_minus_AUC_S2'
METRIC_LABEL = '$1-\\mathrm{AUC}(S_2/N)$'
PANEL_B_METRIC_TICKS = np.array([0.96, 0.97, 0.98, 0.99, 1.0])
PANEL_E_METRIC_TICKS = np.array([0.9, 0.92, 0.94, 0.96, 0.98, 1.0])
PANEL_C_METRIC_TICKS = np.array([0.92, 0.94, 0.96, 0.98, 1.0])
PANEL_D_METRIC_TICKS = np.array([0.92, 0.94, 0.96, 0.98, 1.0])
PANEL_CE_Y_TICKS = np.linspace(0.0, 0.16, 5)
PANEL_D_Y_TICKS = np.linspace(0.0, 60.0, 5)
PANEL_FEATURES = {
    'C': ('Fraction_regulatory_genes', 'Fraction of regulatory genes', PANEL_C_METRIC_TICKS, PANEL_CE_Y_TICKS, 'two'),
    'D': ('Environmental_scope_index', 'Environmental scope index', PANEL_D_METRIC_TICKS, PANEL_CE_Y_TICKS, 'two'),
    'E': ('Co_habitation_index', 'Co-habitation index', PANEL_E_METRIC_TICKS, PANEL_D_Y_TICKS, 'integer'),
}

MANUAL_SPECIES_IDS = {
    'Buchnera aphidicola Cc': '372461',
    'Pseudomonas fluorescens PfO-1': '205922',
    'Xanthomonas campestris pv. campestris ATCC 33913': '190485',
    'Xanthomonas campestris pv. campestris 8004': '314565',
    'Pseudomonas syringae pv. tomato DC3000': '223283',
    'Rickettsia canadensis McKiel': '293613',
    'Rickettsia akari Hartford': '293614',
    'Leptospira borgpetersenii L550': '355276',
    'Tropheryma whipplei Twist': '203267',
    'Listeria welshimeri SLCC5334': '386043',
    'Lactobacillus delbrueckii ATCC 11842': '390333',
    'Ehrlichia ruminantium Welgevonden (France)': '254945',
    'Polynucleobacter sp. QLW-P1DMWA-1': '312153',
    'Shigella flexneri 301 (serotype 2a)': '198214',
    'Anabaena sp. PCC7120': '103690',
    'Salmonella typhimurium LT2': '99287',
    'Salmonella enterica serovar Typhi CT18': '220341',
}

# ============================================================
# 2. Data loading
# ============================================================
def normalize_col_name(value) -> str:
    return re.sub('[^a-z0-9]+', '', str(value).strip().lower())

def normalize_name(value) -> str:
    if pd.isna(value):
        return ''
    value = str(value).strip().casefold()
    return re.sub('[\\s_\\-\'"]+', '', value)

def clean_species_id(value):
    if pd.isna(value):
        return np.nan
    value = str(value).strip()
    value = re.sub('\\.0$', '', value)
    return value if value else np.nan

def species_id_sort_key(value):
    try:
        return (0, int(str(value)))
    except Exception:
        return (1, str(value))

def find_column(df: pd.DataFrame, candidates: list[str], file_label: str) -> str:
    normalized = {normalize_col_name(column): column for column in df.columns}
    for candidate in candidates:
        key = normalize_col_name(candidate)
        if key in normalized:
            return normalized[key]
    raise ValueError(f'Cannot find any expected column in {file_label}: {candidates}\nAvailable columns: {df.columns.tolist()}')

def read_first_nonempty_excel_sheet(path: Path) -> pd.DataFrame:
    try:
        workbook = pd.ExcelFile(path)
    except ImportError as exc:
        raise ImportError('Reading .xls files requires xlrd. Install it with:\npython -m pip install xlrd') from exc
    for sheet_name in workbook.sheet_names:
        dataframe = pd.read_excel(path, sheet_name=sheet_name)
        if dataframe.shape[0] > 0 and dataframe.shape[1] > 0:
            return dataframe
    raise RuntimeError(f'No non-empty sheet found in: {path}')

def normalize_p(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    if p_values.size == 0:
        return p_values
    if np.nanmax(p_values) > 1.5:
        p_values = p_values / 100.0
    return p_values

def trapezoid_auc(x_values: np.ndarray, y_values: np.ndarray) -> float:
    if hasattr(np, 'trapezoid'):
        return float(np.trapezoid(y_values, x_values))
    return float(np.trapz(y_values, x_values))

def is_unknown_category(value) -> bool:
    if pd.isna(value):
        return True
    key = str(value).strip().lower()
    return key in UNKNOWN_CATEGORY_VALUES

CATEGORY_ALIASES = {
    'Habitat': {'terrestrial': 'Terrestrial', 'multiple': 'Multiple',
                'aquatic': 'Aquatic', 'specialized': 'Specialized',
                'host-associated': 'Host cell', 'host associated': 'Host cell',
                'host cell': 'Host cell'},
    'Oxygen': {'aerobic': 'Aerobic', 'microaerophilic': 'Microaerophilic',
               'facultative': 'Facultative', 'anaerobic': 'Anaerobic'},
}

def canonical_category(value, category):
    if is_unknown_category(value):
        return np.nan
    text = str(value).strip()
    return CATEGORY_ALIASES[category].get(re.sub('\\s+', ' ', text.lower()), text)

def build_species_mapping(species_df):
    ids = find_column(species_df, ['Species_ID', 'Species ID'], 'species')
    compact = find_column(species_df, ['Compact_name', 'Compact name'], 'species')
    official = find_column(species_df, ['Official_NCBI_name', 'Official NCBI name'], 'species')
    species = species_df.rename(columns={ids: 'Species_ID', compact: 'Compact_name', official: 'Official_NCBI_name'}).copy()
    species['Species_ID'] = species['Species_ID'].map(clean_species_id)
    manual = MANUAL_SPECIES_IDS
    rows = [{'Species_ID': sid, 'Matched_name': name, 'Matched_by': 'Manual correction'} for name, sid in manual.items()]
    for col in ('Compact_name', 'Official_NCBI_name'):
        rows += [{'Species_ID': row.Species_ID, 'Matched_name': getattr(row, col), 'Matched_by': col} for row in species.itertuples(index=False)]
    mapping = pd.DataFrame(rows)
    mapping['name_key'] = mapping['Matched_name'].map(normalize_name)
    mapping = mapping.dropna(subset=['Species_ID']).query('name_key != ""')
    conflicts = mapping.groupby('name_key')['Species_ID'].nunique()
    ambiguous = conflicts[conflicts > 1].index
    if len(ambiguous):
        manual_keys = {normalize_name(name) for name in MANUAL_SPECIES_IDS}
        unresolved = set(ambiguous) - manual_keys
        if unresolved:
            print(f'[WARNING] {len(unresolved)} ambiguous species names; excluding them.')
            mapping = mapping[~mapping['name_key'].isin(unresolved)]
    mapping = mapping.drop_duplicates('name_key')
    names = species[['Species_ID', 'Compact_name']].drop_duplicates('Species_ID').copy()
    names['Compact_name'] = names['Compact_name'].map(lambda x: str(x).strip().replace('_', ' ') if pd.notna(x) else np.nan)
    return (species, mapping, names)

def match_table_to_species_id(df: pd.DataFrame, *, name_column: str, name_map: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['name_key'] = df[name_column].apply(normalize_name)
    matched = df.merge(name_map[['name_key', 'Species_ID', 'Matched_name', 'Matched_by']], on='name_key', how='left')
    matched['Species_ID'] = matched['Species_ID'].apply(clean_species_id)
    return matched

def ppi_result_species_id(folder_name: str) -> str | None:
    """Recognize results_PPI_<id> and results_PPI_full_<id>."""
    match = re.fullmatch(r'results_PPI_(?:full_)?(.+)', folder_name, flags=re.IGNORECASE)
    if match is None:
        return None
    species_id = clean_species_id(match.group(1))
    return str(species_id) if pd.notna(species_id) else None

def find_ppi_result_folders(results_dir: Path) -> dict[str, Path]:
    """Index available PPI result folders once, keyed by species ID."""
    folders = {}
    for folder in sorted(results_dir.iterdir(), key=lambda path: path.name.lower()):
        if not folder.is_dir():
            continue
        species_id = ppi_result_species_id(folder.name)
        if species_id is None or not (folder / 'summary.csv').is_file():
            continue
        if species_id in folders:
            print(f'[WARNING] Duplicate result folder for species {species_id}; '
                  f'using {folders[species_id]}, ignoring {folder}')
            continue
        folders[species_id] = folder
    return folders


def panel_3_5_eligible_species_ids(display_df, regulatory_feature,
                                   environment_feature, result_ids):
    """Preserve the reference marker-normalization population."""
    biological = (
        display_df[['Species_ID']].drop_duplicates('Species_ID')
        .merge(regulatory_feature[['Species_ID', 'Fraction_regulatory_genes']],
               on='Species_ID', how='left')
        .merge(environment_feature[['Species_ID', 'Environmental_scope_index',
                                    'Co_habitation_index']], on='Species_ID', how='left')
    )
    feature_columns = ['Fraction_regulatory_genes',
                       'Environmental_scope_index', 'Co_habitation_index']
    for column in feature_columns:
        biological[column] = pd.to_numeric(biological[column], errors='coerce')
    biological_ids = set(biological.loc[
        biological[feature_columns].notna().any(axis=1), 'Species_ID'
    ].dropna().astype(str))
    display_ids = set(display_df['Species_ID'].dropna().astype(str))
    return sorted(set(result_ids) & display_ids & biological_ids,
                  key=species_id_sort_key)


def collect_ppi_network_node_counts(eligible_species_ids, result_folders):
    """Read config.json n_nodes; scale markers globally across eligible species."""
    rows = []
    for species_id in eligible_species_ids:
        folder = result_folders.get(str(species_id))
        if folder is None:
            continue
        config_path = folder / 'config.json'
        if not config_path.is_file():
            print(f'[WARNING] Skipped species {species_id}: missing {config_path}')
            continue
        try:
            with config_path.open('r', encoding='utf-8') as handle:
                count = int(json.load(handle)['n_nodes'])
            if count <= 0:
                raise ValueError(f'n_nodes must be positive, got {count}')
        except (OSError, UnicodeError, ValueError, TypeError, KeyError) as exc:
            print(f'[WARNING] Skipped species {species_id}: invalid {config_path}: {exc}')
            continue
        rows.append({'Species_ID': str(species_id), NETWORK_N_COL: float(count)})
    if not rows:
        raise RuntimeError(f'No valid PPI config.json node counts found in {RESULTS_DIR}.')
    result = pd.DataFrame(rows)
    counts = result[NETWORK_N_COL].to_numpy(dtype=float)
    lo, hi = float(counts.min()), float(counts.max())
    if hi > lo:
        result[MARKER_SIZE_COL] = NODE_SIZE_MIN + (counts - lo) / (hi - lo) * (NODE_SIZE_MAX - NODE_SIZE_MIN)
    else:
        result[MARKER_SIZE_COL] = (NODE_SIZE_MIN + NODE_SIZE_MAX) / 2.0
    return result

def extract_protocol_curve(frame: pd.DataFrame, protocol: str, species_id: str):
    """Extract a protocol from a normalized summary; preserve original AUC rules."""
    subset = frame
    subset = subset.dropna(subset=['p', 'S1_over_N', 'S2_over_N'])
    if len(subset) < 2:
        print(f'[WARNING] Skipped species {species_id}, protocol {protocol}: '
              f'fewer than 2 valid data points ({len(subset)}).')
        return None
    curve = (subset.groupby('p', as_index=False)[['S1_over_N', 'S2_over_N']]
             .mean().sort_values('p').reset_index(drop=True))
    curve['p'] = normalize_p(curve['p'].to_numpy(dtype=float))
    curve = curve.sort_values('p').drop_duplicates('p').reset_index(drop=True)
    if len(curve) < 2:
        print(f'[WARNING] Skipped species {species_id}, protocol {protocol}: '
              f'fewer than 2 distinct valid p values ({len(curve)}).')
        return None
    auc = trapezoid_auc(curve['p'].to_numpy(), curve['S2_over_N'].to_numpy())
    return curve, auc

def normalize_summary(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Resolve variable CSV headers once, not once per attack protocol."""
    columns = {
        'protocol': ['protocol'],
        'p': ['p', 'fraction removed', 'fraction_removed'],
        'S1_over_N': ['mean_S1_over_N', 'S1_over_N', 'mean_S1/N', 'S1/N', 'meanS1overN'],
        'S2_over_N': ['mean_S2_over_N', 'S2_over_N', 'mean_S2/N', 'S2/N', 'meanS2overN'],
    }
    renamed = {find_column(frame, options, str(path)): key for key, options in columns.items()}
    m_column = next((col for col in frame if normalize_col_name(col) == 'm'), None)
    if m_column is not None:
        renamed[m_column] = 'm'
    data = frame.rename(columns=renamed).copy()
    data['protocol'] = data['protocol'].astype(str).str.strip().str.lower()
    for col in ('p', 'S1_over_N', 'S2_over_N'):
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data['m'] = (pd.to_numeric(data['m'], errors='coerce')
                 if 'm' in data else np.nan)
    return data

def build_regulatory_features(frame, name_map):
    name = find_column(frame, ['Full name', 'Species', 'Name', 'Official_NCBI_name', 'Compact_name'], 'regulatory')
    value = find_column(frame, ['Fraction of regulatory genes', 'Fraction regulatory genes', 'Regulatory genes fraction'], 'regulatory')
    frame = frame.rename(columns={name: 'Regulatory_species_name', value: 'Fraction_regulatory_genes'}).copy()
    frame['Fraction_regulatory_genes'] = pd.to_numeric(frame['Fraction_regulatory_genes'], errors='coerce')
    matched = match_table_to_species_id(frame, name_column='Regulatory_species_name', name_map=name_map)
    return matched[['Species_ID', 'Fraction_regulatory_genes']].dropna(subset=['Species_ID']).sort_values('Species_ID').drop_duplicates('Species_ID')

def build_environment_features(frame, name_map):
    candidates = {'Environment_species_name': ['Full name', 'Compact_name', 'Species', 'Name', 'Official_NCBI_name'], 'Co_habitation_index': ['Max CHS (seeds)', 'Max CHS', 'CHS', 'Co-habitation index', 'Co-habitation'], 'Environmental_scope_index': ['Environmental diversity I', 'Environmental diversity', 'Environmental scope index'], 'Habitat': ['Habitat', 'Ecological habitat'], 'Oxygen': ['Oxygen', 'Oxygen requirement', 'Oxygen_requirement']}
    frame = frame.rename(columns={find_column(frame, choices, 'environment'): key for key, choices in candidates.items()}).copy()
    for col in ('Co_habitation_index', 'Environmental_scope_index'):
        frame[col] = pd.to_numeric(frame[col], errors='coerce')
    frame.loc[frame['Co_habitation_index'] == 0, 'Co_habitation_index'] = np.nan
    for col in ('Habitat', 'Oxygen'):
        frame[col] = frame[col].map(lambda x: canonical_category(x, col))
    matched = match_table_to_species_id(frame, name_column='Environment_species_name', name_map=name_map)
    return matched[['Species_ID', 'Co_habitation_index', 'Environmental_scope_index', 'Habitat', 'Oxygen']].dropna(subset=['Species_ID']).sort_values('Species_ID').drop_duplicates('Species_ID')

def load_protocol_results(result_folders: dict[str, Path], eligible_species_ids):
    """Load each available attack protocol independently for every PPI species.
    The six-protocol intersection is used only by compare_protocols(),
    never to restrict the SEAMLESS species shown in Panels A-E.
    """
    eligible = set(map(str, eligible_species_ids))
    records = {protocol: [] for protocol in PROTOCOLS}
    curves = {'seamless': {}}
    for species_id, folder in result_folders.items():
        if species_id not in eligible:
            continue
        summary_file = folder / 'summary.csv'
        try:
            frame = normalize_summary(pd.read_csv(summary_file), summary_file)
        except (OSError, UnicodeError, pd.errors.ParserError,
                pd.errors.EmptyDataError, ValueError) as exc:
            print(f'[WARNING] Cannot process {summary_file}: {exc}')
            continue
        for protocol, protocol_frame in frame.groupby('protocol', sort=False):
            if protocol not in records:
                continue
            if protocol == 'seamless':
                protocol_frame = protocol_frame.loc[
                    np.isclose(protocol_frame['m'], SEAMLESS_M, equal_nan=False)
                ]
                if protocol_frame.empty:
                    continue
            result = extract_protocol_curve(protocol_frame, protocol, species_id)
            if result is None:
                continue
            curve, auc = result
            records[protocol].append({
                'Species_ID': species_id,
                'protocol': protocol,
                'AUC_S2': auc,
                METRIC_COL: 1.0 - auc,
                'summary_file': str(summary_file),
            })
            if protocol == 'seamless':
                curves['seamless'][species_id] = curve
    columns = ['Species_ID', 'protocol', 'AUC_S2', METRIC_COL, 'summary_file']
    metrics = {
        protocol: pd.DataFrame(rows, columns=columns).drop_duplicates('Species_ID')
        for protocol, rows in records.items()
    }
    return metrics, curves

# ============================================================
# 3. Statistical analysis
# ============================================================
def bootstrap_mean_ci(values: pd.Series, *, n_boot: int=BOOTSTRAP_N, ci: float=BOOTSTRAP_CI, seed: int=BOOTSTRAP_SEED):
    values = pd.to_numeric(values, errors='coerce').dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return (np.nan, np.nan, np.nan)
    mean_value = float(np.mean(values))
    if len(values) == 1:
        return (mean_value, mean_value, mean_value)
    rng = np.random.default_rng(seed)
    bootstrap_samples = rng.choice(values, size=(n_boot, len(values)), replace=True)
    bootstrap_means = bootstrap_samples.mean(axis=1)
    alpha = (100.0 - ci) / 2.0
    ci_low = float(np.percentile(bootstrap_means, alpha))
    ci_high = float(np.percentile(bootstrap_means, 100.0 - alpha))
    return (mean_value, ci_low, ci_high)

def linear_regression_statistics(df: pd.DataFrame, x_col: str, y_col: str, *, prepared=False):
    if prepared:
        plot_data = df
    else:
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

def representative_species(data):
    required = ['Compact_name', METRIC_COL, 'Fraction_regulatory_genes', 'Co_habitation_index', 'Environmental_scope_index', NETWORK_N_COL, MARKER_SIZE_COL]
    pool = data.dropna(subset=required).drop_duplicates('Species_ID').copy()
    pool['_order'] = pool['Species_ID'].map(species_id_sort_key)
    pool = pool.sort_values('_order').drop(columns='_order')
    if len(pool) < 3:
        raise RuntimeError('Fewer than three PPI species have complete data.')
    return pool.sample(n=3, random_state=RANDOM_SEED)

def compare_protocols(merged):
    """Use common species for each feature; do not alter plotted SEAMLESS regressions."""
    specs = [('C', 'Fraction of regulatory genes', 'Fraction_regulatory_genes'), ('D', 'Environmental scope index', 'Environmental_scope_index'), ('E', 'Co-habitation index', 'Co_habitation_index')]
    rows = []
    for panel, feature, column in specs:
        if any((p not in merged for p in PROTOCOLS)):
            print(f'[WARNING] {feature}: missing protocol, comparison skipped.')
            continue
        valid = {p: merged[p].dropna(subset=[METRIC_COL, column, NETWORK_N_COL, MARKER_SIZE_COL]) for p in PROTOCOLS}
        common = set.intersection(*(set(df['Species_ID'].astype(str)) for df in valid.values()))
        for protocol in PROTOCOLS:
            data = valid[protocol][valid[protocol]['Species_ID'].astype(str).isin(common)]
            stats = linear_regression_statistics(data, METRIC_COL, column)
            rows.append({'Attack strategy': PROTOCOL_DISPLAY_NAMES[protocol], 'Panel': panel, 'Biological feature': feature, 'n': stats['n'] if stats else np.nan, 'R_squared': stats['r_squared'] if stats else np.nan, 'p_value': stats['p_value'] if stats else np.nan, 'AIC': stats['aic'] if stats else np.nan, 'BIC': stats['bic'] if stats else np.nan})
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    grouped = result.groupby('Biological feature')
    result['Rank'] = grouped['AIC'].rank(method='min', na_option='bottom')
    result['Delta_AIC'] = result['AIC'] - grouped['AIC'].transform('min')
    result['Delta_BIC'] = result['BIC'] - grouped['BIC'].transform('min')
    return result.sort_values(['Panel', 'AIC', 'BIC', 'Attack strategy'], na_position='last').reset_index(drop=True)

def print_regression_comparison(models: pd.DataFrame) -> None:
    """Display already-computed six-strategy results; never refit models."""
    print("\nPanels C-E: six-strategy regression comparison")
    if models.empty:
        print("[WARNING] No comparable six-strategy regression results.")
        return
    features = (
        ("C", "Fraction of regulatory genes"),
        ("D", "Environmental scope index"),
        ("E", "Co-habitation index"),
    )
    for panel, title in features:
        print(f"\n{title} (Panel {panel})")
        subset = models.loc[models["Panel"] == panel]
        if subset.empty:
            print("[WARNING] No common valid species across all six strategies.")
            continue
        table = subset[["Rank", "Attack strategy", "n", "R_squared",
                        "p_value", "AIC", "BIC", "Delta_AIC", "Delta_BIC"]].rename(
            columns={"Delta_AIC": "ΔAIC", "Delta_BIC": "ΔBIC"}
        )
        formatters = {
            "Rank": lambda x: f"{int(x)}" if pd.notna(x) else "NA",
            "n": lambda x: f"{int(x)}" if pd.notna(x) else "NA",
            "R_squared": lambda x: f"{x:.3f}" if pd.notna(x) else "NA",
            "p_value": lambda x: f"{x:.3e}" if pd.notna(x) else "NA",
            **{name: (lambda x: f"{x:.3f}" if pd.notna(x) else "NA")
               for name in ("AIC", "BIC", "ΔAIC", "ΔBIC")},
        }
        print(table.to_string(index=False, formatters=formatters))

def prepare_panel_data(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Prepare each panel's numeric data once for regression and plotting."""
    valid = data.copy()
    numeric_columns = {METRIC_COL, NETWORK_N_COL, MARKER_SIZE_COL}
    numeric_columns.update(config[0] for config in PANEL_FEATURES.values())
    for column in numeric_columns:
        valid[column] = pd.to_numeric(valid[column], errors='coerce')
    valid = valid.dropna(subset=[METRIC_COL])
    panels = {
        'Habitat': valid.dropna(subset=['Habitat']),
        'Oxygen': valid.dropna(subset=['Oxygen']).loc[
            lambda frame: frame['Oxygen'].astype(str) != 'Microaerophilic'
        ],
    }
    for panel, (column, *_rest) in PANEL_FEATURES.items():
        panels[panel] = valid.dropna(
            subset=[column, NETWORK_N_COL, MARKER_SIZE_COL]
        )
    return panels

def prepare_panel_b_statistics(panel_data: dict[str, pd.DataFrame]) -> dict:
    """Calculate grouped bootstrap intervals and extreme-group tests once."""
    results = {}
    for category in ('Habitat', 'Oxygen'):
        samples = {}
        rows = []
        for name, group in panel_data[category].groupby(category, dropna=True):
            values = pd.to_numeric(group[METRIC_COL], errors='coerce').dropna()
            if values.empty:
                continue
            samples[name] = values
            mean, low, high = bootstrap_mean_ci(values, seed=BOOTSTRAP_SEED)
            rows.append({'category': name, 'mean': mean, 'ci_low': low,
                         'ci_high': high, 'n': len(values)})
        summary = pd.DataFrame(rows, columns=['category', 'mean', 'ci_low', 'ci_high', 'n'])
        summary = summary.sort_values(['mean', 'category'], ascending=[False, True]).reset_index(drop=True)
        p_value = None
        if len(summary) >= 2:
            high = samples[summary.iloc[0]['category']]
            low = samples[summary.iloc[-1]['category']]
            if len(high) >= 2 and len(low) >= 2:
                p_value = float(mannwhitneyu(high, low, alternative='two-sided').pvalue)
        results[category] = {'summary': summary, 'p_value': p_value}
    return results

# ============================================================
# 4. Figure plotting
# ============================================================
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

def apply_figure3_panel_geometry(ax) -> None:
    ax.set_box_aspect(1.0)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontname('Times New Roman')
        tick_label.set_fontsize(TEXT_SIZE)
    ax.xaxis.set_label_coords(0.5, -0.13)

def format_p_value(p_value: float) -> str:
    if pd.isna(p_value):
        return ''
    if p_value < 0.01:
        exponent = int(np.floor(np.log10(p_value)))
        mantissa = p_value / 10 ** exponent
        return f'$p={mantissa:.1f}\\times10^{{{exponent}}}$'
    return f'$p={p_value:.2f}$'

def add_vertical_category_bracket(ax, x1: float, x2: float, y: float, text: str, *, cap: float=0.0009) -> None:
    ax.plot([x1, x1, x2, x2], [y - cap, y, y, y - cap], color='black', linewidth=1.5, clip_on=True, zorder=8)
    ax.annotate(text, xy=((x1 + x2) / 2.0, y), xytext=(0, -4), textcoords='offset points', ha='center', va='top', fontsize=TEXT_SIZE - 3, fontname='Times New Roman', clip_on=True, zorder=9)

def tick_formatter(kind):
    if kind == 'two':
        return lambda value, pos: f'{value:.2f}'
    if kind == 'integer':
        return lambda value, pos: f'{int(round(value))}'
    return lambda value, pos: '0' if np.isclose(value, 0.0) else f'{value:.4g}'

def configure_axis(ax, *, x_ticks=None, y_ticks=None, x_formatter=None, y_formatter=None, pad_fraction=0.04, style=False, grid=False):
    """Configure ticks, limits, typography and (optionally) axis styling."""
    if grid:
        ax.grid(True, color='#D9D9D9', linewidth=0.8, alpha=0.9, zorder=0)
    for axis, ticks, formatter in ((ax.xaxis, x_ticks, x_formatter), (ax.yaxis, y_ticks, y_formatter)):
        if ticks is None:
            continue
        ticks = np.asarray(ticks, dtype=float)
        span = float(ticks[-1] - ticks[0])
        pad = span * pad_fraction if span > 0 else pad_fraction
        limits = (float(ticks[0] - pad), float(ticks[-1] + pad))
        (ax.set_xlim if axis is ax.xaxis else ax.set_ylim)(*limits)
        axis.set_ticks(ticks)
        if formatter is not None:
            axis.set_major_formatter(FuncFormatter(formatter))
    if style:
        ax.tick_params(axis='both', which='major', labelsize=TEXT_SIZE, length=5, width=0.9, direction='out')
        for spine in ax.spines.values():
            spine.set_color('#444444')
            spine.set_linewidth(0.9)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Times New Roman')
        label.set_color('black')
        label.set_fontsize(TEXT_SIZE)

def plot_panel_a(ax, selected_species: pd.DataFrame, curve_store: dict[str, pd.DataFrame], selected_color_map: dict[str, str]) -> None:
    species_handles = []
    for row in selected_species.itertuples(index=False):
        species_id = str(row.Species_ID)
        species_name = str(row.Compact_name)
        if species_name.strip() == 'Pseudomonas putida KT2440':
            species_name = 'Pseudomonas putida'
        color = selected_color_map[species_id]
        curve = curve_store[species_id].sort_values('p').copy()
        x_values = pd.to_numeric(curve['p'], errors='coerce').to_numpy(dtype=float)
        y_values = pd.to_numeric(curve['S2_over_N'], errors='coerce').to_numpy(dtype=float)
        valid = np.isfinite(x_values) & np.isfinite(y_values)
        x_values = x_values[valid]
        y_values = y_values[valid]
        if x_values.size == 0:
            print(f'[WARNING] Skipped Panel A curve for species {species_id}: no finite points.')
            continue
        ax.fill_between(x_values, 0.0, y_values, facecolor=SELECTED_SPECIES_FILL_MAP[color], alpha=0.28, linewidth=0.0, zorder=2)
        ax.plot(x_values, y_values, color=color, linewidth=2.4, linestyle='-', zorder=5)
        species_handles.append(Line2D([0], [0], color=color, linewidth=3.0, linestyle='-', label=species_name))
    legend_order = [2, 0, 1] if len(species_handles) == 3 else list(range(len(species_handles)))
    ordered_species_handles = [species_handles[i] for i in legend_order]
    legend = ax.legend(handles=ordered_species_handles, loc='upper right', frameon=False, fontsize=TEXT_SIZE - 3, handlelength=1.6, labelspacing=0.45, borderpad=0.55)
    legend.set_zorder(20)
    for text_item in legend.get_texts():
        text_item.set_fontstyle('italic')
        text_item.set_fontname('Times New Roman')
    configure_axis(ax, style=True, grid=True)
    ax.set_xlim(-0.032, 0.832)
    ax.set_xticks(np.linspace(0.0, 0.8, 5))
    ax.xaxis.set_major_formatter(FuncFormatter(fraction_tick_formatter))
    ax.set_ylim(-0.008, 0.208)
    ax.set_yticks([0.0, 0.05, 0.1, 0.15, 0.2])
    ax.yaxis.set_major_formatter(FuncFormatter(tick_formatter("two")))
    ax.set_xlabel('Fraction of nodes removed, $f$', fontsize=TEXT_SIZE)
    ax.set_ylabel('$S_2/N$', fontsize=TEXT_SIZE)
    ax.set_box_aspect(1.0)


def plot_panel_b(ax_bar, panel_b_stats: dict) -> None:
    """Draw precomputed category means, intervals, and significance tests."""
    habitat_summary = panel_b_stats['Habitat']['summary']
    oxygen_summary = panel_b_stats['Oxygen']['summary']
    habitat_categories = habitat_summary['category'].astype(str).tolist()
    oxygen_categories = oxygen_summary['category'].astype(str).tolist()
    n_habitat = len(habitat_summary)
    n_oxygen = len(oxygen_summary)
    gap = 0.9
    habitat_positions = np.arange(n_habitat, dtype=float)
    oxygen_positions = np.arange(n_oxygen, dtype=float) + n_habitat + gap
    all_positions = list(habitat_positions) + list(oxygen_positions)
    all_labels = habitat_categories + oxygen_categories
    side_pad = 0.45
    divider_x = None
    if all_positions:
        category_left = min(all_positions) - 0.5
        category_right = max(all_positions) + 0.5
        x_left = category_left - side_pad
        x_right = category_right + side_pad
        ax_bar.set_xlim(x_left, x_right)
        if n_habitat > 0 and n_oxygen > 0:
            habitat_group_right = habitat_positions[-1] + 0.5
            oxygen_group_left = oxygen_positions[0] - 0.5
            divider_x = (habitat_group_right + oxygen_group_left) / 2.0
            ax_bar.axvspan(x_left, divider_x, facecolor=HABITAT_BG, edgecolor='none', zorder=-4)
            ax_bar.axvspan(divider_x, x_right, facecolor=OXYGEN_BG, edgecolor='none', zorder=-4)
            ax_bar.axvline(divider_x, color='#D9D9D9', linewidth=0.8, linestyle='-', alpha=0.9, zorder=1)
        elif n_habitat > 0:
            ax_bar.axvspan(x_left, x_right, facecolor=HABITAT_BG, edgecolor='none', zorder=-4)
        elif n_oxygen > 0:
            ax_bar.axvspan(x_left, x_right, facecolor=OXYGEN_BG, edgecolor='none', zorder=-4)
    for positions, summary, face, edge in ((habitat_positions, habitat_summary, HABITAT_FACE, HABITAT_COLOR), (oxygen_positions, oxygen_summary, OXYGEN_FACE, OXYGEN_COLOR)):
        if len(summary):
            ax_bar.bar(positions, summary['mean'], yerr=np.vstack([summary['mean'] - summary['ci_low'], summary['ci_high'] - summary['mean']]), width=0.64, facecolor=face, edgecolor=edge, linewidth=1.8, error_kw={'ecolor': edge, 'zorder': 5, 'elinewidth': 1.6, 'capsize': 4, 'capthick': 1.6}, zorder=3)
    ax_bar.set_xticks(all_positions)
    ax_bar.set_xticklabels(all_labels, fontsize=TEXT_SIZE, rotation=32, ha='right', rotation_mode='anchor')
    configure_axis(ax_bar, y_ticks=PANEL_B_METRIC_TICKS, y_formatter=tick_formatter("two"))
    ax_bar.set_ylabel(METRIC_LABEL, fontsize=TEXT_SIZE)
    ax_bar.grid(True, axis='y', color='#D9D9D9', linewidth=0.8, alpha=0.9, zorder=0)
    ax_bar.tick_params(axis='both', which='major', labelsize=TEXT_SIZE, length=5, width=0.9, direction='out')
    for spine_name in ['left', 'right', 'top', 'bottom']:
        ax_bar.spines[spine_name].set_visible(True)
        ax_bar.spines[spine_name].set_color('#444444')
        ax_bar.spines[spine_name].set_linewidth(0.9)
    if all_positions:
        x_left, x_right = ax_bar.get_xlim()
        group_header_transform = ax_bar.get_xaxis_transform()
        for present, center, title in ((n_habitat > 0, (x_left + (divider_x if divider_x is not None else x_right)) / 2.0, 'Ecological habitat'), (n_oxygen > 0, ((divider_x if divider_x is not None else x_left) + x_right) / 2.0, 'Oxygen requirement')):
            if present:
                ax_bar.text(center, 0.982, title, transform=group_header_transform, ha='center', va='top', fontsize=TEXT_SIZE - 3, fontname='Times New Roman', color='black', zorder=10)
    def compact_bracket_y(summary_df: pd.DataFrame) -> float:
        if summary_df.empty:
            return float(PANEL_B_METRIC_TICKS[-1])
        bar_tops = pd.to_numeric(summary_df['ci_high'], errors='coerce')
        return float(bar_tops.max()) + 0.00075
    for category, positions, summary, offset in (
        ('Habitat', habitat_positions, habitat_summary, 0.0),
        ('Oxygen', oxygen_positions, oxygen_summary, 0.0028),
    ):
        p_value = panel_b_stats[category]['p_value']
        if p_value is not None and len(positions) >= 2:
            add_vertical_category_bracket(
                ax_bar, float(positions[0]) - 0.32,
                float(positions[-1]) + 0.32,
                compact_bracket_y(summary) + offset,
                format_p_value(p_value), cap=0.0014,
            )
    if all_positions:
        trend_transform = ax_bar.get_xaxis_transform()
        def draw_group_triangle(positions: np.ndarray, color: str, description: str, y_offset: float=0.0) -> None:
            if len(positions) == 0:
                return
            if len(positions) == 1:
                x_left = float(positions[0] - 0.28)
                x_right = float(positions[0] + 0.28)
            else:
                x_left = float(positions[0] + 0.05)
                x_right = float(positions[-1] - 0.05)
            y_bottom = 0.755 + y_offset
            y_top = 0.83 + y_offset
            triangle = Polygon([(x_left, y_bottom), (x_left, y_top), (x_right, y_bottom)], closed=True, facecolor=HABITAT_FACE if color == HABITAT_COLOR else OXYGEN_FACE, edgecolor=color, linewidth=1.7, transform=trend_transform, zorder=7)
            ax_bar.add_patch(triangle)
            ax_bar.text(x_left, y_top + 0.012, 'High', transform=trend_transform, color=color, fontsize=TEXT_SIZE, ha='left', va='bottom', zorder=8)
            ax_bar.text(x_right, y_bottom + 0.012, 'Low', transform=trend_transform, color=color, fontsize=TEXT_SIZE, ha='right', va='bottom', zorder=8)
            ax_bar.text((x_left + x_right) / 2.0, 0.71 + y_offset, description, transform=trend_transform, color=color, fontsize=TEXT_SIZE, rotation=0, ha='center', va='center', zorder=8)
        draw_group_triangle(habitat_positions, HABITAT_COLOR, 'Complexity of bacterial lifestyle')
        oxygen_trend_offset = 0.0
        if not habitat_summary.empty and (not oxygen_summary.empty):
            y_min, y_max = ax_bar.get_ylim()
            oxygen_trend_offset = (compact_bracket_y(oxygen_summary) + 0.0028 - compact_bracket_y(habitat_summary)) / (y_max - y_min)
        draw_group_triangle(oxygen_positions, OXYGEN_COLOR, 'Oxygen dependence', y_offset=oxygen_trend_offset)

def plot_scatter_panel(ax, df: pd.DataFrame, *, y_col: str, y_label: str, selected_color_map: dict[str, str], regression_stats=None) -> None:
    # Panel data is already cleaned once by prepare_panel_data().
    plot_data = df
    selected_ids = set((str(value) for value in selected_color_map))
    ordinary = plot_data[~plot_data['Species_ID'].astype(str).isin(selected_ids)]
    highlighted = plot_data[plot_data['Species_ID'].astype(str).isin(selected_ids)]
    if not ordinary.empty:
        ax.scatter(ordinary[METRIC_COL], ordinary[y_col], s=ordinary[MARKER_SIZE_COL].to_numpy(dtype=float), color=SCATTER_GRAY, alpha=0.68, edgecolor='white', linewidth=0.35, zorder=3)
    for _, row in highlighted.iterrows():
        species_id = str(row['Species_ID'])
        ax.scatter(float(row[METRIC_COL]), float(row[y_col]), s=float(row[MARKER_SIZE_COL]), color=selected_color_map[species_id], alpha=1.0, edgecolor='none', linewidth=0.0, zorder=6)
    stats = regression_stats
    if stats is not None:
        stats_text = f"$R^2={stats['r_squared']:.3f}$" + '\n' + format_p_value(stats['p_value'])
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, ha='left', va='top', fontsize=TEXT_SIZE - 3, fontname='Times New Roman')
    ax.set_xlabel(METRIC_LABEL, fontsize=TEXT_SIZE)
    ax.set_ylabel(y_label, fontsize=TEXT_SIZE)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.set_box_aspect(1.0)

def apply_reference_layout(fig, ax_a, ax_b, ax_c, ax_d, ax_e):
    """Preserve the original pixel-aware panel geometry and lettering."""
    width, height = fig.get_size_inches()
    occupied_width = 3 * PANEL_WIDTH_IN + 2 * PANEL_GAP_IN
    left = (18.1 - occupied_width) / 2
    bottom = BOTTOM_MARGIN_IN
    top_bottom = bottom + PANEL_WIDTH_IN + ROW_GAP_IN
    for ax, column, y in ((ax_a, 0, top_bottom), (ax_c, 0, bottom), (ax_d, 1, bottom), (ax_e, 2, bottom)):
        ax.set_position([(left + column * (PANEL_WIDTH_IN + PANEL_GAP_IN)) / width, y / height, PANEL_WIDTH_IN / width, PANEL_WIDTH_IN / height])
        ax.set_box_aspect(1)
    ax_b.set_position([(left + PANEL_WIDTH_IN + PANEL_GAP_IN) / width, top_bottom / height, (2 * PANEL_WIDTH_IN + PANEL_GAP_IN) / width, PANEL_WIDTH_IN / height])
    ax_b.set_box_aspect(None)
    ax_b.set_aspect('auto')
    from matplotlib.text import Text
    for label in fig.findobj(match=Text):
        label.set_fontname('Times New Roman')
        label.set_color('black')
        label.set_fontsize(TEXT_SIZE)
        if label.get_text() in {'Low adaptive fragility', 'High adaptive fragility', 'High', 'Low', 'Complexity of bacterial lifestyle', 'Oxygen dependence'}:
            label.set_fontsize(22)
    legend = ax_a.get_legend()
    if legend is not None:
        for label in legend.get_texts():
            label.set_fontsize(22)
    ax_a.yaxis.set_major_formatter(FuncFormatter(lambda value, position: '0' if np.isclose(value, 0.0) else f'{value:.2f}'))
    for ax, label in ((ax_a, 'A'), (ax_c, 'C'), (ax_d, 'D'), (ax_e, 'E')):
        ax.text(0.98, 0.98, label, transform=ax.transAxes, ha='right', va='top', fontsize=TEXT_SIZE, fontname='Times New Roman', fontweight='bold', color='black', zorder=30)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    def visible_gap(left_ax, right_ax):
        visible_ticks = [tick for tick in left_ax.xaxis.get_major_ticks() if left_ax.get_xlim()[0] <= tick.get_loc() <= left_ax.get_xlim()[1] and tick.tick1line.get_visible()]
        last_tick = max(visible_ticks, key=lambda tick: tick.get_loc(), default=None)
        tick_right = (last_tick.tick1line.get_window_extent(renderer).x1
                      if last_tick is not None else left_ax.bbox.x1)
        title_left = right_ax.yaxis.label.get_window_extent(renderer).x0
        return title_left - tick_right
    def shift_axis(ax, dx_pixels):
        position = ax.get_position()
        ax.set_position([position.x0 + dx_pixels / fig.bbox.width, position.y0, position.width, position.height])
    reference_gap = visible_gap(ax_d, ax_e)
    delta = visible_gap(ax_c, ax_d) - reference_gap
    shift_axis(ax_c, delta)
    shift_axis(ax_a, delta)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    shift_axis(ax_b, reference_gap - visible_gap(ax_a, ax_b))
    e_pos, b_pos = (ax_e.get_position(), ax_b.get_position())
    letter_x = e_pos.x0 + 0.98 * e_pos.width
    letter_y = b_pos.y0 + 0.98 * b_pos.height
    fig.text(letter_x, letter_y, 'B', transform=fig.transFigure, ha='right', va='top', fontsize=TEXT_SIZE, fontname='Times New Roman', fontweight='bold', color='black', zorder=30)

def plot_figure(curves, selected, panel_data, panel_regressions, panel_b_stats):
    color_map = dict(zip(selected['Species_ID'].astype(str), SELECTED_SPECIES_COLORS))
    fig = plt.figure(figsize=FIGSIZE, facecolor='white')
    ax_a, ax_b, ax_c, ax_d, ax_e = [fig.add_axes([0.1, 0.1, 0.1, 0.1]) for _ in range(5)]
    plot_panel_a(ax_a, selected, curves, color_map)
    plot_panel_b(ax_b, panel_b_stats)
    for letter, ax in zip('CDE', (ax_c, ax_d, ax_e)):
        column, label, xticks, yticks, yformat = PANEL_FEATURES[letter]
        subset = panel_data[letter]
        stats = panel_regressions[letter]
        plot_scatter_panel(ax, subset, y_col=column, y_label=label, selected_color_map=color_map, regression_stats=stats)
        configure_axis(ax, x_ticks=xticks, y_ticks=yticks, x_formatter=tick_formatter("two"), y_formatter=tick_formatter(yformat), style=True, grid=True)
    for ax in (ax_a, ax_c, ax_d, ax_e):
        apply_figure3_panel_geometry(ax)
    apply_reference_layout(fig, ax_a, ax_b, ax_c, ax_d, ax_e)
    target = BASE_DIR / 'seamless_figures' / 'PPI.svg'
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, format='svg', bbox_inches='tight', transparent=True)
    if SHOW_FIGURE:
        try:
            from IPython.display import display
            display(fig)
        except ImportError:
            plt.show()
    plt.close(fig)
    return target


def report_panel_species_sets(panel_data: dict, selected: pd.DataFrame) -> None:
    """Report plotted species counts and their union (not intersection)."""
    def ids(frame):
        return set(frame['Species_ID'].dropna().astype(str))
    habitat, oxygen = ids(panel_data['Habitat']), ids(panel_data['Oxygen'])
    sets = {
        'A': ids(selected),
        'B': habitat | oxygen,
        'C': ids(panel_data['C']),
        'D': ids(panel_data['D']),
        'E': ids(panel_data['E']),
    }
    entries = [
        ('Panel A selected species', len(sets['A'])),
        ('Panel B - Ecological habitat', len(habitat)),
        ('Panel B - Oxygen requirement', len(oxygen)),
        ('Panel C - Fraction of regulatory genes', len(sets['C'])),
        ('Panel D - Environmental scope index', len(sets['D'])),
        ('Panel E - Co-habitation index', len(sets['E'])),
        ('Panels A-E union species', len(set.union(*sets.values()))),
    ]
    print('========== Panel sample sizes ==========')
    for label, count in entries:
        print(f'{label:<40}: {count}')

# ============================================================
# 5. Main
# ============================================================
def main():
    configure_matplotlib()
    for path in (SPECIES_FILE, REGULATORY_FILE, ENVIRONMENT_FILE, RESULTS_DIR):
        if not path.exists():
            raise FileNotFoundError(f'Required path not found: {path}')
    species = pd.read_csv(SPECIES_FILE, sep='\t')
    _, mapping, names = build_species_mapping(species)
    regulatory = build_regulatory_features(read_first_nonempty_excel_sheet(REGULATORY_FILE), mapping)
    environment = build_environment_features(read_first_nonempty_excel_sheet(ENVIRONMENT_FILE), mapping)
    features = (names.merge(regulatory[['Species_ID', 'Fraction_regulatory_genes']],
                            on='Species_ID', how='left')
                .merge(environment[['Species_ID', 'Co_habitation_index',
                                    'Environmental_scope_index', 'Habitat', 'Oxygen']],
                       on='Species_ID', how='left'))
    result_folders = find_ppi_result_folders(RESULTS_DIR)
    ppi_species = set(features['Species_ID'].dropna().astype(str)) & set(result_folders)
    metrics, curves = load_protocol_results(result_folders, ppi_species)
    if metrics['seamless'].empty:
        raise RuntimeError('No valid SEAMLESS (m=4) results were found.')
    eligible_ids = panel_3_5_eligible_species_ids(names, regulatory, environment, result_folders)
    nodes = collect_ppi_network_node_counts(eligible_ids, result_folders)
    features = features.merge(
        nodes[['Species_ID', NETWORK_N_COL, MARKER_SIZE_COL]],
        on='Species_ID', how='left'
    )
    merged = {
        protocol: metrics[protocol].merge(features, on='Species_ID', how='left')
        for protocol in PROTOCOLS if not metrics[protocol].empty
    }
    selected = representative_species(merged['seamless'])
    panel_data = prepare_panel_data(merged['seamless'])
    report_panel_species_sets(panel_data, selected)
    panel_regressions = {
        panel: linear_regression_statistics(panel_data[panel], METRIC_COL, config[0], prepared=True)
        for panel, config in PANEL_FEATURES.items()
    }
    panel_b_stats = prepare_panel_b_statistics(panel_data)
    image = plot_figure(curves['seamless'], selected, panel_data, panel_regressions, panel_b_stats)
    print(f'Saved: {image}')
    models = compare_protocols(merged)
    print_regression_comparison(models)
    return (merged, {'seamless': selected}, nodes, models)

if __name__ == "__main__":
    main()
