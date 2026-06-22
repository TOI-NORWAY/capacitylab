"""Traffic Flow and Speed Prediction GUI (fixed feedforward inference mode)."""

import streamlit as st
import streamlit.components.v1 as components
import pydeck as pdk
import altair as alt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import json
import copy

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import DataLoader
from prediction_engine import PredictionEngine

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GUI_README_PATH = PROJECT_ROOT / 'README_GUI.md'

# Default capacity value for all links
DEFAULT_CAPACITY = 3600.0
FLOW_DISPLAY_SCALE = 10.0
FLOW_DISPLAY_LABEL = 'Flow (veh/hr)'
RELATIVE_PERCENT_NEUTRAL_THRESHOLD = 0.05
RELATIVE_SPEED_DELTA_NEUTRAL_THRESHOLD = 0.01
RELATIVE_FLOW_DELTA_NEUTRAL_THRESHOLD = 1.0
CAPACITY_MIN = 100
CAPACITY_MAX = 5000
CAPACITY_STEP = 50

BASE_SCENARIO_1000 = '1000 capacity all links'
BASE_SCENARIO_1800 = '1800 capacity on all links'
BASE_SCENARIO_3600 = '3600 capacity on all links'
BASE_SCENARIO_CUSTOM = 'Custom base scenario'
BASE_SCENARIO_VALUES = {
    BASE_SCENARIO_1000: 1000.0,
    BASE_SCENARIO_1800: 1800.0,
    BASE_SCENARIO_3600: 3600.0,
}
BASE_SCENARIO_OPTIONS = [
    BASE_SCENARIO_1000,
    BASE_SCENARIO_1800,
    BASE_SCENARIO_3600,
    BASE_SCENARIO_CUSTOM,
]

V3_GLOBALCAP_SOURCE = 'v3_globalcap'
FIXED_MODEL_CODE = 'FF'
FIXED_MODEL_NAME = 'Feedforward NN'
TABLE_SORT_SELECTED_FIRST = 'Selected link first'
TABLE_SORT_MODIFIED_FIRST = 'Modified first'


def build_uniform_capacities(link_ids, capacity):
    return {link_id: float(capacity) for link_id in link_ids}


def clear_prediction_caches(include_baseline=False):
    st.session_state.predictions_cache = {}
    st.session_state.predictions_cache_flowspeed = {}
    st.session_state.local_sanity_cache = {}
    if include_baseline:
        st.session_state.baseline_predictions = {}
        st.session_state.baseline_speed_predictions = {}


def reset_capacity_table_generation():
    st.session_state.capacity_table_generation = st.session_state.get('capacity_table_generation', 0) + 1


def reset_to_uniform_base(loader, base_capacity):
    st.session_state.base_capacities = build_uniform_capacities(loader.unique_link_ids, base_capacity)
    st.session_state.capacities = copy.deepcopy(st.session_state.base_capacities)
    clear_prediction_caches(include_baseline=True)
    reset_capacity_table_generation()


def get_query_value(query_params, key):
    value = query_params.get(key)
    if isinstance(value, list):
        return value[0] if value else None
    return value


def clear_query_keys(*keys):
    for key in keys:
        if key in st.query_params:
            del st.query_params[key]


def resolve_link_id(loader, raw_link_id):
    if raw_link_id is None:
        return None
    raw_str = str(raw_link_id)
    return next((link_id for link_id in loader.unique_link_ids if str(link_id) == raw_str), None)


def sync_state_to_loaded_source(source_context, loader):
    """Reset source-dependent session state when the active output folder changes."""
    if st.session_state.get('active_model_context') == source_context:
        return

    st.session_state.active_model_context = source_context
    st.session_state.base_scenario = BASE_SCENARIO_3600
    st.session_state.base_capacities = build_uniform_capacities(loader.unique_link_ids, DEFAULT_CAPACITY)
    st.session_state.capacities = copy.deepcopy(st.session_state.base_capacities)
    st.session_state.edited_capacities = copy.deepcopy(st.session_state.capacities)
    st.session_state.predictions_cache = {}
    st.session_state.predictions_cache_flowspeed = {}
    st.session_state.baseline_predictions = {}
    st.session_state.baseline_speed_predictions = {}
    st.session_state.local_sanity_cache = {}
    st.session_state.selected_link = loader.unique_link_ids[0]
    st.session_state.clicked_link = None
    st.session_state.display_mode = 'speed'
    st.session_state.value_mode = 'relative_percent'
    st.session_state.selected_model_code = FIXED_MODEL_CODE
    st.session_state.local_sensitivity_hour = int(st.session_state.get('selected_hour', 12))
    st.session_state.blink_link = None
    st.session_state.blink_generation = 0
    st.session_state.locate_generation = 0
    st.session_state.table_sort_mode = TABLE_SORT_SELECTED_FIRST
    st.session_state.table_sort_ascending = True
    st.session_state.capacity_table_generation = st.session_state.get('capacity_table_generation', 0) + 1


def create_node_mapping(loader):
    """Create a mapping from original node IDs to simple 1-N numbering."""
    # Collect all unique nodes from the network
    all_nodes = set()
    for link_id in loader.unique_link_ids:
        link_data = loader.links_info[link_id]
        all_nodes.add(link_data['from'])
        all_nodes.add(link_data['to'])
    
    # Sort nodes to get consistent numbering
    sorted_nodes = sorted(all_nodes)
    
    # Create mappings: original_id -> simple_number and reverse
    node_to_number = {node_id: idx + 1 for idx, node_id in enumerate(sorted_nodes)}
    number_to_node = {idx + 1: node_id for idx, node_id in enumerate(sorted_nodes)}
    
    return node_to_number, number_to_node


def value_to_color(value, min_val, max_val, color_mode='flow'):
    """Convert value to RGB color using a red-blue-green diverging scale.
    
    Args:
        value: The value to convert
        min_val: Minimum value for normalization
        max_val: Maximum value for normalization  
        color_mode: kept for backwards compatibility
    
    Returns:
        list: [R, G, B, A] color values
    """
    return value_to_color_sequential(value, min_val, max_val)


def value_to_color_sequential(value, min_val, max_val):
    """Map value to red (low) -> yellow (mid) -> green (high)."""
    if max_val == min_val:
        normalized = 0.5
    else:
        normalized = (value - min_val) / (max_val - min_val)

    normalized = max(0.0, min(1.0, normalized))
    if normalized < 0.5:
        r = 255
        g = int(255 * (normalized * 2))
    else:
        r = int(255 * (1 - (normalized - 0.5) * 2))
        g = 255
    b = 0
    return [r, g, b, 200]


def value_to_color_diverging(value, min_val, max_val, center_val=None):
    """Map value to red (low) -> blue (center) -> green (high)."""
    if max_val == min_val:
        return [0, 120, 255, 200]

    if center_val is None:
        center_val = (min_val + max_val) / 2.0

    center_val = max(min_val, min(max_val, center_val))

    def blend(c1, c2, t):
        t = max(0.0, min(1.0, t))
        return [
            int(c1[0] + (c2[0] - c1[0]) * t),
            int(c1[1] + (c2[1] - c1[1]) * t),
            int(c1[2] + (c2[2] - c1[2]) * t),
            200,
        ]

    red = [220, 20, 60]
    blue = [0, 120, 255]
    green = [20, 180, 20]

    if value <= center_val:
        if center_val == min_val:
            t = 1.0
        else:
            t = (value - min_val) / (center_val - min_val)
        return blend(red, blue, t)

    if max_val == center_val:
        t = 1.0
    else:
        t = (value - center_val) / (max_val - center_val)
    return blend(blue, green, t)


def flow_to_color(flow, min_flow, max_flow):
    """Convert flow value to RGB color (red -> yellow -> green)."""
    return value_to_color(flow, min_flow, max_flow, color_mode='flow')


def compute_relative_percent(current_val, baseline_val):
    """Return percent change relative to baseline, handling near-zero baselines."""
    delta = current_val - baseline_val
    if abs(baseline_val) < 1e-6:
        if abs(delta) < 1e-6:
            return 0.0
        return float(np.sign(delta) * 100.0)
    return float(100.0 * delta / abs(baseline_val))


def get_relative_neutral_threshold(display_mode, value_mode):
    if value_mode == 'relative_percent':
        return RELATIVE_PERCENT_NEUTRAL_THRESHOLD
    if value_mode in ('relative', 'relative_delta'):
        if display_mode == 'speed':
            return RELATIVE_SPEED_DELTA_NEUTRAL_THRESHOLD
        return RELATIVE_FLOW_DELTA_NEUTRAL_THRESHOLD
    return 0.0


def apply_relative_neutral_deadband(value, display_mode, value_mode):
    threshold = get_relative_neutral_threshold(display_mode, value_mode)
    if threshold <= 0.0:
        return float(value)
    value = float(value)
    if abs(value) < threshold:
        return 0.0
    return value


def get_sensitive_color_limits(values, center_val=None):
    """Return robust color bounds so outliers do not wash out map contrast."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return -1.0, 1.0

    if center_val is not None:
        deviations = np.abs(arr - center_val)
        nonzero_deviations = deviations[deviations > 1e-9]
        if nonzero_deviations.size == 0:
            limit = 1.0
        else:
            limit = float(np.percentile(nonzero_deviations, 90))
            if limit <= 1e-9:
                limit = float(nonzero_deviations.max())
        return center_val - limit, center_val + limit

    if arr.size >= 4:
        min_val = float(np.percentile(arr, 5))
        max_val = float(np.percentile(arr, 95))
    else:
        min_val = float(arr.min())
        max_val = float(arr.max())

    if max_val <= min_val:
        pad = max(abs(min_val) * 0.05, 1.0)
        min_val -= pad
        max_val += pad

    return min_val, max_val


def build_directional_neighbor_maps(loader):
    """Build downstream/upstream link maps using directed topology.

    For center link A->B, downstream links are links entering A and upstream links
    are links leaving B. The direct reverse B->A is excluded from both sides.
    """
    node_to_incoming_links = {}
    node_to_outgoing_links = {}

    for lid in loader.unique_link_ids:
        link_data = loader.links_info[lid]
        from_node = link_data['from']
        to_node = link_data['to']
        node_to_outgoing_links.setdefault(from_node, []).append(lid)
        node_to_incoming_links.setdefault(to_node, []).append(lid)

    upstream_map = {}
    downstream_map = {}
    for lid in loader.unique_link_ids:
        link_data = loader.links_info[lid]
        from_node = link_data['from']
        to_node = link_data['to']

        def is_reverse_link(candidate_id):
            candidate = loader.links_info[candidate_id]
            return candidate['from'] == to_node and candidate['to'] == from_node

        upstream_map[lid] = sorted(
            [
                cand for cand in node_to_outgoing_links.get(to_node, [])
                if cand in loader.link_to_idx and cand != lid and not is_reverse_link(cand)
            ],
            key=lambda x: loader.link_to_idx[x],
        )
        downstream_map[lid] = sorted(
            [
                cand for cand in node_to_incoming_links.get(from_node, [])
                if cand in loader.link_to_idx and cand != lid and not is_reverse_link(cand)
            ],
            key=lambda x: loader.link_to_idx[x],
        )

    return upstream_map, downstream_map


def get_parallel_links(loader, center_link_id, upstream_links, downstream_links):
    """Return existing links that bypass the center through nearby side nodes."""
    center_link = loader.links_info[center_link_id]
    center_from = center_link['from']
    center_to = center_link['to']

    source_nodes = {center_from}
    source_nodes.update(loader.links_info[lid]['from'] for lid in downstream_links)
    target_nodes = {center_to}
    target_nodes.update(loader.links_info[lid]['to'] for lid in upstream_links)

    excluded = set(upstream_links) | set(downstream_links) | {center_link_id}

    def is_center_reverse(candidate_id):
        candidate = loader.links_info[candidate_id]
        return candidate['from'] == center_to and candidate['to'] == center_from

    return sorted(
        [
            candidate_id for candidate_id in loader.unique_link_ids
            if candidate_id not in excluded
            and candidate_id in loader.link_to_idx
            and not is_center_reverse(candidate_id)
            and loader.links_info[candidate_id]['from'] in source_nodes
            and loader.links_info[candidate_id]['to'] in target_nodes
        ],
        key=lambda x: loader.link_to_idx[x],
    )


def get_local_subnetwork_links(
    loader,
    center_link_id,
    max_upstream=None,
    max_downstream=None,
    include_parallel_links=False,
):
    """Return upstream/center/downstream/parallel link IDs around selected center link."""
    upstream_map, downstream_map = build_directional_neighbor_maps(loader)
    upstream_links = upstream_map.get(center_link_id, [])
    downstream_links = downstream_map.get(center_link_id, [])

    if max_upstream is not None:
        upstream_links = upstream_links[:max_upstream]
    if max_downstream is not None:
        downstream_links = downstream_links[:max_downstream]

    parallel_links = []
    if include_parallel_links:
        parallel_links = get_parallel_links(loader, center_link_id, upstream_links, downstream_links)

    tracked_links = downstream_links + [center_link_id] + parallel_links + upstream_links
    return upstream_links, downstream_links, parallel_links, tracked_links


def compute_local_capacity_sweep(
    loader,
    model_code,
    base_capacities,
    center_link_id,
    hour,
    tracked_links,
    speed_available,
):
    """Sweep center-link capacity and collect local flow/speed responses."""
    max_cap = max(2000, int(np.ceil(max(base_capacities.values()) / 100.0) * 100.0))
    min_cap = 100
    sweep_values = np.arange(max_cap, min_cap - 1, -100, dtype=np.float32)

    pred_engine = PredictionEngine(loader)
    records = []

    for center_cap in sweep_values:
        trial_capacities = dict(base_capacities)
        trial_capacities[center_link_id] = float(center_cap)
        pred_engine.capacities = trial_capacities

        flow_predictions = pred_engine.predict_all(model_code)
        speed_predictions = None
        if speed_available:
            _, speed_predictions = pred_engine.predict_all_flowspeed(model_code)

        for lid in tracked_links:
            flow_val = flow_predictions[lid][hour]
            speed_val = speed_predictions[lid][hour] if speed_predictions else np.nan
            records.append(
                {
                    'center_capacity': float(center_cap),
                    'link_id': lid,
                    'flow': float(flow_val),
                    'speed': float(speed_val) if not np.isnan(speed_val) else np.nan,
                }
            )

    return pd.DataFrame(records), sweep_values


def create_local_dependency_figure(
    loader,
    link_coords_df,
    sweep_df,
    center_link_id,
    upstream_links,
    downstream_links,
    parallel_links,
    hour,
    model_label,
    current_center_capacity,
    speed_available,
):
    """Render local subnetwork sketch and flow/speed response curves."""
    node_to_number, _ = create_node_mapping(loader)
    link_name_lookup = dict(zip(link_coords_df['link_id'], link_coords_df['link_name']))

    plot_order = downstream_links + [center_link_id] + parallel_links + upstream_links
    cmap = plt.cm.get_cmap('tab10', max(len(plot_order), 1))
    link_colors = {lid: cmap(i) for i, lid in enumerate(plot_order)}

    fig = plt.figure(figsize=(22, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.6, 1.6], wspace=0.30)
    ax_graph = fig.add_subplot(gs[0, 0])
    ax_flow = fig.add_subplot(gs[0, 1])
    ax_speed = fig.add_subplot(gs[0, 2])

    center_from = loader.links_info[center_link_id]['from']
    center_to = loader.links_info[center_link_id]['to']

    node_positions_by_id = {
        center_from: {
            'node_id': center_from,
            'position': (0.38, 0.50),
            'color': '#fdd0a2',
        },
        center_to: {
            'node_id': center_to,
            'position': (0.72, 0.50),
            'color': '#fdae6b',
        },
    }

    def set_node_position(node_id, position, color):
        if node_id not in node_positions_by_id:
            node_positions_by_id[node_id] = {
                'node_id': node_id,
                'position': position,
                'color': color,
            }

    if upstream_links:
        up_y = np.linspace(0.75, 0.25, len(upstream_links))
        for idx, lid in enumerate(upstream_links):
            set_node_position(loader.links_info[lid]['to'], (1.04, float(up_y[idx])), '#9ecae1')

    if downstream_links:
        down_y = np.linspace(0.75, 0.25, len(downstream_links))
        for idx, lid in enumerate(downstream_links):
            set_node_position(loader.links_info[lid]['from'], (0.06, float(down_y[idx])), '#c7e9c0')

    parallel_from_nodes = [loader.links_info[lid]['from'] for lid in parallel_links]
    parallel_to_nodes = [loader.links_info[lid]['to'] for lid in parallel_links]
    missing_from_nodes = sorted(
        {node for node in parallel_from_nodes if node not in node_positions_by_id},
        key=lambda node: node_to_number.get(node, 10**9),
    )
    missing_to_nodes = sorted(
        {node for node in parallel_to_nodes if node not in node_positions_by_id},
        key=lambda node: node_to_number.get(node, 10**9),
    )
    if missing_from_nodes:
        y_values = np.linspace(0.82, 0.18, len(missing_from_nodes))
        for y_pos, node_id in zip(y_values, missing_from_nodes):
            set_node_position(node_id, (0.34, float(y_pos)), '#fdd0d0')
    if missing_to_nodes:
        y_values = np.linspace(0.82, 0.18, len(missing_to_nodes))
        for y_pos, node_id in zip(y_values, missing_to_nodes):
            set_node_position(node_id, (0.76, float(y_pos)), '#fdd0d0')

    role_positions = {
        f'node_{node_id}': node['position']
        for node_id, node in node_positions_by_id.items()
    }

    def node_role(node_id):
        return f'node_{node_id}'

    def draw_link(ax, from_role, to_role, link_id, y_shift=0.0, dashed=False):
        x1, y1 = role_positions[from_role]
        x2, y2 = role_positions[to_role]
        color = link_colors[link_id]
        link_name = link_name_lookup.get(link_id, str(link_id))
        ax.annotate(
            '',
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle='-|>',
                lw=1.8,
                color=color,
                shrinkA=12,
                shrinkB=12,
                linestyle='--' if dashed else '-',
            ),
        )
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + y_shift, link_name, fontsize=8, color=color, ha='center', va='center')

    for idx, lid in enumerate(downstream_links):
        draw_link(
            ax_graph,
            node_role(loader.links_info[lid]['from']),
            node_role(center_from),
            lid,
            y_shift=0.03 if idx % 2 == 0 else -0.03,
        )

    draw_link(ax_graph, node_role(center_from), node_role(center_to), center_link_id, y_shift=0.03)

    for idx, lid in enumerate(parallel_links):
        draw_link(
            ax_graph,
            node_role(loader.links_info[lid]['from']),
            node_role(loader.links_info[lid]['to']),
            lid,
            y_shift=0.03 if idx % 2 == 0 else -0.03,
            dashed=True,
        )

    for idx, lid in enumerate(upstream_links):
        draw_link(
            ax_graph,
            node_role(center_to),
            node_role(loader.links_info[lid]['to']),
            lid,
            y_shift=0.03 if idx % 2 == 0 else -0.03,
        )

    for node in node_positions_by_id.values():
        node_id = node['node_id']
        x, y = node['position']
        node_color = node['color']
        node_label = str(node_to_number.get(node_id, node_id))
        ax_graph.scatter(x, y, s=260, color=node_color, edgecolor='black', zorder=3)
        ax_graph.text(x, y - 0.06, node_label, fontsize=8, ha='center', va='top')

    column_labels = [
        ('Downstream', 0.06),
        ('Link start', 0.38),
        ('Parallel', 0.55),
        ('Link end', 0.72),
        ('Upstream', 1.04),
    ]
    for label, x_pos in column_labels:
        ax_graph.text(x_pos, 0.94, label, fontsize=8, ha='center', va='center', color='#555555')

    ax_graph.set_title('Local directional subnetwork', fontsize=13, fontweight='bold')
    ax_graph.set_xlim(-0.05, 1.13)
    ax_graph.set_ylim(-0.02, 0.98)
    ax_graph.axis('off')

    for lid in plot_order:
        link_name = link_name_lookup.get(lid, str(lid))
        if lid == center_link_id:
            role = 'Center'
        elif lid in parallel_links:
            role = 'Parallel'
        elif lid in upstream_links:
            role = 'Upstream'
        else:
            role = 'Downstream'
        line_df = sweep_df[sweep_df['link_id'] == lid].sort_values('center_capacity')
        ax_flow.plot(
            line_df['center_capacity'],
            line_df['flow'],
            marker='o',
            linewidth=2,
            markersize=4,
            color=link_colors[lid],
            label=f'{role} | {link_name}',
        )
    ax_flow.axvline(current_center_capacity, color='black', linestyle='--', linewidth=1.2, alpha=0.7)
    ax_flow.set_title(f'Flow response @ {hour}:00', fontsize=13, fontweight='bold')
    ax_flow.set_xlabel('Center-link capacity')
    ax_flow.set_ylabel('Predicted flow (veh/hr)')
    ax_flow.grid(True, alpha=0.3)
    ax_flow.invert_xaxis()

    if speed_available and sweep_df['speed'].notna().any():
        for lid in plot_order:
            link_name = link_name_lookup.get(lid, str(lid))
            if lid == center_link_id:
                role = 'Center'
            elif lid in parallel_links:
                role = 'Parallel'
            elif lid in upstream_links:
                role = 'Upstream'
            else:
                role = 'Downstream'
            line_df = sweep_df[sweep_df['link_id'] == lid].sort_values('center_capacity')
            ax_speed.plot(
                line_df['center_capacity'],
                line_df['speed'],
                marker='o',
                linewidth=2,
                markersize=4,
                color=link_colors[lid],
                label=f'{role} | {link_name}',
            )
        ax_speed.axvline(current_center_capacity, color='black', linestyle='--', linewidth=1.2, alpha=0.7)
        ax_speed.set_title(f'Speed response @ {hour}:00', fontsize=13, fontweight='bold')
        ax_speed.set_xlabel('Center-link capacity')
        ax_speed.set_ylabel('Predicted speed (m/s)')
        ax_speed.grid(True, alpha=0.3)
        ax_speed.invert_xaxis()
        ax_speed.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=8)
    else:
        ax_speed.text(0.5, 0.5, 'Speed predictions not available\nfor the active source/model.', ha='center', va='center', fontsize=11)
        ax_speed.set_title('Speed response', fontsize=13, fontweight='bold')
        ax_speed.axis('off')

    center_name = link_name_lookup.get(center_link_id, str(center_link_id))
    plt.suptitle(
        f'Local capacity sanity view | Model: {model_label} | Center link: {center_name} | Hour: {hour}:00',
        fontsize=14,
        fontweight='bold',
        y=1.02,
    )
    plt.tight_layout()
    return fig


@st.cache_resource
def load_data_and_models():
    """Load the fixed v3 global-capacity feedforward artifacts."""
    base_path = PROJECT_ROOT
    loader = DataLoader(base_path)
    loader.load_v3_globalcap_models()
    return loader


@st.cache_data
def get_link_coords_df(_loader, _version=5):
    """Get link coordinates as DataFrame for pydeck (cached).
    
    _version parameter forces cache invalidation when code changes.
    """
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:32633", "EPSG:4326", always_xy=True)
    
    # Create node mapping for simple naming
    node_to_number, _ = create_node_mapping(_loader)
    
    records = []
    for link_id in _loader.unique_link_ids:
        link_data = _loader.links_info[link_id]
        from_node = link_data['from']
        to_node = link_data['to']
        
        # Get simple node numbers
        from_num = node_to_number[from_node]
        to_num = node_to_number[to_node]
        
        from_x, from_y = _loader.nodes[from_node]['x'], _loader.nodes[from_node]['y']
        to_x, to_y = _loader.nodes[to_node]['x'], _loader.nodes[to_node]['y']
        
        from_lon, from_lat = transformer.transform(from_x, from_y)
        to_lon, to_lat = transformer.transform(to_x, to_y)
        
        # Calculate link direction vector
        dx_lon = to_lon - from_lon
        dx_lat = to_lat - from_lat
        
        # Arrow position: offset along link to avoid overlap with reverse direction
        # Use 35% along link (closer to from_node) to separate from reverse link's arrow
        arrow_lon = from_lon + 0.35 * dx_lon
        arrow_lat = from_lat + 0.35 * dx_lat
        
        # Also offset perpendicular to the link direction (to the right side)
        # This separates arrows on parallel links going in opposite directions
        link_length = np.sqrt(dx_lon**2 + dx_lat**2)
        if link_length > 0:
            # Perpendicular unit vector (90 degrees clockwise = right side)
            perp_lon = dx_lat / link_length
            perp_lat = -dx_lon / link_length
            # Offset by a small amount perpendicular to the link
            offset_amount = 0.00015  # Adjust this for more/less separation
            arrow_lon += perp_lon * offset_amount
            arrow_lat += perp_lat * offset_amount
        
        # Calculate angle for arrow direction (pointing from->to)
        # Note: atan2(dy, dx) where dy=lat diff, dx=lon diff
        angle_rad = np.arctan2(to_lat - from_lat, to_lon - from_lon)
        angle_deg = np.degrees(angle_rad)
        
        records.append({
            'link_id': link_id,
            'from_node': from_node,
            'to_node': to_node,
            'from_num': from_num,
            'to_num': to_num,
            'link_name': f"{from_num}-{to_num}",  # Simple link name like "3-7"
            'from_lat': from_lat,
            'from_lon': from_lon,
            'to_lat': to_lat,
            'to_lon': to_lon,
            'mid_lat': (from_lat + to_lat) / 2,
            'mid_lon': (from_lon + to_lon) / 2,
            'arrow_lat': arrow_lat,
            'arrow_lon': arrow_lon,
            'angle': angle_deg,
            'original_capacity': link_data['capacity'],
            'direction': f"{from_num}-{to_num}"  # Use simple numbers
        })
    
    return pd.DataFrame(records)


@st.cache_data
def get_node_coords_df(_loader, _version=1):
    """Get node coordinates for displaying labels on the map."""
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:32633", "EPSG:4326", always_xy=True)
    
    node_to_number, _ = create_node_mapping(_loader)
    
    records = []
    for node_id, num in node_to_number.items():
        x, y = _loader.nodes[node_id]['x'], _loader.nodes[node_id]['y']
        lon, lat = transformer.transform(x, y)
        records.append({
            'node_id': node_id,
            'node_num': num,
            'lat': lat,
            'lon': lon
        })
    
    return pd.DataFrame(records)


@st.cache_data
def load_gui_readme_text():
    """Load GUI documentation markdown content from the repository README."""
    if not GUI_README_PATH.exists():
        return "# GUI Documentation\n\nDocumentation file not found: `README_GUI.md`."
    return GUI_README_PATH.read_text(encoding='utf-8')


def create_map_data(
    link_coords_df,
    flows,
    speeds,
    baseline_flows,
    baseline_speeds,
    capacities,
    selected_link,
    min_val,
    max_val,
    display_mode='flow',
    value_mode='absolute',
):
    """Create DataFrame with colors for pydeck based on display mode.
    
    Args:
        link_coords_df: DataFrame with link coordinates
        flows: dict of {link_id: flow}
        speeds: dict of {link_id: speed} (can be None if not available)
        baseline_flows: dict of {link_id: baseline flow at selected hour}
        baseline_speeds: dict of {link_id: baseline speed at selected hour} (can be None)
        capacities: dict of {link_id: capacity}
        selected_link: currently selected link ID
        min_val: minimum value for color scaling
        max_val: maximum value for color scaling
        display_mode: 'flow' or 'speed'
        value_mode: 'absolute' or 'relative'
    """
    df = link_coords_df.copy()
    
    # Add flow, speed and color data
    df['flow'] = df['link_id'].map(flows)
    df['speed'] = df['link_id'].map(speeds) if speeds else 0
    df['baseline_flow'] = df['link_id'].map(baseline_flows) if baseline_flows else 0
    df['baseline_speed'] = df['link_id'].map(baseline_speeds) if baseline_speeds else 0
    df['capacity'] = df['link_id'].map(capacities)
    
    # Determine which values to use for coloring/display
    if display_mode == 'speed' and speeds:
        raw_values = df['speed']
        baseline_values = df['baseline_speed']
        metric_name = 'Speed'
    else:
        raw_values = df['flow']
        baseline_values = df['baseline_flow']
        metric_name = 'Flow'

    if value_mode in ('relative', 'relative_delta'):
        color_values = [
            apply_relative_neutral_deadband(
                float(cur) - float(base),
                display_mode,
                value_mode,
            )
            for cur, base in zip(raw_values, baseline_values)
        ]
        df['map_metric_label'] = f"{metric_name} Δ vs baseline"
        df['map_metric_unit'] = ''
    elif value_mode == 'relative_percent':
        color_values = [
            apply_relative_neutral_deadband(
                compute_relative_percent(float(cur), float(base)),
                display_mode,
                value_mode,
            )
            for cur, base in zip(raw_values, baseline_values)
        ]
        df['map_metric_label'] = f"{metric_name} % vs baseline"
        df['map_metric_unit'] = '%'
    else:
        color_values = raw_values
        df['map_metric_label'] = f"{metric_name} prediction"
        if display_mode == 'speed' and speeds:
            df['map_metric_unit'] = 'm/s'
        else:
            df['map_metric_unit'] = 'veh/hr'

    df['map_metric_value'] = color_values
    
    # Compute arrow color for every link (blue for selected, flow/speed color otherwise)
    arrow_colors = []
    for idx, row in df.iterrows():
        if row['link_id'] == selected_link:
            arrow_colors.append([0, 100, 255, 255])  # Blue for selected link
        else:
            val = color_values.iloc[idx] if hasattr(color_values, 'iloc') else color_values[idx]
            if value_mode in ('relative', 'relative_delta', 'relative_percent'):
                arrow_colors.append(value_to_color_diverging(val, min_val, max_val, center_val=0.0))
            else:
                arrow_colors.append(value_to_color_sequential(val, min_val, max_val))
    
    df['arrow_color'] = arrow_colors
    
    return df


def create_custom_map_html(
    map_df,
    node_coords_df,
    center_lat,
    center_lon,
    selected_link,
    display_mode='flow',
    blink_link=None,
    blink_generation=0,
    network_bounds=None,
    locate_generation=0,
):
    """Create custom HTML/JS map with full-link arrow polygons coloured by flow or speed.
    
    Args:
        map_df: DataFrame with map data
        node_coords_df: DataFrame with node coordinates for labels
        center_lat: Map center latitude
        center_lon: Map center longitude
        selected_link: Currently selected link ID
        display_mode: 'flow' or 'speed' - affects tooltip display
        blink_link: Link ID to pulse briefly when selected from the table
        blink_generation: Monotonic token so repeat clicks retrigger the pulse
        network_bounds: Bounds dict used to fit the full network after Locate
        locate_generation: Monotonic token for Locate/fitting actions
    """
    
    # Prepare link data as JSON (arrow_color used for the full-link arrow polygons)
    links_data = []
    
    for _, row in map_df.iterrows():
        arrow_color = row['arrow_color']  # flow/speed color (or blue for selected)
        
        # Get speed value (may be 0 if not available)
        speed_val = float(row['speed']) if 'speed' in row and row['speed'] else 0
        
        # Use link_name for display
        link_name = row.get('link_name', row['direction'])
        
        links_data.append({
            'link_id': str(row['link_id']),
            'link_name': link_name,
            'from': [row['from_lon'], row['from_lat']],
            'to': [row['to_lon'], row['to_lat']],
            'color': arrow_color,
            'selected': row['link_id'] == selected_link,
            'flow': float(row['flow']),
            'speed': speed_val,
            'baseline_flow': float(row['baseline_flow']) if 'baseline_flow' in row else 0.0,
            'baseline_speed': float(row['baseline_speed']) if 'baseline_speed' in row else 0.0,
            'map_metric_value': float(row['map_metric_value']) if 'map_metric_value' in row else 0.0,
            'map_metric_label': str(row['map_metric_label']) if 'map_metric_label' in row else 'Value',
            'map_metric_unit': str(row['map_metric_unit']) if 'map_metric_unit' in row else '',
            'capacity': float(row['capacity']),
            'direction': row['direction'],
        })
    
    # Prepare node labels data
    nodes_data = []
    for _, row in node_coords_df.iterrows():
        nodes_data.append({
            'position': [row['lon'], row['lat']],
            'label': str(row['node_num'])
        })
    
    links_json = json.dumps(links_data)
    nodes_json = json.dumps(nodes_data)
    bounds_json = json.dumps(network_bounds or {})
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://unpkg.com/deck.gl@8.9.36/dist.min.js"></script>
        <script src="https://unpkg.com/maplibre-gl@2.4.0/dist/maplibre-gl.js"></script>
        <link href="https://unpkg.com/maplibre-gl@2.4.0/dist/maplibre-gl.css" rel="stylesheet" />
        <style>
            body {{ margin: 0; padding: 0; }}
            #container {{ width: 100%; height: 900px; position: relative; }}
            #click-info {{
                position: absolute;
                bottom: 10px;
                left: 10px;
                background: rgba(255,255,255,0.95);
                padding: 10px 14px;
                border-radius: 6px;
                font-size: 12px;
                z-index: 1000;
                box-shadow: 0 2px 6px rgba(0,0,0,0.2);
                max-width: 280px;
            }}
            #click-info.selected {{
                border-left: 4px solid #0064ff;
            }}
            #node-label-overlay {{
                position: absolute;
                inset: 0;
                pointer-events: none;
                z-index: 20;
            }}
            .node-label-badge {{
                position: absolute;
                min-width: 24px;
                height: 24px;
                padding: 0 4px;
                border: 2px solid rgb(74, 37, 0);
                border-radius: 999px;
                background: rgb(255, 184, 77);
                color: #111;
                box-sizing: border-box;
                font-family: "Segoe UI", Arial, sans-serif;
                font-size: 13px;
                font-weight: 800;
                line-height: 20px;
                text-align: center;
                transform: translate(-50%, -50%);
                white-space: nowrap;
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.25);
            }}
        </style>
    </head>
    <body>
        <div id="container">
            <div id="click-info">
                💡 <b>Click on a link</b> to select it
            </div>
            <div id="node-label-overlay" aria-hidden="true"></div>
        </div>
        <script>
            const linksData = {links_json};
            const nodesData = {nodes_json};
            const networkBounds = {bounds_json};
            let selectedLinkId = "{selected_link}";
            const blinkLinkId = "{blink_link if blink_link is not None else ''}";
            const blinkGeneration = {int(blink_generation)};
            const locateGeneration = {int(locate_generation)};
            const shouldFitNetwork = locateGeneration > 0;
            let blinkActive = Boolean(blinkLinkId);
            let blinkPulse = 1.0;
            let isDragging = false;
            
            const {{DeckGL, PolygonLayer, WebMercatorViewport}} = deck;

            function fittedNetworkViewState() {{
                if (
                    !shouldFitNetwork
                    || !Number.isFinite(networkBounds.minLon)
                    || typeof WebMercatorViewport !== 'function'
                ) {{
                    return {{
                        longitude: {center_lon},
                        latitude: {center_lat},
                        zoom: 11,
                        pitch: 0
                    }};
                }}
                const container = document.getElementById('container');
                const width = Math.max(container.clientWidth || 1200, 1);
                const height = Math.max(container.clientHeight || 900, 1);
                const viewport = new WebMercatorViewport({{width, height}});
                const fitted = viewport.fitBounds(
                    [
                        [networkBounds.minLon, networkBounds.minLat],
                        [networkBounds.maxLon, networkBounds.maxLat]
                    ],
                    {{padding: 55}}
                );
                return {{
                    longitude: fitted.longitude,
                    latitude: fitted.latitude,
                    zoom: Math.max(4, Math.min(13, fitted.zoom)),
                    pitch: 0
                }};
            }}

            const initialViewState = fittedNetworkViewState();
            let currentZoom = initialViewState.zoom;
            let currentViewState = initialViewState;

            function updateNodeLabels(viewState = currentViewState) {{
                const container = document.getElementById('container');
                const overlay = document.getElementById('node-label-overlay');
                if (!container || !overlay || typeof WebMercatorViewport !== 'function') {{
                    return;
                }}

                const width = Math.max(container.clientWidth || 1200, 1);
                const height = Math.max(container.clientHeight || 900, 1);
                const viewport = new WebMercatorViewport({{
                    width,
                    height,
                    longitude: viewState.longitude,
                    latitude: viewState.latitude,
                    zoom: viewState.zoom,
                    pitch: viewState.pitch || 0,
                    bearing: viewState.bearing || 0,
                }});

                overlay.replaceChildren();
                nodesData.forEach(node => {{
                    const [x, y] = viewport.project(node.position);
                    if (!Number.isFinite(x) || !Number.isFinite(y)) {{
                        return;
                    }}

                    const badge = document.createElement('div');
                    badge.className = 'node-label-badge';
                    badge.textContent = node.label;
                    badge.style.left = `${{x}}px`;
                    badge.style.top = `${{y}}px`;
                    overlay.appendChild(badge);
                }});
            }}

            // Build full-link arrow polygons.
            // Each link is drawn as a coloured chevron (rect body + triangular head)
            // offset perpendicularly so that bidirectional road pairs don't overlap.
            function createLinkArrows(zoom) {{
                const scale = Math.pow(2, 11 - zoom);
                const hw    = 0.00018 * scale;   // body half-width (scales with zoom)
                const hw2   = 0.002 * scale;    // arrowhead half-width (larger, scales with zoom)
                const headLen = 0.006 * scale;  // fixed head length (larger, scales with zoom only)
                const offset = 0.00035 * scale;  // perpendicular offset (scales with zoom)

                const polygons = [];

                linksData.forEach(link => {{
                    const [fLon, fLat] = link.from;
                    const [tLon, tLat] = link.to;
                    const dx = tLon - fLon, dy = tLat - fLat;
                    const len = Math.sqrt(dx*dx + dy*dy);
                    if (len < 1e-10) return;

                    const ux = dx/len, uy = dy/len;   // unit direction
                    const px = uy,  py = -ux;          // unit perp (right side)

                    const ox = px * offset, oy = py * offset;  // lateral offset

                    // Position arrowhead at middle of link, offset perpendicular.
                    const midLon = (fLon + tLon) / 2, midLat = (fLat + tLat) / 2;
                    const tipLon = midLon + ox, tipLat = midLat + oy;
                    const bx = tipLon - ux*headLen, by = tipLat - uy*headLen;

                    // Full-length body as a simple rectangle to avoid triangulation artifacts.
                    const bodyP1 = [fLon + ox + px*hw, fLat + oy + py*hw];
                    const bodyP2 = [tLon + ox + px*hw, tLat + oy + py*hw];
                    const bodyP3 = [tLon + ox - px*hw, tLat + oy - py*hw];
                    const bodyP4 = [fLon + ox - px*hw, fLat + oy - py*hw];

                    // Middle arrowhead as a separate triangle.
                    const headP1 = [bx + ox + px*hw2, by + oy + py*hw2];
                    const headP2 = [tipLon,           tipLat          ];
                    const headP3 = [bx + ox - px*hw2, by + oy - py*hw2];

                    // Table-selected links pulse yellow; otherwise selected links stay blue.
                    const isBlinking = blinkActive && link.link_id === blinkLinkId;
                    const blinkAlpha = Math.round(170 + blinkPulse * 85);
                    const color = isBlinking
                        ? [255, 193, 7, blinkAlpha]
                        : (link.selected ? [0, 100, 255, 255] : link.color);

                    const common = {{
                        color,
                        link_id: link.link_id,
                        link_name: link.link_name,
                        flow: link.flow,
                        speed: link.speed,
                        baseline_flow: link.baseline_flow,
                        baseline_speed: link.baseline_speed,
                        map_metric_value: link.map_metric_value,
                        map_metric_label: link.map_metric_label,
                        map_metric_unit: link.map_metric_unit,
                        capacity: link.capacity,
                        direction: link.direction,
                        selected: link.selected,
                        blink: isBlinking,
                    }};

                    polygons.push({{ ...common, polygon: [bodyP1, bodyP2, bodyP3, bodyP4] }});
                    polygons.push({{ ...common, polygon: [headP1, headP2, headP3] }});
                }});

                return polygons;
            }}

            // Function to select link - try multiple methods to update Streamlit state
            function selectLink(linkId) {{
                sessionStorage.setItem('clicked_link_id', linkId);
                
                let selectionSucceeded = false;
                
                // Approach 1: Try Streamlit API if available
                try {{
                    if (window.parent.streamlit) {{
                        window.parent.streamlit.setComponentValue(linkId);
                        selectionSucceeded = true;
                        console.log('Selection sent via Streamlit API');
                    }}
                }} catch (e) {{
                    console.log('Streamlit API not available:', e.message);
                }}
                
                // Approach 2: Modify parent URL and reload so Python reads clicked_link.
                if (!selectionSucceeded) {{
                    try {{
                        const baseUrl = window.parent.location.href.split('?')[0].split('#')[0];
                        const params = new URLSearchParams(window.parent.location.search);
                        params.set('clicked_link', linkId);
                        const newUrl = baseUrl + '?' + params.toString();
                        window.parent.history.pushState(null, '', newUrl);
                        selectionSucceeded = true;
                        console.log('Selection via URL pushState, reloading parent...');
                        setTimeout(() => {{
                            try {{
                                window.parent.location.reload();
                            }} catch (e) {{
                                try {{
                                    window.parent.postMessage({{type: 'streamlit:rerun'}}, '*');
                                }} catch (postMessageError) {{}}
                            }}
                        }}, 50);
                    }} catch (e) {{
                        console.log('Direct URL modification failed (iframe sandbox):', e.message);
                    }}
                }}
                
                // Approach 3: Try href replacement (full page reload)
                if (!selectionSucceeded) {{
                    try {{
                        const baseUrl = window.parent.location.href.split('?')[0].split('#')[0];
                        const params = new URLSearchParams(window.parent.location.search);
                        params.set('clicked_link', linkId);
                        window.parent.location.href = baseUrl + '?' + params.toString();
                        selectionSucceeded = true;
                        console.log('Selection via href replacement (full page reload)');
                    }} catch (e) {{
                        console.log('Href replacement failed:', e.message);
                    }}
                }}
                
                if (!selectionSucceeded) {{
                    console.warn('All selection methods failed. The dropdown menu can still be used to select links.');
                    const clickInfo = document.getElementById('click-info');
                    if (clickInfo) {{
                        clickInfo.className = 'selected';
                        clickInfo.innerHTML = '<small style="color: #f00;"><b>⚠ Selection failed.</b> Please use the dropdown menu above to select this link.</small>';
                    }}
                }}
            }}
            
            const deckgl = new DeckGL({{
                container: 'container',
                mapStyle: 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',
                initialViewState,
                controller: true,
                onDragStart: () => {{
                    isDragging = true;
                }},
                onDragEnd: () => {{
                    // Defer reset so the drag-end mouseup doesn't get treated as a click.
                    setTimeout(() => {{
                        isDragging = false;
                    }}, 0);
                }},
                onViewStateChange: ({{viewState}}) => {{
                    currentViewState = viewState;
                    currentZoom = viewState.zoom;
                    updateLayers();
                    updateNodeLabels(viewState);
                }},
                getTooltip: ({{object}}) => {{
                    if (object && object.link_id) {{
                        const linkDisplay = object.link_name || object.direction;
                        const metricValue = Number.isFinite(object.map_metric_value) ? object.map_metric_value : 0;
                        const flowValue = Number.isFinite(object.flow) ? object.flow : 0;
                        const speedValue = Number.isFinite(object.speed) ? object.speed : 0;
                        const capacityValue = Number.isFinite(object.capacity) ? object.capacity : 0;
                        const speedStr = object.speed > 0
                            ? `<b>Speed:</b> ${{speedValue.toFixed(1)}} m/s (${{(speedValue * 3.6).toFixed(1)}} km/h)<br/>`
                            : '';
                        const metricUnit = object.map_metric_unit || '';
                        const metricValueStr = metricValue >= 0
                            ? `+${{metricValue.toFixed(2)}}`
                            : `${{metricValue.toFixed(2)}}`;
                        return {{
                            html: `<div style="padding:4px">
                                <b>Link:</b> ${{linkDisplay}}<br/>
                                <b>${{object.map_metric_label || 'Value'}}:</b> ${{metricValueStr}}${{metricUnit ? ' ' + metricUnit : ''}}<br/>
                                <b>Flow:</b> ${{flowValue.toFixed(1)}} veh/hr<br/>
                                ${{speedStr}}
                                <b>Capacity:</b> ${{capacityValue.toFixed(0)}}<br/>
                            </div>`,
                            style: {{
                                backgroundColor: 'rgba(0, 0, 0, 0.85)',
                                color: 'white',
                                borderRadius: '6px',
                                fontSize: '12px'
                            }}
                        }};
                    }}
                    return null;
                }},
                onClick: (info) => {{
                    if (isDragging) return;
                    if (info.object && info.object.link_id) {{
                        const linkDisplay = info.object.link_name || info.object.direction;
                        const clickInfo = document.getElementById('click-info');
                        selectedLinkId = info.object.link_id;
                        blinkActive = false;
                        blinkPulse = 0.0;
                        linksData.forEach(link => {{
                            link.selected = link.link_id === selectedLinkId;
                        }});
                        updateLayers();

                        if (clickInfo) {{
                            clickInfo.className = 'selected';
                            clickInfo.innerHTML = `✅ <b>Selected:</b> ${{linkDisplay}}<br/><small style="display: block; margin-top: 6px; color: #555;">Updating selected link...</small>`;
                        }}
                        selectLink(selectedLinkId);
                    }}
                }},
                layers: []
            }});
            
            function updateLayers() {{
                // Full-link arrow polygons (body + arrowhead), coloured by flow or speed
                const linkArrows = createLinkArrows(currentZoom);
                const linkArrowLayer = new PolygonLayer({{
                    id: 'link-arrows-' + currentZoom.toFixed(1),
                    data: linkArrows,
                    getPolygon: d => d.polygon,
                    getFillColor: d => d.color,
                    getLineColor: d => d.blink ? [255, 77, 0, 255] : (d.selected ? [0, 60, 200, 255] : [0, 0, 0, 60]),
                    getLineWidth: d => d.blink ? (7 + blinkPulse * 7) : (d.selected ? 3 : 1),
                    lineWidthUnits: 'pixels',
                    lineWidthMinPixels: 1,
                    pickable: true,
                    extruded: false,
                }});
                
                deckgl.setProps({{layers: [linkArrowLayer]}});
            }}


            
            updateLayers();
            updateNodeLabels(initialViewState);
            if (blinkActive) {{
                const blinkStart = performance.now();
                const blinkDuration = 2800;
                function animateBlink(now) {{
                    const elapsed = now - blinkStart;
                    if (elapsed >= blinkDuration) {{
                        blinkActive = false;
                        blinkPulse = 0.0;
                        updateLayers();
                        return;
                    }}
                    blinkPulse = 0.5 + 0.5 * Math.sin((elapsed / 260) * Math.PI * 2);
                    updateLayers();
                    requestAnimationFrame(animateBlink);
                }}
                requestAnimationFrame(animateBlink);
            }}
        </script>
    </body>
    </html>
    """
    return html


def main():
    st.set_page_config(
        page_title="Traffic Flow & Speed Prediction",
        page_icon="🚗",
        layout="wide"
    )

    st.markdown(
        """
        <style>
            /* Keep main content background consistent with the navigation/sidebar frame. */
            [data-testid="stAppViewContainer"] > .main,
            [data-testid="stMain"],
            [data-testid="stMainBlockContainer"] {
                background-color: var(--secondary-background-color);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    st.title("🚗 Traffic Flow & Speed Prediction")
    st.markdown("Fixed feedforward model mode. Adjust hour and capacities to see predicted flows and speeds.")

    # Dedicated documentation route (opened in a new tab from the sidebar).
    if get_query_value(st.query_params, 'view') == 'documentation':
        st.title("📘 GUI Documentation")
        st.markdown(load_gui_readme_text())
        st.caption("Close this tab to return to the main GUI.")
        return

    source_context = (V3_GLOBALCAP_SOURCE, FIXED_MODEL_CODE)
    loading_text = 'Loading fixed feedforward model...'

    try:
        with st.spinner(loading_text):
            loader = load_data_and_models()
            link_coords_df = get_link_coords_df(loader, _version=6)
            node_coords_df = get_node_coords_df(loader, _version=1)
    except Exception as exc:
        st.error(f'Failed to load fixed feedforward model artifacts: {exc}')
        return

    sync_state_to_loaded_source(source_context, loader)
    st.caption(f"Active model source: {loader.model_source_label} | Model: {FIXED_MODEL_NAME}")
    
    # Check if speed prediction is available
    speed_available = loader.flowspeed_available
    
    # Initialize session state
    if 'predictions_cache' not in st.session_state:
        st.session_state.predictions_cache = {}
    if 'predictions_cache_flowspeed' not in st.session_state:
        st.session_state.predictions_cache_flowspeed = {}
    if 'baseline_predictions' not in st.session_state:
        st.session_state.baseline_predictions = {}
    if 'baseline_speed_predictions' not in st.session_state:
        st.session_state.baseline_speed_predictions = {}
    if 'local_sanity_cache' not in st.session_state:
        st.session_state.local_sanity_cache = {}
    if 'base_scenario' not in st.session_state:
        st.session_state.base_scenario = BASE_SCENARIO_3600
    if 'base_capacities' not in st.session_state:
        st.session_state.base_capacities = build_uniform_capacities(loader.unique_link_ids, DEFAULT_CAPACITY)
    if 'capacities' not in st.session_state:
        st.session_state.capacities = copy.deepcopy(st.session_state.base_capacities)
    if 'selected_link' not in st.session_state:
        st.session_state.selected_link = loader.unique_link_ids[0]
    if 'clicked_link' not in st.session_state:
        st.session_state.clicked_link = None
    if 'display_mode' not in st.session_state:
        st.session_state.display_mode = 'speed'
    if 'value_mode' not in st.session_state:
        st.session_state.value_mode = 'relative_percent'
    elif st.session_state.value_mode == 'relative':
        # Backward compatibility for previous two-option radio state.
        st.session_state.value_mode = 'relative_delta'
    if 'selected_hour' not in st.session_state:
        st.session_state.selected_hour = 12
    if 'local_sensitivity_hour' not in st.session_state:
        st.session_state.local_sensitivity_hour = int(st.session_state.selected_hour)
    if 'capacity_table_generation' not in st.session_state:
        st.session_state.capacity_table_generation = 0
    if 'blink_link' not in st.session_state:
        st.session_state.blink_link = None
    if 'blink_generation' not in st.session_state:
        st.session_state.blink_generation = 0
    if 'locate_generation' not in st.session_state:
        st.session_state.locate_generation = 0
    if 'table_sort_mode' not in st.session_state:
        st.session_state.table_sort_mode = TABLE_SORT_SELECTED_FIRST
    if 'table_sort_ascending' not in st.session_state:
        st.session_state.table_sort_ascending = True

    loaded_link_ids = set(loader.unique_link_ids)
    if set(st.session_state.base_capacities.keys()) != loaded_link_ids:
        st.session_state.base_capacities = build_uniform_capacities(loader.unique_link_ids, DEFAULT_CAPACITY)
    if set(st.session_state.capacities.keys()) != loaded_link_ids:
        st.session_state.capacities = copy.deepcopy(st.session_state.base_capacities)

    # Create link options with simple naming (e.g., "3-7")
    link_options = []
    link_id_to_display = {}
    display_to_link_id = {}
    for lid in loader.unique_link_ids:
        row = link_coords_df[link_coords_df['link_id'] == lid].iloc[0]
        link_name = row['link_name']
        display_name = f"{link_name}"
        link_options.append(display_name)
        link_id_to_display[lid] = display_name
        display_to_link_id[display_name] = lid

    st.session_state.selected_model_code = FIXED_MODEL_CODE

    if st.session_state.selected_link not in link_id_to_display:
        st.session_state.selected_link = loader.unique_link_ids[0]
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Controls")
        st.caption(f"Model: {FIXED_MODEL_NAME}")

        st.markdown("---")
        st.header("🎨 Display Mode")
        if speed_available:
            display_mode_pending = st.radio(
                "Color arrows by:",
                options=['flow', 'speed'],
                index=['flow', 'speed'].index(st.session_state.display_mode),
                format_func=lambda x: '🚗 Flow (veh/hr)' if x == 'flow' else '⚡ Speed (m/s)',
                horizontal=True,
            )
        else:
            st.info("Speed models not available. Showing flow only.")
            display_mode_pending = 'flow'
        st.session_state.display_mode = display_mode_pending

        value_modes = ['absolute', 'relative_delta', 'relative_percent']
        value_mode_pending = st.radio(
            'Map values:',
            options=value_modes,
            index=value_modes.index(st.session_state.value_mode),
            format_func=lambda x: (
                'Absolute prediction' if x == 'absolute'
                else 'Relative delta (pred - baseline)' if x == 'relative_delta'
                else 'Relative percent (% vs baseline)'
            ),
            horizontal=True,
        )
        st.session_state.value_mode = value_mode_pending

        st.markdown("---")
        st.header("🧱 Base Scenario")
        current_base_scenario = st.session_state.get('base_scenario', BASE_SCENARIO_3600)
        if current_base_scenario not in BASE_SCENARIO_OPTIONS:
            current_base_scenario = BASE_SCENARIO_3600
        selected_base_scenario = st.selectbox(
            "Select base scenario",
            BASE_SCENARIO_OPTIONS,
            index=BASE_SCENARIO_OPTIONS.index(current_base_scenario),
        )

        if selected_base_scenario != st.session_state.base_scenario:
            st.session_state.base_scenario = selected_base_scenario
            if selected_base_scenario in BASE_SCENARIO_VALUES:
                reset_to_uniform_base(loader, BASE_SCENARIO_VALUES[selected_base_scenario])
            else:
                reset_capacity_table_generation()
            st.rerun()

        st.markdown("---")
        st.header("📍 Link Selection")
        current_display = link_id_to_display.get(st.session_state.selected_link, link_options[0])
        try:
            current_idx = link_options.index(current_display)
        except ValueError:
            current_idx = 0

        selected_display_pending = st.selectbox("Select Link", link_options, index=current_idx)
        st.session_state.selected_link = display_to_link_id[selected_display_pending]

        st.markdown("---")
        st.header("📘 Documentation")
        st.markdown(
            '<a href="?view=documentation" target="_blank" rel="noopener noreferrer">'
            '<button style="width:100%; padding:8px 12px; border-radius:8px; '
            'border:1px solid rgba(128, 128, 128, 0.35); '
            'background:var(--secondary-background-color); color:var(--text-color); '
            'cursor:pointer; font-weight:600;">Show documentation</button>'
            '</a>',
            unsafe_allow_html=True,
        )
        st.caption("Opens in a new tab. If your browser blocks it, allow pop-ups/new tabs for this site.")

    query_params = st.query_params
    clicked_link = resolve_link_id(loader, get_query_value(query_params, 'clicked_link'))
    if clicked_link is not None:
        st.session_state.selected_link = clicked_link
        st.session_state.blink_link = None
        clear_query_keys('clicked_link')
        st.rerun()
    
    # Get applied sidebar values
    model_code = FIXED_MODEL_CODE
    selected_model_name = FIXED_MODEL_NAME
    hour = int(st.session_state.selected_hour)
    selected_link = st.session_state.selected_link
    display_mode = st.session_state.display_mode
    value_mode = st.session_state.value_mode
    source_cache_key = (V3_GLOBALCAP_SOURCE, FIXED_MODEL_CODE)
    base_scenario = st.session_state.base_scenario
    base_capacities = st.session_state.base_capacities.copy()
    base_capacity_cache_key = tuple(sorted(base_capacities.items()))

    # ----- BASELINE PREDICTIONS (Base Capacity column values) -----
    # Compute once and cache for comparison
    baseline_cache_key = (source_cache_key, model_code, 'baseline', base_capacity_cache_key)
    if baseline_cache_key not in st.session_state.baseline_predictions:
        pred_engine = PredictionEngine(loader)
        pred_engine.capacities = base_capacities
        baseline_flow_preds = pred_engine.predict_all(model_code)
        if speed_available:
            flow_preds, speed_preds = pred_engine.predict_all_flowspeed(model_code)
            st.session_state.baseline_speed_predictions[baseline_cache_key] = speed_preds
        st.session_state.baseline_predictions[baseline_cache_key] = baseline_flow_preds
    
    baseline_flow_predictions = st.session_state.baseline_predictions[baseline_cache_key]
    baseline_speed_predictions = st.session_state.baseline_speed_predictions.get(baseline_cache_key)
    baseline_flows_hour = {lid: preds[hour] for lid, preds in baseline_flow_predictions.items()}
    baseline_speeds_hour = (
        {lid: preds[hour] for lid, preds in baseline_speed_predictions.items()}
        if baseline_speed_predictions else None
    )
    
    # ----- CURRENT PREDICTIONS (with user-modified capacities) -----
    # Get predictions (cached in session state)
    cache_key = (source_cache_key, model_code, tuple(sorted(st.session_state.capacities.items())))
    
    # Always get flow predictions (flow-only models)
    if cache_key not in st.session_state.predictions_cache:
        pred_engine = PredictionEngine(loader)
        pred_engine.capacities = st.session_state.capacities.copy()
        st.session_state.predictions_cache[cache_key] = pred_engine.predict_all(model_code)
    
    all_flow_predictions = st.session_state.predictions_cache[cache_key]
    flows = {lid: preds[hour] for lid, preds in all_flow_predictions.items()}
    
    # Get speed predictions if available and display mode is speed
    speeds = None
    all_speed_predictions = None
    if speed_available:
        if cache_key not in st.session_state.predictions_cache_flowspeed:
            pred_engine = PredictionEngine(loader)
            pred_engine.capacities = st.session_state.capacities.copy()
            flow_preds, speed_preds = pred_engine.predict_all_flowspeed(model_code)
            st.session_state.predictions_cache_flowspeed[cache_key] = (flow_preds, speed_preds)
        
        _, all_speed_predictions = st.session_state.predictions_cache_flowspeed[cache_key]
        speeds = {lid: preds[hour] for lid, preds in all_speed_predictions.items()}

    flows_display = {lid: value * FLOW_DISPLAY_SCALE for lid, value in flows.items()}
    baseline_flows_hour_display = {
        lid: value * FLOW_DISPLAY_SCALE for lid, value in baseline_flows_hour.items()
    }

    # Calculate color limits based on display and value mode for map coloring
    if display_mode == 'speed' and speeds and all_speed_predictions:
        if value_mode == 'relative_percent' and baseline_speeds_hour:
            rel_values = [
                apply_relative_neutral_deadband(
                    compute_relative_percent(speeds[lid], baseline_speeds_hour.get(lid, 0.0)),
                    display_mode,
                    value_mode,
                )
                for lid in loader.unique_link_ids
            ]
            min_val, max_val = get_sensitive_color_limits(rel_values, center_val=0.0)
        elif value_mode in ('relative', 'relative_delta') and baseline_speeds_hour:
            rel_values = [
                apply_relative_neutral_deadband(
                    speeds[lid] - baseline_speeds_hour.get(lid, 0.0),
                    display_mode,
                    value_mode,
                )
                for lid in loader.unique_link_ids
            ]
            min_val, max_val = get_sensitive_color_limits(rel_values, center_val=0.0)
        else:
            current_vals = list(speeds.values())
            min_val, max_val = get_sensitive_color_limits(current_vals)
    else:
        if value_mode == 'relative_percent':
            rel_values = [
                apply_relative_neutral_deadband(
                    compute_relative_percent(flows[lid], baseline_flows_hour.get(lid, 0.0)),
                    display_mode,
                    value_mode,
                )
                for lid in loader.unique_link_ids
            ]
            min_val, max_val = get_sensitive_color_limits(rel_values, center_val=0.0)
        elif value_mode in ('relative', 'relative_delta'):
            rel_values = [
                apply_relative_neutral_deadband(
                    flows_display[lid] - baseline_flows_hour_display.get(lid, 0.0),
                    display_mode,
                    value_mode,
                )
                for lid in loader.unique_link_ids
            ]
            min_val, max_val = get_sensitive_color_limits(rel_values, center_val=0.0)
        else:
            current_vals = list(flows_display.values())
            min_val, max_val = get_sensitive_color_limits(current_vals)
    
    # Main layout - map on top, table below
    # ----- MAP SECTION -----
    metric_label = "Speed" if display_mode == 'speed' else "Flow"
    if value_mode in ('relative', 'relative_delta'):
        value_label = 'Relative Δ'
    elif value_mode == 'relative_percent':
        value_label = 'Relative %'
    else:
        value_label = 'Absolute'
    st.subheader(f"📍 Network Map - {selected_model_name} @ {hour}:00 ({metric_label}, {value_label})")
    
    # Prepare map data
    map_df = create_map_data(
        link_coords_df,
        flows_display,
        speeds,
        baseline_flows_hour_display,
        baseline_speeds_hour,
        st.session_state.capacities,
        selected_link,
        min_val,
        max_val,
        display_mode,
        value_mode,
    )
    
    # Calculate center
    center_lat = map_df['mid_lat'].mean()
    center_lon = map_df['mid_lon'].mean()
    all_lons = pd.concat([map_df['from_lon'], map_df['to_lon']]).astype(float)
    all_lats = pd.concat([map_df['from_lat'], map_df['to_lat']]).astype(float)
    lon_pad = max(float(all_lons.max() - all_lons.min()) * 0.04, 0.002)
    lat_pad = max(float(all_lats.max() - all_lats.min()) * 0.04, 0.002)
    network_bounds = {
        'minLon': float(all_lons.min() - lon_pad),
        'maxLon': float(all_lons.max() + lon_pad),
        'minLat': float(all_lats.min() - lat_pad),
        'maxLat': float(all_lats.max() + lat_pad),
    }
    
    # Create map with click support and node labels.
    # This Streamlit version doesn't support `key` for components.html.
    # Force remount by embedding a state token and slightly varying height.
    map_component_token = (
        f"map_{V3_GLOBALCAP_SOURCE}_{FIXED_MODEL_CODE}_{model_code}_{hour}_"
        f"{display_mode}_{value_mode}_{selected_link}_"
        f"blink_{st.session_state.get('blink_generation', 0)}_"
        f"locate_{st.session_state.get('locate_generation', 0)}_"
        f"{min_val:.6g}_{max_val:.6g}_"
        f"{hash(tuple(sorted(st.session_state.capacities.items())))}"
    )
    map_html = create_custom_map_html(
        map_df,
        node_coords_df,
        center_lat,
        center_lon,
        selected_link,
        display_mode,
        st.session_state.get('blink_link'),
        st.session_state.get('blink_generation', 0),
        network_bounds,
        st.session_state.get('locate_generation', 0),
    )
    map_html = f"<!-- {map_component_token} -->\n" + map_html
    components.html(map_html, height=550)
    
    # Instructions and color legend
    if display_mode == 'speed':
        if value_mode in ('relative', 'relative_delta'):
            legend_text = "Lower Speed Δ → Unchanged → Higher Speed Δ"
            gradient = "linear-gradient(to right, #dc143c, #0078ff, #14b414)"
        elif value_mode == 'relative_percent':
            legend_text = "Lower Speed % → Unchanged → Higher Speed %"
            gradient = "linear-gradient(to right, #dc143c, #0078ff, #14b414)"
        else:
            legend_text = "Low Speed → Mid → High Speed"
            gradient = "linear-gradient(to right, #ff0000, #ffff00, #00ff00)"
    else:
        if value_mode in ('relative', 'relative_delta'):
            legend_text = "Lower Flow Δ → Unchanged → Higher Flow Δ"
            gradient = "linear-gradient(to right, #dc143c, #0078ff, #14b414)"
        elif value_mode == 'relative_percent':
            legend_text = "Lower Flow % → Unchanged → Higher Flow %"
            gradient = "linear-gradient(to right, #dc143c, #0078ff, #14b414)"
        else:
            legend_text = "Low Flow → Mid → High Flow"
            gradient = "linear-gradient(to right, #ff0000, #ffff00, #00ff00)"
    
    st.markdown(f"""
    <div style="display: flex; align-items: center; justify-content: space-between; margin-top: 5px;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="background: {gradient}; 
                         width: 150px; height: 16px; display: inline-block; border-radius: 3px;"></span>
            <span style="font-size: 12px;">{legend_text}</span>
        </div>
        <div style="font-size: 11px; color: #666;">
            🔵 Selected link | Node labels shown (1-{len(node_coords_df)})
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.slider(
        "Hour of Day",
        min_value=5,
        max_value=22,
        key='selected_hour',
    )
    
    st.markdown("---")
    
    # ----- EDITABLE CAPACITY TABLE -----
    st.subheader("📋 Link Capacities & Predictions")
    
    # Build table data with all columns
    table_data = []
    for lid in loader.unique_link_ids:
        row = link_coords_df[link_coords_df['link_id'] == lid].iloc[0]
        
        # Get baseline predictions at current hour
        base_flow = baseline_flows_hour_display.get(lid, 0.0)
        base_speed = baseline_speed_predictions[lid][hour] if baseline_speed_predictions and lid in baseline_speed_predictions else 0
        
        # Get current predictions
        pred_flow = flows_display.get(lid, 0.0)
        pred_speed = speeds.get(lid, 0) if speeds else 0
        base_capacity = float(st.session_state.base_capacities.get(lid, DEFAULT_CAPACITY))
        current_capacity = float(st.session_state.capacities.get(lid, base_capacity))
        selected_uniform_base = BASE_SCENARIO_VALUES.get(base_scenario)
        if base_scenario == BASE_SCENARIO_CUSTOM:
            is_modified = not np.isclose(current_capacity, base_capacity)
        else:
            is_modified = (
                selected_uniform_base is not None
                and not np.isclose(base_capacity, selected_uniform_base)
            ) or not np.isclose(current_capacity, base_capacity)
        
        table_data.append({
            'Locate': False,
            'Modified': '●' if is_modified else '',
            'Link': row['link_name'],  # e.g., "3-7"
            'From': row['from_num'],
            'To': row['to_num'],
            'Base Capacity': int(round(base_capacity)),
            'Capacity': int(round(current_capacity)),
            'Base Flow': round(base_flow, 1),
            'Base Speed': round(base_speed, 2),
            'Pred Flow': round(pred_flow, 1),
            'Pred Speed': round(pred_speed, 2),
            '_link_id': lid,  # Hidden column for lookup
            '_order': int(loader.link_to_idx.get(lid, 0)),
            '_modified': is_modified,
            '_selected': lid == selected_link,
        })

    table_df = pd.DataFrame(table_data)
    sort_options = [
        TABLE_SORT_SELECTED_FIRST,
        TABLE_SORT_MODIFIED_FIRST,
        'Link',
        'From-To',
        'Base Capacity',
        'Capacity',
        'Base Flow',
        'Base Speed',
        'Pred Flow',
        'Pred Speed',
    ]
    sort_cols = st.columns([2, 1, 4])
    with sort_cols[0]:
        st.selectbox(
            'Table sort',
            sort_options,
            key='table_sort_mode',
        )
    with sort_cols[1]:
        st.checkbox(
            'Ascending',
            key='table_sort_ascending',
            disabled=st.session_state.table_sort_mode in (TABLE_SORT_SELECTED_FIRST, TABLE_SORT_MODIFIED_FIRST),
        )

    sort_mode = st.session_state.table_sort_mode
    sort_ascending = bool(st.session_state.table_sort_ascending)
    if sort_mode == TABLE_SORT_SELECTED_FIRST:
        table_display_df = table_df.sort_values(
            by=['_selected', '_order'],
            ascending=[False, True],
            kind='mergesort',
        ).reset_index(drop=True)
    elif sort_mode == TABLE_SORT_MODIFIED_FIRST:
        table_display_df = table_df.sort_values(
            by=['_modified', '_order'],
            ascending=[False, True],
            kind='mergesort',
        ).reset_index(drop=True)
    elif sort_mode == 'From-To':
        table_display_df = table_df.sort_values(
            by=['From', 'To', '_order'],
            ascending=[sort_ascending, sort_ascending, True],
            kind='mergesort',
        ).reset_index(drop=True)
    elif sort_mode in table_df.columns:
        table_display_df = table_df.sort_values(
            by=[sort_mode, '_order'],
            ascending=[sort_ascending, True],
            kind='mergesort',
        ).reset_index(drop=True)
    else:
        table_display_df = table_df.sort_values('_order', kind='mergesort').reset_index(drop=True)

    table_columns = [
        'Locate',
        'Modified',
        'Link',
        'From',
        'To',
        'Base Capacity',
        'Capacity',
        'Base Flow',
        'Base Speed',
        'Pred Flow',
        'Pred Speed',
    ]
    edited_df = st.data_editor(
        table_display_df[table_columns],
        column_config={
            'Locate': st.column_config.CheckboxColumn('Locate', help='Check to select and pulse this link on the map.', width='small'),
            'Modified': st.column_config.TextColumn('Modified', disabled=True, width='small'),
            'Link': st.column_config.TextColumn('Link', disabled=True, width='small'),
            'From': st.column_config.NumberColumn('From', disabled=True, width='small'),
            'To': st.column_config.NumberColumn('To', disabled=True, width='small'),
            'Base Capacity': st.column_config.NumberColumn(
                'Base Capacity', min_value=CAPACITY_MIN, max_value=CAPACITY_MAX, step=CAPACITY_STEP, width='small'
            ),
            'Capacity': st.column_config.NumberColumn(
                'Capacity', min_value=CAPACITY_MIN, max_value=CAPACITY_MAX, step=CAPACITY_STEP, width='small'
            ),
            'Base Flow': st.column_config.NumberColumn('Base Flow', disabled=True, format='%.1f', width='small'),
            'Base Speed': st.column_config.NumberColumn('Base Spd', disabled=True, format='%.2f', width='small'),
            'Pred Flow': st.column_config.NumberColumn('Pred Flow', disabled=True, format='%.1f', width='small'),
            'Pred Speed': st.column_config.NumberColumn('Pred Spd', disabled=True, format='%.2f', width='small'),
        },
        disabled=['Modified', 'Link', 'From', 'To', 'Base Flow', 'Base Speed', 'Pred Flow', 'Pred Speed'],
        hide_index=True,
        use_container_width=True,
        key=f"capacity_table_{st.session_state.capacity_table_generation}_{sort_mode}_{sort_ascending}",
    )

    table_changed = False
    baseline_changed = False
    current_changed = False
    blink_link_from_table = None
    locate_requested = False

    locate_rows = edited_df.index[edited_df['Locate'].fillna(False)].tolist()
    if locate_rows:
        locate_idx = int(locate_rows[0])
        blink_link_from_table = table_display_df.iloc[locate_idx]['_link_id']
        locate_requested = True
        table_changed = True

    for i in range(len(edited_df)):
        lid = table_display_df.iloc[i]['_link_id']
        old_base_capacity = float(st.session_state.base_capacities.get(lid, DEFAULT_CAPACITY))
        old_capacity = float(st.session_state.capacities.get(lid, old_base_capacity))
        try:
            new_base_capacity = float(edited_df.iloc[i]['Base Capacity'])
        except (TypeError, ValueError):
            new_base_capacity = old_base_capacity
        try:
            new_capacity = float(edited_df.iloc[i]['Capacity'])
        except (TypeError, ValueError):
            new_capacity = old_capacity

        if not np.isfinite(new_base_capacity):
            new_base_capacity = old_base_capacity
        if not np.isfinite(new_capacity):
            new_capacity = old_capacity

        new_base_capacity = float(np.clip(new_base_capacity, CAPACITY_MIN, CAPACITY_MAX))
        new_capacity = float(np.clip(new_capacity, CAPACITY_MIN, CAPACITY_MAX))
        base_cell_changed = not np.isclose(new_base_capacity, old_base_capacity)
        capacity_cell_changed = not np.isclose(new_capacity, old_capacity)

        if base_cell_changed:
            was_current_synced_to_base = np.isclose(old_capacity, old_base_capacity)
            st.session_state.base_capacities[lid] = new_base_capacity
            baseline_changed = True
            table_changed = True
            blink_link_from_table = lid
            if base_scenario == BASE_SCENARIO_CUSTOM and was_current_synced_to_base and not capacity_cell_changed:
                st.session_state.capacities[lid] = new_base_capacity
                current_changed = True

        if capacity_cell_changed:
            st.session_state.capacities[lid] = new_capacity
            current_changed = True
            table_changed = True
            blink_link_from_table = lid

    if table_changed:
        if blink_link_from_table is not None:
            st.session_state.selected_link = blink_link_from_table
            st.session_state.blink_link = blink_link_from_table
            st.session_state.blink_generation = st.session_state.get('blink_generation', 0) + 1
            if locate_requested:
                st.session_state.locate_generation = st.session_state.get('locate_generation', 0) + 1
        if baseline_changed:
            st.session_state.baseline_predictions = {}
            st.session_state.baseline_speed_predictions = {}
        if current_changed:
            st.session_state.predictions_cache = {}
            st.session_state.predictions_cache_flowspeed = {}
        if baseline_changed or current_changed:
            st.session_state.local_sanity_cache = {}
        reset_capacity_table_generation()
        st.rerun()

    col_reset, col_spacer = st.columns([1, 5])
    with col_reset:
        if st.button("↩️ Reset All", use_container_width=True):
            st.session_state.capacities = copy.deepcopy(st.session_state.base_capacities)
            clear_prediction_caches(include_baseline=False)
            reset_capacity_table_generation()
            st.rerun()

    # ----- SELECTED LINK TIME SERIES -----
    st.markdown("---")
    selected_link_row = link_coords_df[link_coords_df['link_id'] == selected_link].iloc[0]
    selected_link_name = selected_link_row['link_name']
    selected_link_base_capacity = float(st.session_state.base_capacities.get(selected_link, DEFAULT_CAPACITY))

    if display_mode == 'speed' and speed_available and all_speed_predictions:
        st.subheader(f"📈 Selected Link vs Hour - Speed ({selected_link_name})")
        pred_series_dict = all_speed_predictions[selected_link]
        baseline_series_dict = baseline_speed_predictions[selected_link] if baseline_speed_predictions else None
        y_label = 'Speed (m/s)'
    else:
        st.subheader(f"📈 Selected Link vs Hour - Flow ({selected_link_name})")
        pred_series_dict = {
            h: value * FLOW_DISPLAY_SCALE for h, value in all_flow_predictions[selected_link].items()
        }
        baseline_series_dict = (
            {h: value * FLOW_DISPLAY_SCALE for h, value in baseline_flow_predictions[selected_link].items()}
            if baseline_flow_predictions else None
        )
        y_label = FLOW_DISPLAY_LABEL

    hours_sorted = sorted(pred_series_dict.keys())
    pred_values = [pred_series_dict[h] for h in hours_sorted]
    baseline_values = [baseline_series_dict[h] for h in hours_sorted] if baseline_series_dict else None

    fig_ts, ax_ts = plt.subplots(figsize=(10, 4))
    if baseline_values is not None:
        ax_ts.plot(
            hours_sorted,
            baseline_values,
            '--',
            color='gray',
            linewidth=1.8,
            label=f'Baseline (base capacity={selected_link_base_capacity:.0f})',
        )
    ax_ts.plot(hours_sorted, pred_values, '-o', color='#0a84ff', linewidth=2.3, markersize=4, label='Current prediction')
    if hour in pred_series_dict:
        ax_ts.scatter([hour], [pred_series_dict[hour]], color='#ff3b30', s=60, zorder=5, label=f'Selected hour ({hour}:00)')
    ax_ts.set_xlabel('Hour of day')
    ax_ts.set_ylabel(y_label)
    ax_ts.set_xticks(hours_sorted)
    ax_ts.grid(True, alpha=0.3)
    ax_ts.legend(loc='best')
    fig_ts.tight_layout()
    st.pyplot(fig_ts, use_container_width=True)
    plt.close(fig_ts)

    # ----- FLOW VS SPEED SCATTER (selected link, one point per hour) -----
    if speed_available and all_speed_predictions:
        st.subheader(f"📉 Selected Link Flow vs Speed ({selected_link_name})")

        flow_series = all_flow_predictions[selected_link]
        speed_series = all_speed_predictions[selected_link]
        shared_hours = sorted(set(flow_series.keys()) & set(speed_series.keys()))

        if shared_hours:
            flow_speed_chart_flow_label = FLOW_DISPLAY_LABEL
            flow_speed_df = pd.DataFrame({
                'Hour': shared_hours,
                flow_speed_chart_flow_label: [flow_series[h] * FLOW_DISPLAY_SCALE for h in shared_hours],
                'Speed (m/s)': [speed_series[h] for h in shared_hours],
            })

            scatter_chart = (
                alt.Chart(flow_speed_df)
                .mark_circle(size=90, opacity=0.85, color='#0a84ff')
                .encode(
                    x=alt.X(f'{flow_speed_chart_flow_label}:Q', title=flow_speed_chart_flow_label),
                    y=alt.Y('Speed (m/s):Q', title='Speed (m/s)'),
                    tooltip=[
                        alt.Tooltip('Hour:Q', title='Hour'),
                        alt.Tooltip(f'{flow_speed_chart_flow_label}:Q', title='Flow', format='.1f'),
                        alt.Tooltip('Speed (m/s):Q', title='Speed', format='.2f'),
                    ],
                )
                .properties(height=320)
                .interactive()
            )

            path_chart = (
                alt.Chart(flow_speed_df)
                .mark_line(color='#7f8c8d', opacity=0.55)
                .encode(
                    x=f'{flow_speed_chart_flow_label}:Q',
                    y='Speed (m/s):Q',
                    order='Hour:Q',
                )
            )

            selected_hour_df = flow_speed_df[flow_speed_df['Hour'] == hour]
            highlight_chart = (
                alt.Chart(selected_hour_df)
                .mark_circle(size=190, color='#ff3b30', opacity=0.95)
                .encode(
                    x=f'{flow_speed_chart_flow_label}:Q',
                    y='Speed (m/s):Q',
                    tooltip=[
                        alt.Tooltip('Hour:Q', title='Hour'),
                        alt.Tooltip(f'{flow_speed_chart_flow_label}:Q', title='Flow', format='.1f'),
                        alt.Tooltip('Speed (m/s):Q', title='Speed', format='.2f'),
                    ],
                )
            )

            st.altair_chart(path_chart + scatter_chart + highlight_chart, use_container_width=True)
            st.caption('Hover over a point to see its hour.')
        else:
            st.info('No overlapping flow/speed hourly data available for this selected link.')
    else:
        st.info('Flow vs speed scatter is shown when speed predictions are available.')

    # ----- LOCAL DIRECTIONAL SANITY FIGURE (selected center link + independent hour) -----
    st.markdown('---')
    st.subheader(f"🧭 Local Upstream/Downstream Sensitivity ({selected_link_name})")
    local_hour = st.slider(
        "Sensitivity Hour",
        min_value=5,
        max_value=22,
        key='local_sensitivity_hour',
    )

    upstream_links, downstream_links, parallel_links, tracked_links = get_local_subnetwork_links(
        loader,
        selected_link,
        max_upstream=None,
        max_downstream=None,
        include_parallel_links=True,
    )

    if len(tracked_links) <= 1:
        st.info('No upstream/downstream neighbors found for the selected link.')
    else:
        local_cache_key = (
            source_cache_key,
            model_code,
            selected_link,
            local_hour,
            tuple(tracked_links),
            tuple(sorted(st.session_state.capacities.items())),
            speed_available,
        )

        if local_cache_key not in st.session_state.local_sanity_cache:
            sweep_df, sweep_values = compute_local_capacity_sweep(
                loader,
                model_code,
                st.session_state.capacities,
                selected_link,
                local_hour,
                tracked_links,
                speed_available,
            )
            st.session_state.local_sanity_cache[local_cache_key] = {
                'sweep_df': sweep_df,
                'sweep_values': sweep_values.tolist(),
            }

        cache_blob = st.session_state.local_sanity_cache[local_cache_key]
        sweep_df = cache_blob['sweep_df']
        sweep_df_display = sweep_df.copy()
        sweep_df_display['flow'] = sweep_df_display['flow'] * FLOW_DISPLAY_SCALE
        current_center_capacity = float(st.session_state.capacities.get(selected_link, DEFAULT_CAPACITY))

        fig_local = create_local_dependency_figure(
            loader,
            link_coords_df,
            sweep_df_display,
            selected_link,
            upstream_links,
            downstream_links,
            parallel_links,
            local_hour,
            selected_model_name,
            current_center_capacity,
            speed_available,
        )
        st.pyplot(fig_local, use_container_width=True)
        plt.close(fig_local)

        upstream_names = [
            link_coords_df[link_coords_df['link_id'] == lid].iloc[0]['link_name']
            for lid in upstream_links
        ]
        downstream_names = [
            link_coords_df[link_coords_df['link_id'] == lid].iloc[0]['link_name']
            for lid in downstream_links
        ]
        parallel_names = [
            link_coords_df[link_coords_df['link_id'] == lid].iloc[0]['link_name']
            for lid in parallel_links
        ]
        st.caption(
            'Center link is selected from Link Selection. '
            f"Detected upstream links: {', '.join(upstream_names) if upstream_names else 'none'} | "
            f"Detected downstream links: {', '.join(downstream_names) if downstream_names else 'none'} | "
            f"Detected parallel links: {', '.join(parallel_names) if parallel_names else 'none'}."
        )
    
if __name__ == "__main__":
    main()
