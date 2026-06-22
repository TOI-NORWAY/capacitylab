"""
Data loader and preprocessor for traffic flow and speed prediction.
Loads network data, builds graph, and prepares features.
Supports both flow-only and flow+speed prediction models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
import torch
import pickle
import os

from models import FeedforwardNN_GlobalCapacity


V3_GLOBALCAP_FILES = (
    'best_ff_flowspeed_v3_globalcap.pth',
    'artifacts_flowspeed_v3_globalcap.pkl',
    'scalers_flowspeed_v3_globalcap.pkl',
    'model_configs_flowspeed_v3_globalcap.pkl',
)
DEFAULT_V3_GLOBALCAP_DIR = Path('./model')
V3_GUI_FF_CHECKPOINT = 'best_ff_flowspeed_v3_globalcap_gui.pth'
NETWORK_XML_FILENAME = 'output_network.xml'
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DataLoader:
    """Load and preprocess traffic flow/speed data and network topology."""
    
    def __init__(self, base_path=None):
        if base_path is None:
            base_path = PROJECT_ROOT
        self.base_path = Path(base_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.DAYTIME_HOURS = list(range(5, 23))
        self.n_hours = len(self.DAYTIME_HOURS)
        
        # Data containers
        self.nodes = {}
        self.links_info = {}
        self.unique_link_ids = []
        self.link_to_idx = {}
        self.n_links = 0
        self.n_static_features = 10
        
        # Preprocessed data
        self.static_features = None
        self.static_features_scaled = None
        self.edge_index = None
        self.edge_attr = None
        self.upstream_indices = []
        self.downstream_indices = []
        self.sample_context_feature_names = []
        
        # Scalers
        self.static_scaler = StandardScaler()
        self.capacity_scaler = StandardScaler()
        self.flow_scaler = StandardScaler()
        self.speed_scaler = StandardScaler()  # Added for speed
        
        # Trained models (v3_globalcap flow+speed feedforward)
        self.model_ff = None
        self.model_ff_flowspeed = None
        
        # Flag to track if flowspeed models are available
        self.flowspeed_available = False

        # Active model source metadata
        self.model_source_id = None
        self.model_source_label = None
        self.supported_model_codes = ['LR', 'FF', 'GNN']
        self.embeds_network_effects = False

    def _set_active_model_source(self, source_id, source_label, supported_model_codes, embeds_network_effects):
        """Track which output folder is active so the app can adapt its UI and inference path."""
        self.model_source_id = source_id
        self.model_source_label = source_label
        self.supported_model_codes = list(supported_model_codes)
        self.embeds_network_effects = embeds_network_effects

    def _resolve_v3_globalcap_dir(self, folder_path=None):
        """Resolve path to v3_globalcap artifacts root directory.
        
        Now simplified for flat structure: ./model with ./model/models and ./model/data subdirectories.
        No longer checks for nested runs/ layout.
        """
        if folder_path is None or str(folder_path).strip() == '':
            source_root = self.base_path / DEFAULT_V3_GLOBALCAP_DIR
        else:
            source_root = Path(folder_path).expanduser()
            if not source_root.is_absolute():
                source_root = (self.base_path / source_root).resolve()
        
        return source_root.resolve()

    def _resolve_v3_globalcap_artifacts(self, folder_path=None):
        """Resolve v3_globalcap artifact paths.
        
        Expects flat structure: ./model/models and ./model/data subdirectories.
        """
        source_root = self._resolve_v3_globalcap_dir(folder_path)
        return {
            'root': source_root,
            'ff': source_root / 'models' / 'best_ff_flowspeed_v3_globalcap.pth',
            'ff_gui': source_root / 'models' / V3_GUI_FF_CHECKPOINT,
            'artifacts': source_root / 'data' / 'artifacts_flowspeed_v3_globalcap.pkl',
            'scalers': source_root / 'data' / 'scalers_flowspeed_v3_globalcap.pkl',
            'configs': source_root / 'data' / 'model_configs_flowspeed_v3_globalcap.pkl',
        }

    def get_missing_v3_globalcap_files(self, folder_path=None):
        """Return files required for the v3 global-capacity flow+speed output."""
        artifacts = self._resolve_v3_globalcap_artifacts(folder_path)
        file_to_key = {
            'best_ff_flowspeed_v3_globalcap.pth': 'ff',
            'artifacts_flowspeed_v3_globalcap.pkl': 'artifacts',
            'scalers_flowspeed_v3_globalcap.pkl': 'scalers',
            'model_configs_flowspeed_v3_globalcap.pkl': 'configs',
        }
        return [name for name in V3_GLOBALCAP_FILES if not artifacts[file_to_key[name]].exists()]

    @staticmethod
    def _extract_state_dict(checkpoint):
        """Normalize checkpoint payloads to a plain state_dict."""
        if isinstance(checkpoint, dict):
            for key in ('state_dict', 'model_state_dict'):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    return checkpoint[key]
        return checkpoint

    @staticmethod
    def _adapt_ff_globalcap_state_dict(state_dict):
        """Adapt legacy Sequential FF keys (net.*) to named-layer keys (fc*)."""
        expected_fc_keys = {
            'fc1.weight', 'fc1.bias',
            'fc2.weight', 'fc2.bias',
            'fc3.weight', 'fc3.bias',
        }
        if not isinstance(state_dict, dict):
            return state_dict

        # Already in the expected format.
        if expected_fc_keys.issubset(state_dict.keys()):
            return state_dict

        legacy_map = {
            'net.0.weight': 'fc1.weight',
            'net.0.bias': 'fc1.bias',
            'net.3.weight': 'fc2.weight',
            'net.3.bias': 'fc2.bias',
            'net.6.weight': 'fc3.weight',
            'net.6.bias': 'fc3.bias',
        }

        if not all(old_key in state_dict for old_key in legacy_map):
            return state_dict

        adapted = {}
        for key, value in state_dict.items():
            adapted[legacy_map.get(key, key)] = value
        return adapted

    @staticmethod
    def _cache_is_fresh(cache_path, source_path):
        """Return True when a converted GUI checkpoint can be reused."""
        return (
            cache_path.exists()
            and source_path.exists()
            and cache_path.stat().st_mtime >= source_path.stat().st_mtime
        )

    def _load_or_convert_checkpoint(self, source_path, cache_path, adapt_fn=None):
        """Load a GUI-ready checkpoint, writing a converted cache if needed."""
        if self._cache_is_fresh(cache_path, source_path):
            return torch.load(cache_path, map_location=self.device, weights_only=True)

        checkpoint = torch.load(source_path, map_location='cpu', weights_only=True)
        state = self._extract_state_dict(checkpoint)
        if adapt_fn is not None:
            state = adapt_fn(state)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, cache_path)
        print(f"Converted checkpoint for GUI reuse: {cache_path}")
        return state



    def load_v3_globalcap_models(self, folder_path=None):
        """Load the v3 global-capacity flow+speed checkpoints for visualization."""
        if not self.links_info or not self.nodes:
            self.load_network()

        missing_files = self.get_missing_v3_globalcap_files(folder_path)
        if missing_files:
            raise FileNotFoundError(
                'Missing v3 global-capacity files: ' + ', '.join(missing_files)
            )

        resolved = self._resolve_v3_globalcap_artifacts(folder_path)
        ff_path = resolved['ff']
        ff_gui_path = resolved['ff_gui']
        artifacts_path = resolved['artifacts']
        scalers_path = resolved['scalers']
        model_configs_path = resolved['configs']

        with open(artifacts_path, 'rb') as handle:
            artifacts = pickle.load(handle)
        with open(scalers_path, 'rb') as handle:
            scalers = pickle.load(handle)
        with open(model_configs_path, 'rb') as handle:
            model_configs = pickle.load(handle)

        missing_links = sorted(set(artifacts['unique_link_ids']) - set(self.links_info.keys()))
        if missing_links:
            raise ValueError(
                'The network XML is missing links required by the v3 output: ' + ', '.join(missing_links[:5])
            )

        self.static_scaler = scalers['static_scaler']
        self.capacity_scaler = scalers['capacity_scaler']
        self.flow_scaler = scalers['flow_scaler']
        self.speed_scaler = scalers.get('speed_scaler', self.speed_scaler)
        self.static_features_scaled = np.asarray(artifacts['static_features_scaled'])
        self.n_hours = int(artifacts['n_hours'])
        self.DAYTIME_HOURS = list(artifacts['DAYTIME_HOURS'])
        self.unique_link_ids = list(artifacts['unique_link_ids'])
        self.link_to_idx = dict(artifacts['link_to_idx'])
        self.n_links = int(artifacts['n_links'])
        self.n_static_features = int(artifacts['n_static_features'])
        self.edge_index = np.asarray(artifacts['edge_index'])
        self.edge_attr = np.asarray(artifacts['edge_attr'])
        self.upstream_indices = [list(indices) for indices in artifacts.get('upstream_indices', [[] for _ in range(self.n_links)])]
        self.downstream_indices = [list(indices) for indices in artifacts.get('downstream_indices', [[] for _ in range(self.n_links)])]
        self.sample_context_feature_names = list(artifacts.get('sample_context_feature_names', []))

        ff_config = model_configs['ff']
        model_ff_instance = FeedforwardNN_GlobalCapacity(
            input_size=int(ff_config['input_dim']),
            hidden_sizes=tuple(ff_config.get('hidden', (512, 256))),
            dropout=float(ff_config.get('dropout', 0.2)),
            output_dim=int(ff_config.get('output_dim', 1)),
        )
        try:
            model_ff_instance = model_ff_instance.to(self.device)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if 'out of memory' in str(exc).lower():
                print(f"CUDA out of memory – falling back to CPU: {exc}")
                self.device = torch.device('cpu')
                model_ff_instance = model_ff_instance.to(self.device)
            else:
                raise
        self.model_ff = model_ff_instance
        ff_state = self._load_or_convert_checkpoint(
            ff_path,
            ff_gui_path,
            adapt_fn=self._adapt_ff_globalcap_state_dict,
        )
        self.model_ff.load_state_dict(ff_state)
        self.model_ff.eval()


        self.model_ff_flowspeed = self.model_ff if int(ff_config.get('output_dim', 1)) == 2 else None
        self.model_gnn_flowspeed = None
        self.flowspeed_available = self.model_ff_flowspeed is not None
        self._set_active_model_source(
            'v3_globalcap',
            f"v3 global-capacity run ({resolved['root']})",
            ['FF'],
            embeds_network_effects=True,
        )
        return self
        
    def load_network(self):
        """Load network topology from XML file."""
        network_file = self.base_path / DEFAULT_V3_GLOBALCAP_DIR / NETWORK_XML_FILENAME
        tree = ET.parse(network_file)
        root = tree.getroot()
        
        # Extract nodes
        self.nodes = {}
        for node in root.findall('.//node'):
            node_id = node.get('id')
            x = float(node.get('x'))
            y = float(node.get('y'))
            self.nodes[node_id] = {'x': x, 'y': y}
        
        # Extract links
        self.links_info = {}
        self.node_to_incoming_links = defaultdict(list)
        self.node_to_outgoing_links = defaultdict(list)
        
        for link in root.findall('.//link'):
            link_id = link.get('id')
            from_node = link.get('from')
            to_node = link.get('to')
            length = float(link.get('length'))
            freespeed = float(link.get('freespeed'))
            capacity = float(link.get('capacity'))
            permlanes = float(link.get('permlanes'))
            
            self.links_info[link_id] = {
                'from': from_node,
                'to': to_node,
                'length': length,
                'freespeed': freespeed,
                'capacity': capacity,
                'permlanes': permlanes
            }
            
            self.node_to_outgoing_links[from_node].append(link_id)
            self.node_to_incoming_links[to_node].append(link_id)
        
        print(f"Loaded {len(self.nodes)} nodes and {len(self.links_info)} links")
        return self
