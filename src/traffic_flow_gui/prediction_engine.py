"""
Prediction engine for traffic flow and speed models.
Handles predictions with custom capacity values and network effects.
Supports both flow-only and flow+speed prediction modes.
"""

import numpy as np
import torch

DEFAULT_CAPACITY = 3600.0


class PredictionEngine:
    """Handles predictions for all models with custom capacities and network effects.
    
    Supports two modes:
    - Flow-only: Original mode predicting only traffic flow
    - Flow+Speed: Extended mode predicting both flow and average speed
    """
    
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.device = data_loader.device
        self.n_hours = data_loader.n_hours
        self.n_links = data_loader.n_links
        self.DAYTIME_HOURS = data_loader.DAYTIME_HOURS
        
        # Default capacities (3600 for all links)
        self.capacities = {link_id: DEFAULT_CAPACITY for link_id in data_loader.unique_link_ids}
        
        # Cache for predictions (separate for flow and flowspeed)
        self.predictions_cache = {}
        self.predictions_cache_flowspeed = {}
        self.current_model = 'FF'  # Default model
        
        # Check if flow+speed models are available
        self.flowspeed_available = data_loader.flowspeed_available
        
        # Build network adjacency for network effects
        self._build_network_adjacency()
    
    def _build_network_adjacency(self):
        """Build adjacency information for links sharing nodes."""
        self.link_neighbors = {link_id: [] for link_id in self.data_loader.unique_link_ids}
        
        # Find links that share endpoints
        for link_id, link_data in self.data_loader.links_info.items():
            from_node = link_data['from']
            to_node = link_data['to']
            
            for other_link_id, other_link_data in self.data_loader.links_info.items():
                if link_id == other_link_id:
                    continue
                other_from = other_link_data['from']
                other_to = other_link_data['to']
                
                # Links are neighbors if they share a node
                if from_node in (other_from, other_to) or to_node in (other_from, other_to):
                    if other_link_id not in self.link_neighbors[link_id]:
                        self.link_neighbors[link_id].append(other_link_id)
    
    def set_capacity(self, link_id, capacity):
        """Set capacity for a specific link."""
        if link_id in self.capacities:
            self.capacities[link_id] = capacity
            # Clear both caches when capacity changes
            self.predictions_cache = {}
            self.predictions_cache_flowspeed = {}
    
    def reset_capacities(self, default_capacity=DEFAULT_CAPACITY):
        """Reset all capacities to default value."""
        self.capacities = {link_id: default_capacity for link_id in self.data_loader.unique_link_ids}
        self.predictions_cache = {}
        self.predictions_cache_flowspeed = {}
    
    def _apply_network_effects(self, predictions):
        """Apply network effects: adjust neighboring link flows based on capacity changes.
        
        When a link's capacity changes, neighboring links may see flow adjustments.
        This implements a simple spillover effect.
        """
        if self.data_loader.embeds_network_effects:
            return predictions

        # Calculate capacity ratios (how much capacity differs from default)
        capacity_ratios = {}
        for link_id, capacity in self.capacities.items():
            capacity_ratios[link_id] = capacity / DEFAULT_CAPACITY
        
        # Apply network effects iteratively
        adjusted = True
        iterations = 0
        max_iterations = 3
        
        while adjusted and iterations < max_iterations:
            adjusted = False
            iterations += 1
            new_predictions = {}
            
            for link_id in self.data_loader.unique_link_ids:
                new_predictions[link_id] = {}
                
                for hour in self.DAYTIME_HOURS:
                    base_flow = predictions[link_id][hour]
                    
                    # Get neighbor capacity constraints
                    neighbors = self.link_neighbors.get(link_id, [])
                    if not neighbors:
                        new_predictions[link_id][hour] = base_flow
                        continue
                    
                    # Calculate average neighbor capacity ratio
                    neighbor_ratios = [capacity_ratios.get(n, 1.0) for n in neighbors]
                    avg_neighbor_ratio = np.mean(neighbor_ratios) if neighbor_ratios else 1.0
                    
                    # If link has lower capacity than average neighbors, 
                    # slightly increase flow (spillback effect)
                    # If link has higher capacity than neighbors,
                    # slightly decrease flow (capacity not needed)
                    link_ratio = capacity_ratios[link_id]
                    relative_capacity = link_ratio / max(avg_neighbor_ratio, 0.1)
                    
                    # Apply adjustment (±5% max based on relative capacity)
                    adjustment_factor = 1.0 + (relative_capacity - 1.0) * 0.05
                    adjustment_factor = np.clip(adjustment_factor, 0.95, 1.05)
                    
                    adjusted_flow = base_flow * adjustment_factor
                    new_predictions[link_id][hour] = max(0, adjusted_flow)
                    
                    if abs(adjusted_flow - base_flow) > 0.01:
                        adjusted = True
            
            predictions = new_predictions
        
        return predictions
    
    def predict_all(self, model_name='FF'):
        """Predict flow for all links and all hours using specified model.
        
        Args:
            model_name: 'LR', 'FF', or 'GNN'
        
        Returns:
            dict: {link_id: {hour: predicted_flow}}
        """
        self.current_model = model_name

        if self.data_loader.model_source_id == 'v3_globalcap' and model_name != 'FF':
            raise ValueError("Only the feedforward model is available in this GUI build")

        if model_name not in self.data_loader.supported_model_codes:
            raise ValueError(
                f"Model '{model_name}' is not available for source {self.data_loader.model_source_label}"
            )
        
        # Check cache
        cache_key = (model_name, tuple(sorted(self.capacities.items())))
        if cache_key in self.predictions_cache:
            return self.predictions_cache[cache_key]
        
        predictions = {}
        
        if model_name == 'FF':
            predictions = self._predict_ff()
        
        # Apply network effects to propagate capacity changes through network
        predictions = self._apply_network_effects(predictions)
        
        self.predictions_cache[cache_key] = predictions
        return predictions
    
    def _predict_ff(self):
        """Predict using Feedforward NN model."""
        if self.data_loader.model_source_id == 'v3_globalcap':
            return self._predict_ff_global_capacity()

        predictions = {link_id: {} for link_id in self.data_loader.unique_link_ids}
        
        # Build input tensors for all links and hours
        X_all = []
        link_hour_pairs = []
        
        for link_id in self.data_loader.unique_link_ids:
            link_idx = self.data_loader.link_to_idx[link_id]
            capacity = self.capacities[link_id]
            capacity_scaled = self.data_loader.capacity_scaler.transform([[capacity]])[0, 0]
            link_static = self.data_loader.static_features_scaled[link_idx]
            
            for hour_idx, hour in enumerate(self.DAYTIME_HOURS):
                hour_one_hot = np.zeros(self.n_hours)
                hour_one_hot[hour_idx] = 1
                
                x_sample = np.concatenate([hour_one_hot, link_static, [capacity_scaled]])
                X_all.append(x_sample)
                link_hour_pairs.append((link_id, hour))
        
        X_all = np.array(X_all)
        X_tensor = torch.FloatTensor(X_all).to(self.device)
        
        # Predict
        self.data_loader.model_ff.eval()
        with torch.no_grad():
            y_pred = self.data_loader.model_ff(X_tensor).cpu().numpy()
        
        # Convert to original scale
        y_pred_orig = self.data_loader.flow_scaler.inverse_transform(y_pred)
        
        # Organize predictions
        for i, (link_id, hour) in enumerate(link_hour_pairs):
            predictions[link_id][hour] = max(0, y_pred_orig[i, 0])
        
        return predictions

    def _predict_ff_global_capacity(self):
        """Predict using the v3 feedforward model with full-network capacity context."""
        predictions = {link_id: {} for link_id in self.data_loader.unique_link_ids}

        capacity_values = np.array(
            [self.capacities[link_id] for link_id in self.data_loader.unique_link_ids],
            dtype=np.float32,
        )
        cap_scaled = self.data_loader.capacity_scaler.transform(
            capacity_values.reshape(-1, 1)
        ).flatten().astype(np.float32)
        sample_context = self._v3_sample_context_features(cap_scaled)
        static_features = self.data_loader.static_features_scaled.astype(np.float32)

        self.data_loader.model_ff.eval()
        for hour_idx, hour in enumerate(self.DAYTIME_HOURS):
            hour_one_hot = np.zeros(self.n_hours, dtype=np.float32)
            hour_one_hot[hour_idx] = 1.0
            x_hour = np.hstack(
                [
                    np.tile(hour_one_hot, (self.n_links, 1)),
                    static_features,
                        sample_context,
                    np.tile(cap_scaled.reshape(1, -1), (self.n_links, 1)),
                ]
            ).astype(np.float32)

            with torch.no_grad():
                y_pred = self.data_loader.model_ff(
                    torch.FloatTensor(x_hour).to(self.device)
                ).cpu().numpy()

            flow_scaled = y_pred[:, 0] if y_pred.ndim == 2 else y_pred.reshape(-1)
            y_pred_orig = self.data_loader.flow_scaler.inverse_transform(flow_scaled.reshape(-1, 1)).flatten()
            for link_id, flow_value in zip(self.data_loader.unique_link_ids, y_pred_orig):
                predictions[link_id][hour] = max(0, float(flow_value))

        return predictions

    def _v3_sample_context_features(self, cap_scaled):
        """Build the local/upstream/downstream capacity features used by v3 training."""
        cap_scaled = np.asarray(cap_scaled, dtype=np.float32).reshape(-1)
        upstream_avg = np.empty(self.n_links, dtype=np.float32)
        downstream_avg = np.empty(self.n_links, dtype=np.float32)

        for link_idx in range(self.n_links):
            upstream = self.data_loader.upstream_indices[link_idx] if self.data_loader.upstream_indices else []
            downstream = self.data_loader.downstream_indices[link_idx] if self.data_loader.downstream_indices else []
            upstream_avg[link_idx] = float(np.mean(cap_scaled[upstream])) if upstream else cap_scaled[link_idx]
            downstream_avg[link_idx] = float(np.mean(cap_scaled[downstream])) if downstream else cap_scaled[link_idx]

        return np.column_stack(
            [
                cap_scaled,
                upstream_avg,
                downstream_avg,
                cap_scaled - upstream_avg,
                cap_scaled - downstream_avg,
            ]
        ).astype(np.float32)

    
    def get_flow_for_hour(self, hour, model_name='FF'):
        """Get predicted flow for all links at a specific hour.
        
        Returns:
            dict: {link_id: predicted_flow}
        """
        all_predictions = self.predict_all(model_name)
        return {link_id: preds[hour] for link_id, preds in all_predictions.items()}
    
    def get_flow_statistics(self, model_name='FF'):
        """Get statistics about predicted flows.
        
        Returns:
            dict: {'min': float, 'max': float, 'mean': float}
        """
        all_predictions = self.predict_all(model_name)
        
        all_flows = []
        for link_preds in all_predictions.values():
            all_flows.extend(link_preds.values())
        
        return {
            'min': min(all_flows),
            'max': max(all_flows),
            'mean': np.mean(all_flows)
        }

    # ========================================================================
    # FLOW + SPEED PREDICTION METHODS
    # ========================================================================
    
    def predict_all_flowspeed(self, model_name='FF'):
        """Predict flow and speed for all links and all hours using specified model.
        
        Args:
            model_name: 'LR', 'FF', or 'GNN'
        
        Returns:
            tuple: (flow_predictions, speed_predictions)
                   Each is dict: {link_id: {hour: predicted_value}}
        """
        if not self.flowspeed_available:
            raise RuntimeError("Flow+Speed models are not available")
        if self.data_loader.model_source_id == 'v3_globalcap' and model_name != 'FF':
            raise ValueError("Only the feedforward model is available in this GUI build")
        if self.data_loader.model_source_id == 'v3_globalcap':
            return self._predict_all_v3_globalcap_flowspeed(model_name)
        
        self.current_model = model_name
        
        # Check cache
        cache_key = (model_name, tuple(sorted(self.capacities.items())))
        if cache_key in self.predictions_cache_flowspeed:
            return self.predictions_cache_flowspeed[cache_key]
        
        flow_predictions = {}
        speed_predictions = {}
        
        if model_name == 'FF':
            flow_predictions, speed_predictions = self._predict_ff_flowspeed()
        
        # Apply network effects (only to flow, speed is independent)
        flow_predictions = self._apply_network_effects(flow_predictions)
        
        self.predictions_cache_flowspeed[cache_key] = (flow_predictions, speed_predictions)
        return flow_predictions, speed_predictions

    def _inverse_flowspeed(self, y_pred):
        flow = self.data_loader.flow_scaler.inverse_transform(y_pred[:, 0].reshape(-1, 1)).flatten()
        speed = self.data_loader.speed_scaler.inverse_transform(y_pred[:, 1].reshape(-1, 1)).flatten()
        return flow, speed

    def _predict_all_v3_globalcap_flowspeed(self, model_name='FF'):
        if model_name != 'FF':
            raise ValueError("This GUI build supports only FF for v3 global-capacity flow+speed output")

        cache_key = (model_name, tuple(sorted(self.capacities.items())))
        if cache_key in self.predictions_cache_flowspeed:
            return self.predictions_cache_flowspeed[cache_key]

        flow_predictions = {link_id: {} for link_id in self.data_loader.unique_link_ids}
        speed_predictions = {link_id: {} for link_id in self.data_loader.unique_link_ids}
        capacity_values = np.array(
            [self.capacities[link_id] for link_id in self.data_loader.unique_link_ids],
            dtype=np.float32,
        )
        cap_scaled = self.data_loader.capacity_scaler.transform(
            capacity_values.reshape(-1, 1)
        ).flatten().astype(np.float32)
        sample_context = self._v3_sample_context_features(cap_scaled)

        if model_name == 'FF':
            static_features = self.data_loader.static_features_scaled.astype(np.float32)
            self.data_loader.model_ff.eval()
            for hour_idx, hour in enumerate(self.DAYTIME_HOURS):
                hour_one_hot = np.zeros(self.n_hours, dtype=np.float32)
                hour_one_hot[hour_idx] = 1.0
                x_hour = np.hstack(
                    [
                        np.tile(hour_one_hot, (self.n_links, 1)),
                        static_features,
                        sample_context,
                        np.tile(cap_scaled.reshape(1, -1), (self.n_links, 1)),
                    ]
                ).astype(np.float32)
                with torch.no_grad():
                    y_pred = self.data_loader.model_ff(
                        torch.FloatTensor(x_hour).to(self.device)
                    ).cpu().numpy()
                flow_orig, speed_orig = self._inverse_flowspeed(y_pred)
                for link_id, flow_value, speed_value in zip(self.data_loader.unique_link_ids, flow_orig, speed_orig):
                    flow_predictions[link_id][hour] = max(0, float(flow_value))
                    speed_predictions[link_id][hour] = max(0, float(speed_value))
        self.predictions_cache_flowspeed[cache_key] = (flow_predictions, speed_predictions)
        return flow_predictions, speed_predictions
    
    def _predict_ff_flowspeed(self):
        """Predict using Feedforward NN model (flow+speed)."""
        flow_predictions = {link_id: {} for link_id in self.data_loader.unique_link_ids}
        speed_predictions = {link_id: {} for link_id in self.data_loader.unique_link_ids}
        
        # Build input tensors for all links and hours
        X_all = []
        link_hour_pairs = []
        
        for link_id in self.data_loader.unique_link_ids:
            link_idx = self.data_loader.link_to_idx[link_id]
            capacity = self.capacities[link_id]
            capacity_scaled = self.data_loader.capacity_scaler.transform([[capacity]])[0, 0]
            link_static = self.data_loader.static_features_scaled[link_idx]
            
            for hour_idx, hour in enumerate(self.DAYTIME_HOURS):
                hour_one_hot = np.zeros(self.n_hours)
                hour_one_hot[hour_idx] = 1
                
                x_sample = np.concatenate([hour_one_hot, link_static, [capacity_scaled]])
                X_all.append(x_sample)
                link_hour_pairs.append((link_id, hour))
        
        X_all = np.array(X_all)
        X_tensor = torch.FloatTensor(X_all).to(self.device)
        
        # Predict
        self.data_loader.model_ff_flowspeed.eval()
        with torch.no_grad():
            y_pred = self.data_loader.model_ff_flowspeed(X_tensor).cpu().numpy()
        
        # Convert to original scale
        y_pred_flow_orig = self.data_loader.flow_scaler.inverse_transform(y_pred[:, 0].reshape(-1, 1))
        y_pred_speed_orig = self.data_loader.speed_scaler.inverse_transform(y_pred[:, 1].reshape(-1, 1))
        
        # Organize predictions
        for i, (link_id, hour) in enumerate(link_hour_pairs):
            flow_predictions[link_id][hour] = max(0, y_pred_flow_orig[i, 0])
            speed_predictions[link_id][hour] = max(0, y_pred_speed_orig[i, 0])
        
        return flow_predictions, speed_predictions
    
    def get_speed_for_hour(self, hour, model_name='FF'):
        """Get predicted speed for all links at a specific hour.
        
        Returns:
            dict: {link_id: predicted_speed}
        """
        _, speed_predictions = self.predict_all_flowspeed(model_name)
        return {link_id: preds[hour] for link_id, preds in speed_predictions.items()}
    
    def get_speed_statistics(self, model_name='FF'):
        """Get statistics about predicted speeds.
        
        Returns:
            dict: {'min': float, 'max': float, 'mean': float}
        """
        _, speed_predictions = self.predict_all_flowspeed(model_name)
        
        all_speeds = []
        for link_preds in speed_predictions.values():
            all_speeds.extend(link_preds.values())
        
        return {
            'min': min(all_speeds),
            'max': max(all_speeds),
            'mean': np.mean(all_speeds)
        }
