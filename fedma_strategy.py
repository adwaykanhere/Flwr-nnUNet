# fedma_strategy.py

"""
FedMA (Federated Matched Averaging) implementation for nnUNet architecture harmonization.
Handles heterogeneous architectures with different input channels and output classes.

Based on research:
- Wang et al. "Federated Learning with Matched Averaging" (ICLR 2020)
- Zhu et al. "Federated Learning on Non-IID Data Silos" (INFOCOM 2021)
- Medical federated learning harmonization techniques
"""

import os
import json
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
from flwr.common import NDArrays, Parameters, parameters_to_ndarrays, ndarrays_to_parameters

import warnings
warnings.filterwarnings("ignore")


class LayerMatcher:
    """
    Core FedMA layer matching algorithm using activation-based similarity.
    Implements optimal neuron/filter matching across clients with different architectures.
    """
    
    def __init__(self, matching_method: str = "cosine"):
        """
        Initialize layer matcher.
        
        Args:
            matching_method: "cosine", "euclidean", or "activation_based"
        """
        self.matching_method = matching_method
        
    def compute_layer_similarity(self, layer1: np.ndarray, layer2: np.ndarray) -> np.ndarray:
        """
        Compute similarity matrix between neurons/filters in two layers.
        
        Args:
            layer1: First layer weights [out_features, in_features, ...]
            layer2: Second layer weights [out_features, in_features, ...]
            
        Returns:
            Similarity matrix [out1, out2]
        """
        # Handle 1D tensors (bias parameters) - cannot compute meaningful similarity
        if len(layer1.shape) == 1 or len(layer2.shape) == 1:
            # For 1D tensors, create identity similarity matrix if same size, otherwise zeros
            size1, size2 = layer1.shape[0], layer2.shape[0]
            min_size = min(size1, size2)
            similarity = np.zeros((size1, size2))
            # Set diagonal to 1.0 for matching indices
            np.fill_diagonal(similarity[:min_size, :min_size], 1.0)
            return similarity
        
        # Flatten spatial dimensions for conv layers (2D and higher)
        if len(layer1.shape) > 2:
            layer1_flat = layer1.reshape(layer1.shape[0], -1)  # [out_features, flattened]
        else:
            layer1_flat = layer1
            
        if len(layer2.shape) > 2:
            layer2_flat = layer2.reshape(layer2.shape[0], -1)
        else:
            layer2_flat = layer2
        
        # Check if flattened layers are compatible for similarity computation
        if layer1_flat.shape[1] != layer2_flat.shape[1]:
            print(f"[FedMA] Incompatible feature dimensions {layer1_flat.shape[1]} vs {layer2_flat.shape[1]}, using adaptive similarity")
            # Use adaptive similarity for incompatible dimensions
            return self._compute_adaptive_similarity(layer1_flat, layer2_flat)
        
        try:
            if self.matching_method == "cosine":
                # Normalize rows
                layer1_norm = layer1_flat / (np.linalg.norm(layer1_flat, axis=1, keepdims=True) + 1e-8)
                layer2_norm = layer2_flat / (np.linalg.norm(layer2_flat, axis=1, keepdims=True) + 1e-8)
                
                # Compute cosine similarity
                similarity = np.dot(layer1_norm, layer2_norm.T)
                
            elif self.matching_method == "euclidean":
                # Compute negative euclidean distance (higher = more similar)
                # Use memory-efficient computation for large tensors
                if layer1_flat.shape[0] * layer2_flat.shape[0] > 10000:
                    # For large matrices, compute in chunks to avoid memory issues
                    similarity = np.zeros((layer1_flat.shape[0], layer2_flat.shape[0]))
                    chunk_size = 100
                    for i in range(0, layer1_flat.shape[0], chunk_size):
                        end_i = min(i + chunk_size, layer1_flat.shape[0])
                        chunk_dist = -np.linalg.norm(
                            layer1_flat[i:end_i, None, :] - layer2_flat[None, :, :], 
                            axis=2
                        )
                        similarity[i:end_i, :] = chunk_dist
                else:
                    similarity = -np.linalg.norm(
                        layer1_flat[:, None, :] - layer2_flat[None, :, :], 
                        axis=2
                    )
                    
        except Exception as e:
            print(f"[FedMA] Error computing similarity: {e}, using fallback")
            # Fallback: identity-based similarity
            size1, size2 = layer1_flat.shape[0], layer2_flat.shape[0]
            similarity = np.zeros((size1, size2))
            min_size = min(size1, size2)
            np.fill_diagonal(similarity[:min_size, :min_size], 1.0)
            
        else:
            raise ValueError(f"Unknown matching method: {self.matching_method}")
        
        return similarity
    
    def _compute_adaptive_similarity(self, layer1_flat: np.ndarray, layer2_flat: np.ndarray) -> np.ndarray:
        """
        Compute adaptive similarity for layers with incompatible feature dimensions.
        Uses statistical properties and partial feature matching.
        """
        size1, size2 = layer1_flat.shape[0], layer2_flat.shape[0]
        feat1, feat2 = layer1_flat.shape[1], layer2_flat.shape[1]
        
        # Compute statistical features for each neuron/filter
        stats1 = self._compute_statistical_features(layer1_flat)
        stats2 = self._compute_statistical_features(layer2_flat)
        
        # Compute similarity based on statistical features
        similarity = np.zeros((size1, size2))
        
        for i in range(size1):
            for j in range(size2):
                # Combine multiple similarity measures
                stat_sim = self._statistical_similarity(stats1[i], stats2[j])
                
                # Add partial weight matching for overlapping dimensions
                min_feat = min(feat1, feat2)
                if min_feat > 0:
                    partial_weights1 = layer1_flat[i, :min_feat]
                    partial_weights2 = layer2_flat[j, :min_feat]
                    weight_sim = self._compute_weight_similarity(partial_weights1, partial_weights2)
                else:
                    weight_sim = 0.0
                
                # Combine similarities
                similarity[i, j] = 0.7 * stat_sim + 0.3 * weight_sim
        
        return similarity
    
    def _compute_statistical_features(self, layer_weights: np.ndarray) -> np.ndarray:
        """Compute statistical features for each neuron/filter."""
        features = []
        for i in range(layer_weights.shape[0]):
            weights = layer_weights[i]
            stats = [
                np.mean(weights),           # Mean
                np.std(weights),            # Standard deviation
                np.median(weights),         # Median
                np.max(weights),            # Maximum
                np.min(weights),            # Minimum
                np.percentile(weights, 75), # 75th percentile
                np.percentile(weights, 25), # 25th percentile
                np.mean(np.abs(weights)),   # Mean absolute value
            ]
            features.append(stats)
        return np.array(features)
    
    def _statistical_similarity(self, stats1: np.ndarray, stats2: np.ndarray) -> float:
        """Compute similarity based on statistical features."""
        # Normalize features to prevent scale issues
        norm1 = stats1 / (np.linalg.norm(stats1) + 1e-8)
        norm2 = stats2 / (np.linalg.norm(stats2) + 1e-8)
        
        # Compute cosine similarity
        similarity = np.dot(norm1, norm2)
        return max(0.0, similarity)  # Clamp to positive
    
    def _compute_weight_similarity(self, weights1: np.ndarray, weights2: np.ndarray) -> float:
        """Compute similarity between weight vectors of same dimension."""
        if len(weights1) == 0 or len(weights2) == 0:
            return 0.0
        
        # Normalize weights
        norm1 = weights1 / (np.linalg.norm(weights1) + 1e-8)
        norm2 = weights2 / (np.linalg.norm(weights2) + 1e-8)
        
        # Compute cosine similarity
        similarity = np.dot(norm1, norm2)
        return max(0.0, similarity)  # Clamp to positive
    
    def find_optimal_matching(self, similarity_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find optimal matching using Hungarian algorithm.
        
        Args:
            similarity_matrix: [n1, n2] similarity scores
            
        Returns:
            (indices1, indices2): Optimal matching indices
        """
        # Hungarian algorithm minimizes cost, so negate similarity
        cost_matrix = -similarity_matrix
        
        # Handle different sized layers
        if cost_matrix.shape[0] != cost_matrix.shape[1]:
            max_size = max(cost_matrix.shape)
            padded_cost = np.full((max_size, max_size), cost_matrix.max() + 1)
            padded_cost[:cost_matrix.shape[0], :cost_matrix.shape[1]] = cost_matrix
            cost_matrix = padded_cost
        
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Filter out padding indices
        valid_mask = (row_indices < similarity_matrix.shape[0]) & (col_indices < similarity_matrix.shape[1])
        row_indices = row_indices[valid_mask]
        col_indices = col_indices[valid_mask]
        
        return row_indices, col_indices
    
    def match_layers(self, layers: List[np.ndarray]) -> List[List[int]]:
        """
        Match neurons/filters across multiple layers using FedMA.
        
        Args:
            layers: List of layer weights from different clients
            
        Returns:
            List of matching indices for each client
        """
        if len(layers) < 2:
            return [list(range(layer.shape[0])) for layer in layers]
        
        # Use first layer as reference
        reference_layer = layers[0]
        matchings = [list(range(reference_layer.shape[0]))]  # Reference keeps original order
        
        for i, layer in enumerate(layers[1:], 1):
            # Compute similarity with reference
            similarity = self.compute_layer_similarity(reference_layer, layer)
            
            # Find optimal matching
            ref_indices, layer_indices = self.find_optimal_matching(similarity)
            
            # Create full matching (handle unmatched neurons)
            full_matching = [-1] * layer.shape[0]  # -1 for unmatched
            for ref_idx, layer_idx in zip(ref_indices, layer_indices):
                if ref_idx < len(full_matching):
                    full_matching[layer_idx] = ref_idx
            
            matchings.append(full_matching)
        
        return matchings


class ChannelHarmonizer:
    """
    Handles input channel harmonization for different modalities.
    Supports padding, projection, and learned transformations.
    """
    
    def __init__(self, harmonization_method: str = "padding"):
        """
        Initialize channel harmonizer.
        
        Args:
            harmonization_method: "padding", "projection", "learned"
        """
        self.harmonization_method = harmonization_method
        
    def harmonize_input_channels(self, 
                                input_layers: List[np.ndarray], 
                                target_channels: Optional[int] = None) -> List[np.ndarray]:
        """
        Harmonize input layers with different channel counts.
        
        Args:
            input_layers: List of input layer weights [out, in_channels, ...]
            target_channels: Target channel count (if None, use max)
            
        Returns:
            List of harmonized input layers
        """
        if not input_layers:
            return input_layers
        
        # Determine target channel count
        channel_counts = [layer.shape[1] for layer in input_layers]
        if target_channels is None:
            target_channels = max(channel_counts)
        
        harmonized_layers = []
        
        for layer in input_layers:
            current_channels = layer.shape[1]
            
            if current_channels == target_channels:
                # Already compatible
                harmonized_layers.append(layer)
                
            elif current_channels < target_channels:
                # Need to expand channels
                harmonized_layer = self._expand_channels(layer, target_channels)
                harmonized_layers.append(harmonized_layer)
                
            else:
                # Need to reduce channels
                harmonized_layer = self._reduce_channels(layer, target_channels)
                harmonized_layers.append(harmonized_layer)
        
        return harmonized_layers
    
    def _expand_channels(self, layer: np.ndarray, target_channels: int) -> np.ndarray:
        """Expand layer to have more input channels with intelligent strategies."""
        current_channels = layer.shape[1]
        missing_channels = target_channels - current_channels
        
        if missing_channels <= 0:
            return layer
        
        if self.harmonization_method == "padding":
            # Smart padding: use small random values instead of zeros to avoid dead neurons
            padding_shape = (layer.shape[0], missing_channels) + layer.shape[2:]
            # Use small random values with same scale as existing weights
            existing_std = np.std(layer) if np.std(layer) > 0 else 1e-4
            padding = np.random.normal(0, existing_std * 0.1, padding_shape).astype(layer.dtype)
            harmonized_layer = np.concatenate([layer, padding], axis=1)
            
        elif self.harmonization_method == "projection":
            # Intelligent channel replication with similarity-based selection
            harmonized_layer = self._intelligent_channel_expansion(layer, target_channels)
            
        else:
            raise ValueError(f"Unknown harmonization method: {self.harmonization_method}")
        
        return harmonized_layer
    
    def _intelligent_channel_expansion(self, layer: np.ndarray, target_channels: int) -> np.ndarray:
        """Expand channels using intelligent replication based on channel similarity."""
        current_channels = layer.shape[1]
        missing_channels = target_channels - current_channels
        
        if missing_channels <= 0:
            return layer
        
        # Compute channel similarities to find best candidates for replication
        channel_similarities = np.zeros((current_channels, current_channels))
        for i in range(current_channels):
            for j in range(current_channels):
                if i != j:
                    # Compute cosine similarity between channels
                    ch1 = layer[:, i].flatten()
                    ch2 = layer[:, j].flatten()
                    norm1, norm2 = np.linalg.norm(ch1), np.linalg.norm(ch2)
                    if norm1 > 0 and norm2 > 0:
                        similarity = np.dot(ch1, ch2) / (norm1 * norm2)
                        channel_similarities[i, j] = abs(similarity)
        
        # Select channels to replicate based on diversity (prefer channels that are different)
        replicated_channels = []
        channel_usage = np.zeros(current_channels)
        
        for _ in range(missing_channels):
            # Find least used channel that's most different from already selected
            if len(replicated_channels) == 0:
                # First replication: choose channel with median norm (not too small, not too large)
                channel_norms = [np.linalg.norm(layer[:, i]) for i in range(current_channels)]
                median_norm = np.median(channel_norms)
                best_channel = np.argmin([abs(norm - median_norm) for norm in channel_norms])
            else:
                # Subsequent replications: choose most diverse channel
                diversity_scores = []
                for i in range(current_channels):
                    # Lower similarity to already replicated channels = higher diversity
                    avg_similarity = np.mean([channel_similarities[i, ch] for ch in replicated_channels])
                    usage_penalty = channel_usage[i] * 0.1  # Penalize overused channels
                    diversity_score = 1.0 - avg_similarity - usage_penalty
                    diversity_scores.append(diversity_score)
                best_channel = np.argmax(diversity_scores)
            
            replicated_channels.append(best_channel)
            channel_usage[best_channel] += 1
        
        # Create replicated channels with slight noise to avoid exact duplicates
        expanded_channels = []
        for ch_idx in replicated_channels:
            original_channel = layer[:, ch_idx:ch_idx+1]  # Keep original shape
            # Add small amount of noise to avoid exact duplication
            noise_scale = np.std(original_channel) * 0.05 if np.std(original_channel) > 0 else 1e-5
            noise = np.random.normal(0, noise_scale, original_channel.shape).astype(layer.dtype)
            replicated_channel = original_channel + noise
            expanded_channels.append(replicated_channel)
        
        # Concatenate all channels
        all_channels = [layer] + expanded_channels
        harmonized_layer = np.concatenate(all_channels, axis=1)
        
        return harmonized_layer
    
    def _reduce_channels(self, layer: np.ndarray, target_channels: int) -> np.ndarray:
        """Reduce layer to have fewer input channels using intelligent selection."""
        current_channels = layer.shape[1]
        
        if target_channels >= current_channels:
            return layer
        
        if self.harmonization_method == "padding":
            # Intelligent channel selection instead of simple truncation
            selected_channels = self._select_most_important_channels(layer, target_channels)
            harmonized_layer = layer[:, selected_channels]
            
        elif self.harmonization_method == "projection":
            # Improved weighted combination with clustering
            harmonized_layer = self._cluster_and_combine_channels(layer, target_channels)
                    
        else:
            raise ValueError(f"Unknown harmonization method: {self.harmonization_method}")
        
        return harmonized_layer
    
    def _select_most_important_channels(self, layer: np.ndarray, target_channels: int) -> np.ndarray:
        """Select most important channels based on magnitude and diversity."""
        current_channels = layer.shape[1]
        
        # Compute channel importance scores
        channel_scores = []
        for i in range(current_channels):
            channel = layer[:, i]
            # Combine magnitude and variance as importance measure
            magnitude = np.linalg.norm(channel)
            variance = np.var(channel)
            importance = magnitude * (1 + variance)  # Prefer channels with high magnitude and variance
            channel_scores.append((importance, i))
        
        # Sort by importance and select top channels
        channel_scores.sort(reverse=True)
        selected_indices = [idx for _, idx in channel_scores[:target_channels]]
        selected_indices.sort()  # Maintain original order
        
        return np.array(selected_indices)
    
    def _cluster_and_combine_channels(self, layer: np.ndarray, target_channels: int) -> np.ndarray:
        """Reduce channels by clustering similar channels and combining them."""
        current_channels = layer.shape[1]
        
        # Simple clustering: divide channels into target_channels groups
        channels_per_group = current_channels // target_channels
        remainder = current_channels % target_channels
        
        harmonized_layer = np.zeros((layer.shape[0], target_channels) + layer.shape[2:], dtype=layer.dtype)
        
        current_idx = 0
        for target_idx in range(target_channels):
            # Determine group size (distribute remainder evenly)
            group_size = channels_per_group + (1 if target_idx < remainder else 0)
            end_idx = current_idx + group_size
            
            # Combine channels in this group using weighted average
            if end_idx <= current_channels:
                group_channels = layer[:, current_idx:end_idx]
                
                # Weight channels by their magnitude
                channel_weights = []
                for ch_idx in range(group_size):
                    weight = np.linalg.norm(group_channels[:, ch_idx])
                    channel_weights.append(weight)
                
                # Normalize weights
                total_weight = sum(channel_weights)
                if total_weight > 0:
                    channel_weights = [w / total_weight for w in channel_weights]
                else:
                    channel_weights = [1.0 / group_size] * group_size
                
                # Compute weighted combination
                combined_channel = np.zeros_like(group_channels[:, 0])
                for ch_idx, weight in enumerate(channel_weights):
                    combined_channel += weight * group_channels[:, ch_idx]
                
                harmonized_layer[:, target_idx] = combined_channel
            
            current_idx = end_idx
        
        return harmonized_layer


class ClassHarmonizer:
    """
    Handles output class harmonization for different label spaces.
    Supports label mapping, progressive expansion, and class weighting.
    """
    
    def __init__(self, harmonization_method: str = "union"):
        """
        Initialize class harmonizer.
        
        Args:
            harmonization_method: "union", "intersection", "progressive"
        """
        self.harmonization_method = harmonization_method
        self.global_label_mapping = None
        
    def create_global_label_mapping(self, client_label_spaces: Dict[str, Dict[int, str]]) -> Dict[int, str]:
        """
        Create unified global label space from client label spaces.
        
        Args:
            client_label_spaces: {client_id: {label_id: label_name}}
            
        Returns:
            Global label mapping {global_label_id: label_name}
        """
        if self.harmonization_method == "union":
            # Include all unique labels
            all_labels = set()
            for client_labels in client_label_spaces.values():
                all_labels.update(client_labels.values())
            
            # Create global mapping (background always 0)
            global_mapping = {0: "background"}
            label_id = 1
            for label_name in sorted(all_labels):
                if label_name != "background":
                    global_mapping[label_id] = label_name
                    label_id += 1
                    
        elif self.harmonization_method == "intersection":
            # Only common labels across all clients
            common_labels = None
            for client_labels in client_label_spaces.values():
                client_label_set = set(client_labels.values())
                if common_labels is None:
                    common_labels = client_label_set
                else:
                    common_labels = common_labels.intersection(client_label_set)
            
            global_mapping = {0: "background"}
            label_id = 1
            for label_name in sorted(common_labels):
                if label_name != "background":
                    global_mapping[label_id] = label_name
                    label_id += 1
                    
        else:
            raise ValueError(f"Unknown harmonization method: {self.harmonization_method}")
        
        self.global_label_mapping = global_mapping
        return global_mapping
    
    def harmonize_output_layers(self, 
                               output_layers: List[np.ndarray],
                               client_label_mappings: List[Dict[int, str]]) -> List[np.ndarray]:
        """
        Harmonize output layers with different class counts.
        
        Args:
            output_layers: List of output layer weights [out_classes, ...]
            client_label_mappings: List of label mappings for each client
            
        Returns:
            List of harmonized output layers
        """
        if not self.global_label_mapping:
            # Create global mapping from client mappings
            client_mappings_dict = {f"client_{i}": mapping for i, mapping in enumerate(client_label_mappings)}
            self.create_global_label_mapping(client_mappings_dict)
        
        global_classes = len(self.global_label_mapping)
        harmonized_layers = []
        
        for layer, client_mapping in zip(output_layers, client_label_mappings):
            harmonized_layer = self._harmonize_single_output_layer(layer, client_mapping, global_classes)
            harmonized_layers.append(harmonized_layer)
        
        return harmonized_layers
    
    def _harmonize_single_output_layer(self, 
                                     layer: np.ndarray, 
                                     client_mapping: Dict[int, str], 
                                     global_classes: int) -> np.ndarray:
        """Harmonize a single output layer to global class space."""
        try:
            current_classes = layer.shape[0]
            
            # Handle case where layer has more than 2 dimensions (e.g., conv output layers)
            if len(layer.shape) > 2:
                # For conv layers, we only harmonize the first dimension (output channels/classes)
                if current_classes >= global_classes:
                    # Truncate if current has more classes
                    harmonized_layer = layer[:global_classes]
                else:
                    # Pad if current has fewer classes
                    pad_shape = list(layer.shape)
                    pad_shape[0] = global_classes - current_classes
                    padding = np.zeros(pad_shape, dtype=layer.dtype)
                    harmonized_layer = np.concatenate([layer, padding], axis=0)
                return harmonized_layer
            
            # Handle 2D layers (standard output layers)
            layer_shape = (global_classes,) + layer.shape[1:]
            harmonized_layer = np.zeros(layer_shape, dtype=layer.dtype)
            
            # Map client classes to global classes
            for client_class_id, class_name in client_mapping.items():
                if client_class_id < current_classes:
                    # Find corresponding global class
                    global_class_id = None
                    for global_id, global_name in self.global_label_mapping.items():
                        if global_name == class_name:
                            global_class_id = global_id
                            break
                    
                    if global_class_id is not None and global_class_id < global_classes:
                        harmonized_layer[global_class_id] = layer[client_class_id]
            
            return harmonized_layer
            
        except Exception as e:
            print(f"[FedMA] Error harmonizing output layer: {e}")
            # Fallback: simple size-based harmonization
            current_classes = layer.shape[0]
            if current_classes == global_classes:
                return layer
            elif current_classes > global_classes:
                # Truncate
                return layer[:global_classes]
            else:
                # Pad with zeros
                pad_shape = list(layer.shape)
                pad_shape[0] = global_classes - current_classes
                padding = np.zeros(pad_shape, dtype=layer.dtype)
                return np.concatenate([layer, padding], axis=0)


class FedMAStrategy:
    """
    Main FedMA strategy that combines layer matching with channel/class harmonization.
    Integrates with existing ModalityAwareFederatedStrategy.
    """
    
    def __init__(self,
                 enable_layer_matching: bool = True,
                 enable_channel_harmonization: bool = True,
                 enable_class_harmonization: bool = True,
                 matching_method: str = "cosine",
                 channel_method: str = "padding",
                 class_method: str = "union"):
        """
        Initialize FedMA strategy.
        
        Args:
            enable_layer_matching: Enable FedMA layer matching
            enable_channel_harmonization: Enable input channel harmonization
            enable_class_harmonization: Enable output class harmonization
            matching_method: Method for layer matching
            channel_method: Method for channel harmonization
            class_method: Method for class harmonization
        """
        self.enable_layer_matching = enable_layer_matching
        self.enable_channel_harmonization = enable_channel_harmonization
        self.enable_class_harmonization = enable_class_harmonization
        
        self.layer_matcher = LayerMatcher(matching_method)
        self.channel_harmonizer = ChannelHarmonizer(channel_method)
        self.class_harmonizer = ClassHarmonizer(class_method)
        
        # Track client architectures
        self.client_architectures: Dict[str, Dict] = {}
        self.harmonization_cache: Dict[str, Any] = {}
        
        print(f"[FedMA] Initialized with:")
        print(f"  Layer matching: {enable_layer_matching}")
        print(f"  Channel harmonization: {enable_channel_harmonization}")
        print(f"  Class harmonization: {enable_class_harmonization}")
    
    def register_client_architecture(self, client_id: str, architecture: Dict):
        """Register client architecture information."""
        self.client_architectures[client_id] = architecture
        print(f"[FedMA] Registered architecture for client {client_id}: {architecture}")
    
    def fedma_aggregate(self, 
                       client_parameters_list: List[NDArrays],
                       client_weights: List[float],
                       client_architectures: List[Dict]) -> NDArrays:
        """
        Perform FedMA aggregation with architecture harmonization.
        
        Args:
            client_parameters_list: List of client parameter arrays
            client_weights: List of aggregation weights for each client
            client_architectures: List of architecture info for each client
            
        Returns:
            Harmonized and aggregated parameters
        """
        try:
            # Input validation
            if not client_parameters_list:
                print("[FedMA] Warning: No client parameters provided")
                return []
            
            if len(client_parameters_list) != len(client_weights):
                print(f"[FedMA] Warning: Parameter count ({len(client_parameters_list)}) != weight count ({len(client_weights)})")
                # Pad weights with equal weights if needed
                if len(client_weights) < len(client_parameters_list):
                    default_weight = 1.0 / len(client_parameters_list)
                    client_weights = client_weights + [default_weight] * (len(client_parameters_list) - len(client_weights))
                else:
                    client_weights = client_weights[:len(client_parameters_list)]
            
            if len(client_parameters_list) != len(client_architectures):
                print(f"[FedMA] Warning: Parameter count ({len(client_parameters_list)}) != architecture count ({len(client_architectures)})")
                # Pad architectures with defaults if needed
                while len(client_architectures) < len(client_parameters_list):
                    default_arch = {'input_channels': 1, 'num_classes': 2, 'patch_size': [64, 64, 64]}
                    client_architectures.append(default_arch)
                client_architectures = client_architectures[:len(client_parameters_list)]
            
            print(f"[FedMA] Starting FedMA aggregation for {len(client_parameters_list)} clients")
            
            # Check overall compatibility before proceeding
            if not self._check_overall_compatibility(client_parameters_list):
                print(f"[FedMA] Models are too different for meaningful harmonization")
                print(f"[FedMA] Falling back to selecting first client's model")
                return client_parameters_list[0]
            
            # Use actual parameter names from client architectures if available
            # For now, we'll need to work with the parameter indices since we don't have
            # the actual parameter names in the FedMA strategy
            param_dicts = []
            param_names = [f"param_{i}" for i in range(len(client_parameters_list[0]))]
            
            for client_params in client_parameters_list:
                param_dict = {name: param for name, param in zip(param_names, client_params)}
                param_dicts.append(param_dict)
            
            # Perform layer-wise harmonization and matching
            harmonized_dict = {}
            
            for i, param_name in enumerate(param_names):
                layers = [param_dict[param_name] for param_dict in param_dicts]
                
                # Detect layer type based on parameter index and shape characteristics
                layer_type = self._detect_layer_type(i, layers, len(param_names))
                
                print(f"[FedMA] Processing {param_name} (index {i}): {layer_type}")
                print(f"[FedMA] Shapes: {[layer.shape for layer in layers]}")
                
                # Apply appropriate harmonization based on detected layer type
                try:
                    if layer_type == "input":
                        harmonized_layer = self._harmonize_and_aggregate_input_layer(
                            layers, client_weights, client_architectures
                        )
                    elif layer_type == "output":
                        harmonized_layer = self._harmonize_and_aggregate_output_layer(
                            layers, client_weights, client_architectures
                        )
                    else:
                        # Middle layer - check if harmonization is needed
                        if self._shapes_compatible(layers):
                            harmonized_layer = self._weighted_average(layers, client_weights)
                        else:
                            # Try FedMA matching for incompatible shapes
                            harmonized_layer = self._fedma_aggregate_middle_layer(
                                layers, client_weights
                            )
                    
                    harmonized_dict[param_name] = harmonized_layer
                    
                except Exception as e:
                    print(f"[FedMA] Error harmonizing {param_name} (type: {layer_type}): {e}")
                    print(f"[FedMA] Falling back to simple weighted average for {param_name}")
                    try:
                        # Fallback: attempt simple weighted average if shapes are compatible
                        if self._shapes_compatible(layers):
                            harmonized_layer = self._weighted_average(layers, client_weights)
                            harmonized_dict[param_name] = harmonized_layer
                            print(f"[FedMA] Fallback successful for {param_name}")
                        else:
                            # Last resort: use first client's parameters
                            print(f"[FedMA] Using first client's parameters for {param_name}")
                            harmonized_dict[param_name] = layers[0]
                    except Exception as fallback_e:
                        print(f"[FedMA] Fallback failed for {param_name}: {fallback_e}")
                        # Ultimate fallback: use first client's parameter
                        harmonized_dict[param_name] = layers[0]
        
            # Convert back to list format
            harmonized_params = [harmonized_dict[name] for name in param_names]
            
            # Log comprehensive aggregation summary
            self._log_aggregation_summary(client_parameters_list, client_architectures, harmonized_params)
            
            print(f"[FedMA] Completed FedMA aggregation")
            return harmonized_params
            
        except Exception as e:
            print(f"[FedMA] Critical error during FedMA aggregation: {e}")
            print(f"[FedMA] Falling back to traditional weighted average")
            
            # Emergency fallback: traditional weighted average
            try:
                # Check if all parameter lists have the same length
                param_lengths = [len(params) for params in client_parameters_list]
                if len(set(param_lengths)) == 1:
                    # All clients have same number of parameters
                    traditional_params = []
                    for i in range(param_lengths[0]):
                        # Check if all parameters at position i have same shape
                        layers_at_i = [client_params[i] for client_params in client_parameters_list]
                        if self._shapes_compatible(layers_at_i):
                            # Same shape, can average directly
                            avg_param = self._weighted_average(layers_at_i, client_weights)
                            traditional_params.append(avg_param)
                        else:
                            # Different shapes, use first client's parameter
                            print(f"[FedMA] Shape mismatch at parameter {i}, using first client's parameter")
                            traditional_params.append(layers_at_i[0])
                    
                    print(f"[FedMA] Emergency fallback successful, aggregated {len(traditional_params)} parameters")
                    return traditional_params
                else:
                    # Different parameter counts, use first client entirely
                    print(f"[FedMA] Parameter count mismatch, using first client's full parameter set")
                    return client_parameters_list[0]
                    
            except Exception as fallback_e:
                print(f"[FedMA] Emergency fallback failed: {fallback_e}")
                print(f"[FedMA] Using first client's parameters as final fallback")
                return client_parameters_list[0] if client_parameters_list else []
    
    def _log_aggregation_summary(self, 
                               client_parameters_list: List[NDArrays],
                               client_architectures: List[Dict],
                               harmonized_params: NDArrays):
        """Log comprehensive summary of FedMA aggregation process."""
        print(f"\n[FedMA] ===== AGGREGATION SUMMARY =====")
        print(f"[FedMA] Clients processed: {len(client_parameters_list)}")
        
        # Architecture analysis
        if client_architectures:
            input_channels = [arch.get('input_channels', 'unknown') for arch in client_architectures]
            num_classes = [arch.get('num_classes', 'unknown') for arch in client_architectures]
            
            print(f"[FedMA] Input channels across clients: {input_channels}")
            print(f"[FedMA] Output classes across clients: {num_classes}")
            
            # Check for heterogeneity
            unique_channels = set(ch for ch in input_channels if ch != 'unknown')
            unique_classes = set(cl for cl in num_classes if cl != 'unknown')
            
            if len(unique_channels) > 1:
                print(f"[FedMA] ⚠️  Input channel heterogeneity detected: {unique_channels}")
                if self.enable_channel_harmonization:
                    print(f"[FedMA] ✅ Channel harmonization applied")
                else:
                    print(f"[FedMA] ❌ Channel harmonization disabled")
            
            if len(unique_classes) > 1:
                print(f"[FedMA] ⚠️  Output class heterogeneity detected: {unique_classes}")
                if self.enable_class_harmonization:
                    print(f"[FedMA] ✅ Class harmonization applied")
                else:
                    print(f"[FedMA] ❌ Class harmonization disabled")
            
            if len(unique_channels) <= 1 and len(unique_classes) <= 1:
                print(f"[FedMA] ✅ No metadata-level architecture heterogeneity detected")
                
        # Analyze actual parameter differences
        if client_parameters_list and len(client_parameters_list) > 1:
            param_count_differences = []
            extreme_size_differences = 0
            
            for i in range(len(client_parameters_list[0])):
                if i < len(client_parameters_list[1]):
                    shape1 = client_parameters_list[0][i].shape
                    shape2 = client_parameters_list[1][i].shape
                    size1, size2 = np.prod(shape1), np.prod(shape2)
                    if size1 != size2:
                        ratio = max(size1, size2) / min(size1, size2)
                        if ratio > 10.0:
                            extreme_size_differences += 1
                        param_count_differences.append((i, shape1, shape2, ratio))
                        
            if extreme_size_differences > 0:
                print(f"[FedMA] ⚠️  {extreme_size_differences} parameters with extreme size differences (>10x)")
                print(f"[FedMA] This suggests fundamentally different model architectures")
                
            if len(param_count_differences) > len(client_parameters_list[0]) * 0.8:
                print(f"[FedMA] ⚠️  {len(param_count_differences)} out of {len(client_parameters_list[0])} parameters differ")
                print(f"[FedMA] Models appear to have very different architectures")
        
        # Parameter analysis
        if client_parameters_list and harmonized_params:
            original_shapes = [param.shape for param in client_parameters_list[0]]
            harmonized_shapes = [param.shape for param in harmonized_params]
            
            print(f"[FedMA] Parameter layers: {len(harmonized_shapes)}")
            
            # Check for shape changes
            shape_changes = 0
            for i, (orig_shape, harm_shape) in enumerate(zip(original_shapes, harmonized_shapes)):
                if orig_shape != harm_shape:
                    shape_changes += 1
                    layer_type = "input" if i == 0 else ("output" if i == len(original_shapes)-1 else "middle")
                    print(f"[FedMA] Layer {i} ({layer_type}): {orig_shape} → {harm_shape}")
            
            if shape_changes == 0:
                print(f"[FedMA] ✅ No parameter shape changes needed")
            else:
                print(f"[FedMA] 🔄 {shape_changes} layers harmonized")
        
        # Strategy status
        strategies_used = []
        if self.enable_channel_harmonization:
            strategies_used.append(f"Channel harmonization ({self.channel_harmonizer.harmonization_method})")
        if self.enable_class_harmonization:
            strategies_used.append(f"Class harmonization ({self.class_harmonizer.harmonization_method})")
        if self.enable_layer_matching:
            strategies_used.append(f"Layer matching ({self.layer_matcher.matching_method})")
        
        print(f"[FedMA] Strategies applied: {', '.join(strategies_used) if strategies_used else 'None'}")
        print(f"[FedMA] ================================\n")
    
    def _check_overall_compatibility(self, client_parameters_list: List[NDArrays]) -> bool:
        """
        Check if models are compatible enough for meaningful FedMA harmonization.
        
        Args:
            client_parameters_list: List of client parameter arrays
            
        Returns:
            True if models are compatible enough, False otherwise
        """
        if len(client_parameters_list) < 2:
            return True
            
        # Compare parameter counts
        param_counts = [len(params) for params in client_parameters_list]
        if len(set(param_counts)) > 1:
            print(f"[FedMA] Different parameter counts: {param_counts}")
            max_count, min_count = max(param_counts), min(param_counts)
            if max_count / min_count > 2.0:  # More than 2x difference
                print(f"[FedMA] Parameter count ratio too high: {max_count/min_count:.1f}")
                return False
        
        # Check for extreme size differences across many parameters
        extreme_differences = 0
        moderate_differences = 0
        total_comparisons = 0
        
        min_params = min(param_counts)
        
        for i in range(min_params):
            shapes = [client_params[i].shape for client_params in client_parameters_list]
            sizes = [np.prod(shape) for shape in shapes]
            
            if len(set(sizes)) > 1:  # Sizes differ
                max_size, min_size = max(sizes), min(sizes)
                ratio = max_size / min_size
                
                if ratio > 100.0:  # Extreme difference
                    extreme_differences += 1
                elif ratio > 10.0:  # Moderate difference  
                    moderate_differences += 1
                    
                total_comparisons += 1
        
        # Compatibility thresholds
        extreme_threshold = min_params * 0.3  # 30% of parameters
        moderate_threshold = min_params * 0.7  # 70% of parameters
        
        if extreme_differences > extreme_threshold:
            print(f"[FedMA] Too many extreme differences: {extreme_differences}/{min_params} (>{extreme_threshold:.0f})")
            return False
            
        if moderate_differences > moderate_threshold:
            print(f"[FedMA] Too many moderate differences: {moderate_differences}/{min_params} (>{moderate_threshold:.0f})")
            return False
        
        print(f"[FedMA] Models compatible: {extreme_differences} extreme, {moderate_differences} moderate differences")
        return True
    
    def _detect_layer_type(self, param_index: int, layers: List[np.ndarray], total_params: int) -> str:
        """
        Detect layer type based on parameter index and shape characteristics.
        
        Args:
            param_index: Index of parameter in the parameter list
            layers: List of parameter arrays from different clients
            total_params: Total number of parameters
            
        Returns:
            Layer type: "input", "output", or "middle"
        """
        if not layers:
            return "middle"
        
        # Check if shapes are different across clients (indicates potential input/output layer)
        shapes = [layer.shape for layer in layers]
        shapes_differ = len(set(shapes)) > 1
        
        # Enhanced architecture analysis
        sizes = [np.prod(shape) for shape in shapes]
        max_size, min_size = max(sizes), min(sizes)
        size_ratio = max_size / min_size if min_size > 0 else float('inf')
        
        # If layers are dramatically different in size, be conservative
        if size_ratio > 50.0:
            print(f"[FedMA] Extremely different layer sizes at param {param_index}: ratio {size_ratio:.1f}")
            return "middle"  # Treat as middle layer to use safe fallback
        
        # First parameter is likely input layer, especially if shapes differ
        if param_index == 0:
            return "input" if shapes_differ else "middle"
        
        # Last few parameters might be output layers
        if param_index >= total_params - 5:  # Last 5 parameters (more conservative)
            # Check if this looks like an output layer with small first dimension (class count)
            first_dims = [shape[0] for shape in shapes]
            if all(dim <= 20 for dim in first_dims) and shapes_differ:  # Likely class outputs
                return "output"
            
            # Check for conv layer output patterns (4D/5D with small first dimension)
            if len(layers[0].shape) >= 4 and shapes_differ:
                if all(dim <= 50 for dim in first_dims):  # Likely output channels
                    return "output"
        
        # Check for input layer patterns in early parameters
        if param_index <= 3:  # First few parameters
            if len(layers[0].shape) >= 4 and shapes_differ:
                # Check if second dimension varies (input channels)
                second_dims = [shape[1] for shape in shapes]
                if len(set(second_dims)) > 1:
                    # Also check if the variation is reasonable (not too extreme)
                    max_ch, min_ch = max(second_dims), min(second_dims)
                    if max_ch / min_ch <= 10:  # Reasonable channel variation
                        return "input"
        
        return "middle"
    
    def _shapes_compatible(self, layers: List[np.ndarray]) -> bool:
        """Check if all layers have compatible shapes for direct averaging."""
        if not layers:
            return True
        
        reference_shape = layers[0].shape
        return all(layer.shape == reference_shape for layer in layers)
    
    def _is_input_layer(self, param_name: str) -> bool:
        """Check if parameter is an input layer."""
        # For now, we can't determine actual layer types from generic param names
        # This will be enhanced when we get actual parameter names from clients
        return param_name == "param_0"  # First parameter is likely input layer
    
    def _is_output_layer(self, param_name: str) -> bool:
        """Check if parameter is an output layer."""
        output_patterns = [
            "seg_layers",
            "output_conv",
            "final_conv",
            "classifier"
        ]
        return any(pattern in param_name for pattern in output_patterns)
    
    def _harmonize_and_aggregate_input_layer(self,
                                           layers: List[np.ndarray],
                                           weights: List[float],
                                           architectures: List[Dict]) -> np.ndarray:
        """Harmonize and aggregate input layers with different channel counts."""
        try:
            print(f"[FedMA] Harmonizing input layer with shapes: {[layer.shape for layer in layers]}")
            
            # Always attempt harmonization first (either channel-aware or general)
            if self.enable_channel_harmonization:
                try:
                    # Use specialized channel harmonization
                    harmonized_layers = self.channel_harmonizer.harmonize_input_channels(layers)
                    result = self._weighted_average(harmonized_layers, weights)
                    print(f"[FedMA] Input layer harmonized using channel harmonizer to shape: {result.shape}")
                    return result
                except Exception as e:
                    print(f"[FedMA] Channel harmonization failed: {e}, trying general harmonization")
            
            # Fall back to general shape harmonization
            result = self._weighted_average(layers, weights)  # This now includes harmonization
            print(f"[FedMA] Input layer harmonized using general method to shape: {result.shape}")
            return result
            
        except Exception as e:
            print(f"[FedMA] All input layer harmonization failed: {e}")
            # Last resort fallback with intelligent selection
            return self._fallback_layer_selection(layers, weights)
    
    def _harmonize_and_aggregate_output_layer(self,
                                            layers: List[np.ndarray],
                                            weights: List[float],
                                            architectures: List[Dict]) -> np.ndarray:
        """Harmonize and aggregate output layers with different class counts."""
        try:
            print(f"[FedMA] Harmonizing output layer with shapes: {[layer.shape for layer in layers]}")
            
            # Always attempt harmonization first (either class-aware or general)
            if self.enable_class_harmonization:
                try:
                    # Use specialized class harmonization
                    client_mappings = []
                    for i, (layer, arch) in enumerate(zip(layers, architectures)):
                        # Create basic mapping if not provided
                        num_classes = layer.shape[0]
                        mapping = {i: f"class_{i}" for i in range(num_classes)}
                        client_mappings.append(mapping)
                    
                    harmonized_layers = self.class_harmonizer.harmonize_output_layers(layers, client_mappings)
                    result = self._weighted_average(harmonized_layers, weights)
                    print(f"[FedMA] Output layer harmonized using class harmonizer to shape: {result.shape}")
                    return result
                except Exception as e:
                    print(f"[FedMA] Class harmonization failed: {e}, trying general harmonization")
            
            # Fall back to general shape harmonization
            result = self._weighted_average(layers, weights)  # This now includes harmonization
            print(f"[FedMA] Output layer harmonized using general method to shape: {result.shape}")
            return result
            
        except Exception as e:
            print(f"[FedMA] All output layer harmonization failed: {e}")
            # Last resort fallback with intelligent selection
            return self._fallback_layer_selection(layers, weights)
    
    def _fedma_aggregate_middle_layer(self,
                                    layers: List[np.ndarray],
                                    weights: List[float]) -> np.ndarray:
        """Apply FedMA matching and aggregation to middle layers."""
        if not self.enable_layer_matching or len(layers) < 2:
            return self._weighted_average(layers, weights)
        
        print(f"[FedMA] Applying layer matching to shapes: {[layer.shape for layer in layers]}")
        
        # Check if all layers are 1D (bias parameters) - use simple averaging
        if all(len(layer.shape) == 1 for layer in layers):
            print(f"[FedMA] All layers are 1D (bias), using simple weighted average")
            return self._weighted_average(layers, weights)
        
        # Check if layers have dramatically different sizes - but still try harmonization
        shapes = [layer.shape for layer in layers]
        sizes = [np.prod(shape) for shape in shapes]
        max_size, min_size = max(sizes), min(sizes)
        size_ratio = max_size / min_size if min_size > 0 else float('inf')
        
        if size_ratio > 50.0:  # Only skip matching for extremely different layers
            print(f"[FedMA] Layers extremely different (size ratio: {size_ratio:.1f}), using harmonized weighted average")
            return self._weighted_average(layers, weights)  # This now does harmonization
        elif size_ratio > 10.0:
            print(f"[FedMA] Layers quite different (size ratio: {size_ratio:.1f}), using harmonized average instead of matching")
            return self._weighted_average(layers, weights)  # Skip matching but still harmonize
        
        # Find optimal neuron/filter matching
        try:
            matchings = self.layer_matcher.match_layers(layers)
            
            # Reorder layers according to matching
            reordered_layers = []
            for layer, matching in zip(layers, matchings):
                if all(idx >= 0 for idx in matching):
                    # Complete matching available
                    reordered_layer = layer[matching]
                else:
                    # Partial matching - use original for unmatched
                    reordered_layer = layer  # Could implement more sophisticated handling
                reordered_layers.append(reordered_layer)
            
            # Apply weighted averaging
            result = self._weighted_average(reordered_layers, weights)
            
            print(f"[FedMA] Middle layer matched and aggregated to shape: {result.shape}")
            return result
            
        except Exception as e:
            print(f"[FedMA] Layer matching failed: {e}, falling back to weighted average")
            return self._weighted_average(layers, weights)
    
    def _weighted_average(self, layers: List[np.ndarray], weights: List[float]) -> np.ndarray:
        """Compute weighted average of layers with intelligent harmonization for different shapes."""
        if not layers:
            return np.array([])
        
        if len(layers) == 1:
            return layers[0]
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            weights = [1.0 / len(layers)] * len(layers)
        else:
            weights = [w / total_weight for w in weights]
        
        # Check if all layers have the same shape - if so, use direct averaging
        reference_shape = layers[0].shape
        if all(layer.shape == reference_shape for layer in layers):
            try:
                result = weights[0] * layers[0]
                for layer, weight in zip(layers[1:], weights[1:]):
                    result += weight * layer
                return result
            except Exception as e:
                print(f"[FedMA] Error in direct weighted average: {e}, using first layer")
                return layers[0]
        
        # Layers have different shapes - attempt harmonization
        print(f"[FedMA] Harmonizing layers with different shapes: {[layer.shape for layer in layers]}")
        return self._harmonize_and_average_layers(layers, weights)
    
    def _harmonize_and_average_layers(self, layers: List[np.ndarray], weights: List[float]) -> np.ndarray:
        """Harmonize layers with different shapes and compute weighted average."""
        try:
            # Determine target shape for harmonization
            target_shape = self._determine_target_shape(layers, weights)
            print(f"[FedMA] Target harmonization shape: {target_shape}")
            
            # Harmonize all layers to target shape
            harmonized_layers = []
            for layer in layers:
                harmonized_layer = self._harmonize_layer_to_shape(layer, target_shape)
                harmonized_layers.append(harmonized_layer)
            
            # Compute weighted average of harmonized layers
            result = weights[0] * harmonized_layers[0]
            for layer, weight in zip(harmonized_layers[1:], weights[1:]):
                result += weight * layer
            
            print(f"[FedMA] Successfully harmonized and averaged layers to shape: {result.shape}")
            return result
            
        except Exception as e:
            print(f"[FedMA] Error in layer harmonization: {e}")
            return self._fallback_layer_selection(layers, weights)
    
    def _determine_target_shape(self, layers: List[np.ndarray], weights: List[float]) -> tuple:
        """Determine the target shape for harmonization based on weighted preference."""
        # Strategy: Use weighted voting to determine each dimension
        shapes = [layer.shape for layer in layers]
        max_dims = max(len(shape) for shape in shapes)
        
        # For each dimension, choose size based on weighted preference
        target_shape = []
        for dim_idx in range(max_dims):
            dim_sizes = []
            dim_weights = []
            
            for shape, weight in zip(shapes, weights):
                if dim_idx < len(shape):
                    dim_sizes.append(shape[dim_idx])
                    dim_weights.append(weight)
                else:
                    # Pad missing dimensions with 1
                    dim_sizes.append(1)
                    dim_weights.append(weight)
            
            # Choose dimension size by weighted preference
            # Prefer larger sizes (more expressive capacity)
            weighted_sizes = [(size * weight, size) for size, weight in zip(dim_sizes, dim_weights)]
            weighted_sizes.sort(reverse=True)  # Sort by weighted preference
            target_size = weighted_sizes[0][1]  # Take the highest weighted size
            
            target_shape.append(target_size)
        
        return tuple(target_shape)
    
    def _harmonize_layer_to_shape(self, layer: np.ndarray, target_shape: tuple) -> np.ndarray:
        """Harmonize a single layer to the target shape using padding/truncation."""
        current_shape = layer.shape
        
        if current_shape == target_shape:
            return layer
        
        # Handle different number of dimensions
        if len(current_shape) != len(target_shape):
            # Pad or truncate dimensions
            if len(current_shape) < len(target_shape):
                # Add singleton dimensions
                new_shape = list(current_shape) + [1] * (len(target_shape) - len(current_shape))
                layer = layer.reshape(new_shape)
                current_shape = tuple(new_shape)
            else:
                # Cannot reduce dimensions easily - use first layer as fallback
                print(f"[FedMA] Cannot reduce dimensions from {current_shape} to {target_shape}")
                return layer
        
        # Harmonize each dimension
        result = layer
        for dim_idx, (current_size, target_size) in enumerate(zip(current_shape, target_shape)):
            if current_size != target_size:
                result = self._harmonize_dimension(result, dim_idx, current_size, target_size)
        
        return result
    
    def _harmonize_dimension(self, layer: np.ndarray, dim_idx: int, current_size: int, target_size: int) -> np.ndarray:
        """Harmonize a specific dimension using padding or truncation."""
        if current_size == target_size:
            return layer
        
        if current_size < target_size:
            # Pad dimension - add zeros at the end
            pad_size = target_size - current_size
            pad_config = [(0, 0)] * layer.ndim
            pad_config[dim_idx] = (0, pad_size)
            return np.pad(layer, pad_config, mode='constant', constant_values=0)
        
        else:
            # Truncate dimension - keep first elements
            slices = [slice(None)] * layer.ndim
            slices[dim_idx] = slice(0, target_size)
            return layer[tuple(slices)]
    
    def _fallback_layer_selection(self, layers: List[np.ndarray], weights: List[float]) -> np.ndarray:
        """Intelligent fallback when harmonization fails."""
        # Select layer based on weighted preference and size
        layer_scores = []
        for i, (layer, weight) in enumerate(zip(layers, weights)):
            # Score based on weight and layer size (prefer larger, more expressive layers)
            size_score = np.prod(layer.shape)
            total_score = weight * 0.7 + (size_score / max(np.prod(l.shape) for l in layers)) * 0.3
            layer_scores.append((total_score, i, layer))
        
        # Select best layer
        layer_scores.sort(reverse=True)
        best_layer = layer_scores[0][2]
        
        print(f"[FedMA] Fallback: selected layer with shape {best_layer.shape} based on weighted preference")
        return best_layer


# Export main classes
__all__ = ['FedMAStrategy', 'LayerMatcher', 'ChannelHarmonizer', 'ClassHarmonizer']