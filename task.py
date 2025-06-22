# task.py

import os
import torch
import numpy as np
import json

# nnU-Net v2: Update path if needed
# Prefer the installed nnunetv2 package
from nnUNet.nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class FedNnUNetTrainer(nnUNetTrainer):
    """
    Custom trainer for nnU-Net v2 that supports partial (incremental) training each FL round.
    It automatically loads the data from the dataset.json + 'plans' once you call `initialize()`.
    """

    def __init__(
        self,
        plans: str,      # Path to your "plans.json" containing "3d_fullres" or other configs
        configuration: str,   # "3d_fullres" in your case
        fold: int,
        dataset_json: str,    # Path to dataset.json describing local training data
        output_folder: str,
        device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        max_num_epochs: int = 50,
    ):
        with open(plans, "r") as f:
            plans_dict = json.load(f)
        with open(dataset_json, "r") as f:
            dataset_dict = json.load(f)

        super().__init__(
            plans=plans_dict,           # <--- pass dict, not "plans_path"
            configuration=configuration,
            fold=fold,
            dataset_json=dataset_dict,       # <--- pass dict, not "dataset_json" path
            device=device, # if your trainer expects "device" by name
        )
        
        self.device = device
        self.max_num_epochs = max_num_epochs
        self.current_epoch = 0
        self.all_train_losses = []
    
    def initialize(self):
        if not self.was_initialized:
            super().initialize() # TODO: Need to look into setting 3d_fullres as it might default to 2d

            # If the parent didn't set dataloaders, do it ourselves:
            if self.dataloader_train is None or self.dataloader_val is None:
                self.dataloader_train, self.dataloader_val = self.get_dataloaders()

            self.was_initialized = True

    def run_training_round(self, num_local_epochs: int):
        """Run the official nnU-Net training loop for a few epochs."""
        if not self.was_initialized:
            self.initialize()

        start_epoch = self.current_epoch
        target_epoch = min(self.current_epoch + num_local_epochs, self.max_num_epochs)
        
        print(f"[Trainer] Running training from epoch {start_epoch} to {target_epoch}")
        
        # Train for the specified number of epochs
        for epoch in range(start_epoch, target_epoch):
            epoch_loss = self.run_one_epoch()
            self.all_train_losses.append(epoch_loss)
            self.current_epoch = epoch + 1
            
            print(f"[Trainer] Epoch {self.current_epoch}: loss={epoch_loss:.4f}")
            
            # Save checkpoint periodically
            if (self.current_epoch % 10) == 0:
                self._save_checkpoint()
                
        print(f"[Trainer] Completed {num_local_epochs} epochs, total epochs: {self.current_epoch}")

    def _save_checkpoint(self):
        """Save model checkpoint (optional - can be implemented for recovery)."""
        pass

    def run_one_epoch(self) -> float:
        """Train exactly one epoch over self.dl_tr, returning average training loss."""
        self.network.train()
        losses = []
        for batch_data in self.dataloader_train:
            batch_loss = self.run_iteration(batch_data)
            losses.append(batch_loss)
        return float(np.mean(losses)) if len(losses) > 0 else 0.0

    def run_iteration(self, data_dict) -> float:
        data = data_dict["data"]
        target = data_dict["target"]

        # Move data (always a tensor) to GPU/CPU device
        data = data.to(self.device)

        # If target is a list (deep supervision), convert each item
        if isinstance(target, list):
            target = [t.to(self.device) for t in target]
        else:
            target = target.to(self.device)

        self.optimizer.zero_grad(set_to_none=True)
        logits = self.network(data)

        # If deep supervision is enabled, `logits` might also be a list.
        # Usually you'd sum the losses or handle each level:
        if isinstance(logits, list):
            # Example: sum the losses from each output level
            loss_value = 0.0
            for logit, tar in zip(logits, target):
                loss_value += self.loss_function(logit, tar)
        else:
            # Single output
            loss_value = self.loss_function(logits, target)

        loss_value.backward()
        self.optimizer.step()

        return loss_value.item()

    def get_weights(self):
        """Convert the model's state_dict into {layer_name: numpy_array}."""
        if not self.was_initialized:
            self.initialize()  # create self.network, etc.

        sd = self.network.state_dict()
        return {k: v.cpu().numpy() for k, v in sd.items()}

    def set_weights(self, weights: dict):
        """Load from {layer_name: numpy_array} back into model parameters."""
        new_sd = {}
        for k, arr in weights.items():
            new_sd[k] = torch.tensor(arr, device=self.device)
        self.network.load_state_dict(new_sd, strict=True)


def merge_local_fingerprints(local_fps: list[dict]) -> dict:
    """
    Example aggregator for local “fingerprint” data. 
    local_fps might look like: [{"mean": X, "std": Y, "n": N}, ...].
    """
    if not local_fps:
        return {}

    total_n = 0
    sum_mean = 0.0
    for fp in local_fps:
        n = fp.get("n", 1)
        m = fp.get("mean", 0.0)
        sum_mean += (m * n)
        total_n += n

    if total_n == 0:
        return {"mean": 0.0}
    return {"mean": sum_mean / total_n}


def merge_dataset_fingerprints(local_fps: list[dict]) -> dict:
    """
    Merge nnU-Net dataset_fingerprint.json files from several clients.
    Implements kaapana-style fingerprint aggregation using estimate mode.
    """
    if not local_fps:
        return {}

    print(f"[Fingerprint] Merging {len(local_fps)} local fingerprints")
    
    # Initialize merged fingerprint structure
    merged: dict[str, any] = {
        "shapes_after_crop": [],
        "spacings": [],
        "foreground_intensity_properties_per_channel": {}
    }
    
    # Collect all shapes and spacings
    all_shapes = []
    all_spacings = []
    
    for fp in local_fps:
        shapes = fp.get("shapes_after_crop", [])
        spacings = fp.get("spacings", [])
        all_shapes.extend(shapes)
        all_spacings.extend(spacings)
    
    merged["shapes_after_crop"] = all_shapes
    merged["spacings"] = all_spacings
    
    # Aggregate intensity properties per modality using weighted averaging
    intensity_props: dict[str, dict[str, list[tuple[float, int]]]] = {}
    
    for fp in local_fps:
        n_samples = len(fp.get("spacings", []))  # Number of samples for this client
        
        for mod_key, props in fp.get("foreground_intensity_properties_per_channel", {}).items():
            if mod_key not in intensity_props:
                intensity_props[mod_key] = {
                    "mean": [],
                    "std": [],
                    "min": [], 
                    "max": [],
                    "median": [],
                    "percentile_00_5": [],
                    "percentile_99_5": [],
                }
            
            # Store value and weight (number of samples) for weighted averaging
            for stat_key in intensity_props[mod_key].keys():
                if stat_key in props:
                    intensity_props[mod_key][stat_key].append((props[stat_key], n_samples))

    # Compute weighted averages for intensity properties
    if intensity_props:
        for mod_key, stats in intensity_props.items():
            merged["foreground_intensity_properties_per_channel"][mod_key] = {}
            
            for stat_key, values_weights in stats.items():
                if values_weights:
                    if stat_key in ["min"]:
                        # For min, take the global minimum
                        merged_value = min(v for v, w in values_weights)
                    elif stat_key in ["max"]:
                        # For max, take the global maximum
                        merged_value = max(v for v, w in values_weights)
                    else:
                        # For mean, std, median, percentiles: use weighted average
                        total_weight = sum(w for v, w in values_weights)
                        if total_weight > 0:
                            merged_value = sum(v * w for v, w in values_weights) / total_weight
                        else:
                            merged_value = 0.0
                    
                    merged["foreground_intensity_properties_per_channel"][mod_key][stat_key] = float(merged_value)
                else:
                    merged["foreground_intensity_properties_per_channel"][mod_key][stat_key] = 0.0

    print(f"[Fingerprint] Merged fingerprint: {len(all_shapes)} shapes, {len(all_spacings)} spacings")
    print(f"[Fingerprint] Modalities: {list(merged['foreground_intensity_properties_per_channel'].keys())}")
    
    return merged
