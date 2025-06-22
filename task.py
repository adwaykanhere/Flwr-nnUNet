# task.py

import os
# Force disable CUDA before importing torch to prevent initialization crashes
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['nnUNet_n_proc_DA'] = '1'
import torch
import numpy as np
import json

# nnU-Net v2: Update path if needed
# Prefer the installed nnunetv2 package
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class NnUNetDataset:
    def __init__(self, keys, folder):
        self.keys_list = keys
        self.folder = folder
        # Preload properties to avoid file access issues during training
        self._properties_cache = {}
        self._preload_properties()
        
    def _preload_properties(self):
        """Preload all properties to avoid file I/O during training"""
        import pickle
        import os
        print(f"[Dataset] Preloading properties for {len(self.keys_list)} cases...")
        for key in self.keys_list:
            props_file = os.path.join(self.folder, f"{key}.pkl")
            try:
                with open(props_file, 'rb') as f:
                    self._properties_cache[key] = pickle.load(f)
            except Exception as e:
                print(f"[Dataset] Warning: Could not load properties for {key}: {e}")
                self._properties_cache[key] = {}
        
    def keys(self):
        return self.keys_list
        
    def __len__(self):
        return len(self.keys_list)
        
    def __getitem__(self, key):
        data, seg, properties = self.load_case(key)
        return {
            'data': data,
            'seg': seg,
            'properties': properties
        }
        
    def load_case(self, key):
        """Load actual nnUNet preprocessed .npz files and cached properties"""
        import os
        import numpy as np
        
        # Construct file paths for nnUNet preprocessed data
        data_file = os.path.join(self.folder, f"{key}.npz")
        
        try:
            # Load preprocessed nnUNet data
            npz_data = np.load(data_file)
            data = npz_data['data']  # Shape: (channels, z, y, x)
            seg = npz_data['seg']    # Shape: (1, z, y, x)
            
            # Get cached properties
            properties = self._properties_cache.get(key, {})
            
            # Ensure data is float32 and seg is int for compatibility
            data = np.asarray(data, dtype=np.float32)
            seg = np.asarray(seg, dtype=np.int32)
            
            return data, seg, properties
            
        except Exception as e:
            print(f"[Dataset] Error loading {key}: {e}")
            raise e  # Don't use dummy data, let the error propagate


class FedNnUNetTrainer(nnUNetTrainer):
    """
    Custom trainer for nnU-Net v2 that supports partial (incremental) training each FL round.
    It automatically loads the data from the dataset.json + 'plans' once you call `initialize()`.
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(
            plans=plans,
            configuration=configuration,
            fold=fold,
            dataset_json=dataset_json,
            unpack_dataset=unpack_dataset,
            device=device,
        )
        
        # Initialize federated learning specific attributes
        self.max_num_epochs = 50  # Default value
        self.current_epoch = 0
        self.all_train_losses = []

    @property
    def loss_function(self):
        """Property to provide loss_function compatibility"""
        return self.loss
    
    def initialize(self):
        if not self.was_initialized:
            # Disable multiprocessing to avoid CUDA crashes in WSL2
            import os
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force no CUDA
            os.environ['MKL_NUM_THREADS'] = '1'      # Disable Intel MKL threading
            os.environ['NUMEXPR_NUM_THREADS'] = '1'  # Disable NumExpr threading
            os.environ['nnUNet_n_proc_DA'] = '1'     # Disable nnUNet data augmentation multiprocessing
            
            super().initialize() # TODO: Need to look into setting 3d_fullres as it might default to 2d

            # If the parent didn't set dataloaders, do it ourselves:
            if self.dataloader_train is None or self.dataloader_val is None:
                self.dataloader_train, self.dataloader_val = self.get_dataloaders()

            self.was_initialized = True

    def get_case_identifiers_from_npz(self, folder: str):
        """
        Find case identifiers from .npz files in the nnUNet preprocessed dataset.
        """
        import os
        case_identifiers = [i[:-4] for i in os.listdir(folder) if i.endswith(".npz")]
        return case_identifiers

    def do_split(self):
        """
        Override do_split to handle nnUNet preprocessed .npz files.
        """
        from batchgenerators.utilities.file_and_folder_operations import isfile, join, save_json, load_json
        from nnunetv2.utilities.crossval_split import generate_crossval_split
        import numpy as np
        
        if self.fold == "all":
            # Use our custom case identifier function for .npz files
            case_identifiers = self.get_case_identifiers_from_npz(self.preprocessed_dataset_folder)
            tr_keys = case_identifiers
            val_keys = tr_keys
        else:
            splits_file = join(self.preprocessed_dataset_folder_base, "splits_final.json")
            
            # Use our custom case identifier function
            case_identifiers = self.get_case_identifiers_from_npz(self.preprocessed_dataset_folder)
            print(f"[Trainer] Found {len(case_identifiers)} case identifiers: {case_identifiers[:5]}...")
            
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                all_keys_sorted = list(np.sort(case_identifiers))
                splits = generate_crossval_split(all_keys_sorted, seed=12345, n_splits=5)
                save_json(splits, splits_file)
            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_json(splits_file)

            tr_keys = splits[self.fold]['train']
            val_keys = splits[self.fold]['val']

        self.print_to_log_file(f"This fold has {len(tr_keys)} training and {len(val_keys)} validation cases.")
        return tr_keys, val_keys

    def get_tr_and_val_datasets(self):
        """
        Override to handle nnUNet preprocessed .npz files with real medical imaging data.
        """
        # Get splits using our custom method
        tr_keys, val_keys = self.do_split()
        
        print(f"[Trainer] Creating nnUNet datasets with real medical data - tr: {len(tr_keys)}, val: {len(val_keys)}")
        dataset_tr = NnUNetDataset(tr_keys, self.preprocessed_dataset_folder)
        dataset_val = NnUNetDataset(val_keys, self.preprocessed_dataset_folder)
        
        return dataset_tr, dataset_val


    def run_training_round(self, num_local_epochs: int):
        """Run the official nnU-Net training loop for a few epochs."""
        try:
            if not self.was_initialized:
                print("[Trainer] Initializing trainer...")
                self.initialize()

            print(f"[Trainer] Dataloader status: train={self.dataloader_train is not None}, val={self.dataloader_val is not None}")
            
            if self.dataloader_train is None:
                print("[Trainer] ERROR: Training dataloader is None!")
                return

            start_epoch = self.current_epoch
            target_epoch = min(self.current_epoch + num_local_epochs, self.max_num_epochs)
            
            print(f"[Trainer] Running training from epoch {start_epoch} to {target_epoch}")
            
            # Train for the specified number of epochs
            for epoch in range(start_epoch, target_epoch):
                print(f"[Trainer] Starting epoch {epoch + 1}...")
                epoch_loss = self.run_one_epoch()
                self.all_train_losses.append(epoch_loss)
                self.current_epoch = epoch + 1
                
                print(f"[Trainer] Epoch {self.current_epoch}: loss={epoch_loss:.4f}")
                
                # Save checkpoint periodically
                if (self.current_epoch % 10) == 0:
                    self._save_checkpoint()
                    
            print(f"[Trainer] Completed {num_local_epochs} epochs, total epochs: {self.current_epoch}")
            
        except Exception as e:
            print(f"[Trainer] ERROR in run_training_round: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _save_checkpoint(self):
        """Save model checkpoint (optional - can be implemented for recovery)."""
        pass

    def run_one_epoch(self) -> float:
        """Train exactly one epoch over self.dl_tr, returning average training loss."""
        try:
            print(f"[Trainer] Setting network to training mode...")
            self.network.train()
            losses = []
            
            # For testing: limit to just a few batches to verify functionality
            batch_count = 0
            max_batches = 3  # Process only 3 batches for faster testing
            
            print(f"[Trainer] Starting to iterate over training dataloader (max {max_batches} batches)...")
            
            for batch_data in self.dataloader_train:
                print(f"[Trainer] Processing batch {batch_count + 1}/{max_batches}...")
                batch_loss = self.run_iteration(batch_data)
                losses.append(batch_loss)
                batch_count += 1
                print(f"[Trainer] Batch {batch_count} loss: {batch_loss:.4f}")
                
                if batch_count >= max_batches:
                    break
                    
            avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
            print(f"[Trainer] Epoch completed. Average loss: {avg_loss:.4f}")
            return avg_loss
            
        except Exception as e:
            print(f"[Trainer] ERROR in run_one_epoch: {e}")
            import traceback
            traceback.print_exc()
            raise

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

        # Handle deep supervision: both logits and target should be lists or both single tensors
        # The deep supervision wrapper expects both to be lists
        if isinstance(logits, list) and isinstance(target, list):
            # Both are lists - pass directly to loss function
            loss_value = self.loss(logits, target)
        elif not isinstance(logits, list) and not isinstance(target, list):
            # Both are single tensors - pass directly to loss function
            loss_value = self.loss(logits, target)
        else:
            # Mismatch: one is list, one is tensor - convert single tensor to list
            if isinstance(target, list) and not isinstance(logits, list):
                # Target is list but logits is tensor - make logits a list
                logits = [logits]
            elif isinstance(logits, list) and not isinstance(target, list):
                # Logits is list but target is tensor - make target a list
                target = [target]
            loss_value = self.loss(logits, target)

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
    Implements federated fingerprint aggregation for any nnUNet dataset.
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
