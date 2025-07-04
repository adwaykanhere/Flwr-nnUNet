# task.py

import os
# Set CUDA environment for GPU training
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU 0 for training
os.environ['CUDA_HOME'] = '/usr/local/cuda'
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

# Optimize threading for GPU execution and federated learning
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['nnUNet_n_proc_DA'] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Set to false to avoid conflicts

# CUDA memory management for stability
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# Fix PyTorch compilation threading issues for federated learning
os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
os.environ['TORCH_COMPILE_DEBUG'] = '0'

# Disable torch.compile for stability in federated learning
os.environ['TORCH_COMPILE_DISABLE'] = '1'

# Fix Triton library conflicts
os.environ['TRITON_INTERPRET'] = '1'

# Fix PyTorch logging issues
if 'TORCH_LOGS' in os.environ:
    del os.environ['TORCH_LOGS']
import torch
import numpy as np
import json

# nnU-Net v2: Update path if needed
# Prefer the installed nnunetv2 package
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

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
        device: torch.device = torch.device("cuda:0"),
    ):
        super().__init__(
            plans=plans,
            configuration=configuration,
            fold=fold,
            dataset_json=dataset_json,
            device=device,
        )
        
        # Initialize federated learning specific attributes
        self.max_num_epochs = 50  # Default value
        self.current_epoch = 0
        self.all_train_losses = []
        
        # Override nnUNet multiprocessing settings to avoid conflicts with Ray
        self.num_processes = 1  # Disable multiprocessing for data augmentation

    @property
    def loss_function(self):
        """Property to provide loss_function compatibility"""
        return self.loss
    
    def on_train_epoch_start(self):
        """Override nnUNet's epoch start to avoid lr_scheduler conflicts in federated learning."""
        self.network.train()
        print(f"[Trainer] Starting epoch {self.current_epoch}")
        print(f"[Trainer] Current learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")

    def initialize(self):
        if not self.was_initialized:
            # Configure GPU execution and Ray compatibility
            import os
            print(f"[Trainer] Initializing nnUNet trainer on device: {self.device}")
            print(f"[Trainer] CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"[Trainer] Current GPU: {torch.cuda.get_device_name(0)}")
                print(f"[Trainer] GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            # Ensure single-threaded execution for Ray compatibility
            os.environ['nnUNet_def_n_proc'] = '1'
            
            super().initialize() # This will create the network, optimizer, loss, AND dataloaders
            
            # Now replace the default dataloaders with Ray-compatible ones
            print("[Trainer] Replacing dataloaders with Ray-compatible versions...")
            print(f"[Trainer] Preprocessed dataset folder: {self.preprocessed_dataset_folder}")
            print(f"[Trainer] Configuration: {self.configuration_name}")
            print(f"[Trainer] Data identifier: {self.configuration_manager.data_identifier}")
            
            # Validate real data availability
            import os
            if not os.path.exists(self.preprocessed_dataset_folder):
                raise RuntimeError(f"Preprocessed dataset folder does not exist: {self.preprocessed_dataset_folder}")
                
            npz_files = [f for f in os.listdir(self.preprocessed_dataset_folder) if f.endswith('.npz')]
            b2nd_files = [f for f in os.listdir(self.preprocessed_dataset_folder) if f.endswith('.b2nd') and not f.endswith('_seg.b2nd')]
            files = npz_files + b2nd_files
            print(f"[Trainer] Found {len(npz_files)} .npz files and {len(b2nd_files)} .b2nd files in preprocessed folder")
            
            if len(files) == 0:
                raise RuntimeError(f"No .npz or .b2nd data files found in {self.preprocessed_dataset_folder}")
            
            # Validate that we can access the data files for our splits
            tr_keys, val_keys = self.do_split()
            print(f"[Trainer] Validating data files for {len(tr_keys)} training and {len(val_keys)} validation cases...")
            
            missing_files = []
            for key in tr_keys + val_keys:
                npz_file = os.path.join(self.preprocessed_dataset_folder, f"{key}.npz")
                b2nd_file = os.path.join(self.preprocessed_dataset_folder, f"{key}.b2nd")
                if not os.path.exists(npz_file) and not os.path.exists(b2nd_file):
                    missing_files.append(f"{key} (neither .npz nor .b2nd)")
                    
            if missing_files:
                raise RuntimeError(f"Missing required data files: {missing_files}")
                
            print(f"[Trainer] ✓ All required data files are accessible")
                
            # Replace with our custom dataloaders
            self.dataloader_train, self.dataloader_val = self.get_dataloaders()
            print(f"[Trainer] Ray-compatible dataloaders created - train: {self.dataloader_train is not None}, val: {self.dataloader_val is not None}")
            
            # Validate that we can generate real training batches
            print("[Trainer] Validating dataloader can produce real medical data batches...")
            try:
                batch_count = 0
                for batch_data in self.dataloader_train:
                    if batch_data is None:
                        raise RuntimeError("Dataloader returned None batch - this indicates a problem with transforms or data loading")
                    
                    if not isinstance(batch_data, dict):
                        raise RuntimeError(f"Expected dict batch, got {type(batch_data)}")
                    
                    if 'data' not in batch_data or 'target' not in batch_data:
                        raise RuntimeError(f"Batch missing required keys. Got: {list(batch_data.keys())}")
                    
                    data = batch_data['data']
                    target = batch_data['target']
                    
                    # Handle deep supervision (target can be a list)
                    if isinstance(target, list):
                        target_shape = f"[{len(target)} levels: {[t.shape for t in target]}]"
                        target_range = f"[{target[0].min()}, {target[0].max()}]"
                        target_dtype = target[0].dtype
                    else:
                        target_shape = target.shape
                        target_range = f"[{target.min()}, {target.max()}]"
                        target_dtype = target.dtype
                    
                    print(f"[Trainer] ✓ Batch {batch_count + 1}: data shape {data.shape}, target shape {target_shape}")
                    print(f"[Trainer] ✓ Data range: [{data.min():.3f}, {data.max():.3f}], dtype: {data.dtype}")
                    print(f"[Trainer] ✓ Target range: {target_range}, dtype: {target_dtype}")
                    
                    # Validate this looks like real medical data
                    if data.min() == data.max():
                        raise RuntimeError("Data appears to be constant (dummy data) - need real medical images")
                    
                    batch_count += 1
                    if batch_count >= 1:  # Just validate one batch
                        break
                        
                print("[Trainer] ✓ Dataloader validation successful - real medical data confirmed")
                
            except Exception as e:
                print(f"[Trainer] ✗ Dataloader validation failed: {e}")
                raise RuntimeError(f"Dataloader cannot produce valid real data batches: {e}")

            self.was_initialized = True

    def get_case_identifiers(self, folder: str):
        """
        Find case identifiers from .npz or .b2nd files in the nnUNet preprocessed dataset.
        """
        import os
        npz_identifiers = [i[:-4] for i in os.listdir(folder) if i.endswith(".npz")]
        b2nd_identifiers = [i[:-5] for i in os.listdir(folder) if i.endswith(".b2nd") and not i.endswith("_seg.b2nd")]
        # Combine and deduplicate identifiers
        case_identifiers = list(set(npz_identifiers + b2nd_identifiers))
        return case_identifiers

    def do_split(self):
        """
        Override do_split to handle nnUNet preprocessed .npz and .b2nd files.
        """
        from batchgenerators.utilities.file_and_folder_operations import isfile, join, save_json, load_json
        from nnunetv2.utilities.crossval_split import generate_crossval_split
        import numpy as np
        
        if self.fold == "all":
            # Use our custom case identifier function for .npz and .b2nd files
            case_identifiers = self.get_case_identifiers(self.preprocessed_dataset_folder)
            tr_keys = case_identifiers
            val_keys = tr_keys
        else:
            splits_file = join(self.preprocessed_dataset_folder_base, "splits_final.json")
            
            # Use our custom case identifier function
            case_identifiers = self.get_case_identifiers(self.preprocessed_dataset_folder)
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
        Use nnUNet's native dataset loading approach for proper data handling.
        """
        # Get splits using nnUNet's native method
        tr_keys, val_keys = self.do_split()
        
        print(f"[Trainer] Creating nnUNet datasets with real medical data - tr: {len(tr_keys)}, val: {len(val_keys)}")
        
        # Use nnUNet's native dataset class with automatic inference
        from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
        dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
        dataset_tr = dataset_class(self.preprocessed_dataset_folder, tr_keys,
                                   folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
        dataset_val = dataset_class(self.preprocessed_dataset_folder, val_keys,
                                    folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
        
        return dataset_tr, dataset_val
        
    def get_allowed_n_proc_DA(self):
        """Override to force minimal multiprocessing for Ray compatibility"""
        return 1  # Use single process for Ray compatibility
    
    def get_dataloaders(self):
        """Override get_dataloaders to use nnUNet's native dataloaders with Ray compatibility"""
        # Use nnUNet's native method for proper data pipeline setup
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)

        # Get deep supervision scales using nnUNet's native method
        deep_supervision_scales = self._get_deep_supervision_scales()

        # Get data augmentation parameters using nnUNet's native method
        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        print(f"[Trainer] Setting up nnUNet transforms using native nnUNet methods...")
        print(f"[Trainer] ✓ patch_size: {patch_size}")
        print(f"[Trainer] ✓ rotation_for_DA: {rotation_for_DA}")
        print(f"[Trainer] ✓ deep_supervision_scales: {deep_supervision_scales}")
        print(f"[Trainer] ✓ mirror_axes: {mirror_axes}")
        print(f"[Trainer] ✓ do_dummy_2d_data_aug: {do_dummy_2d_data_aug}")
        print(f"[Trainer] ✓ use_mask_for_norm: {self.configuration_manager.use_mask_for_norm}")

        # Create transforms using nnUNet's native method
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        # validation pipeline
        val_transforms = self.get_validation_transforms(deep_supervision_scales,
                                                        is_cascaded=self.is_cascaded,
                                                        foreground_labels=self.label_manager.foreground_labels,
                                                        regions=self.label_manager.foreground_regions if
                                                        self.label_manager.has_regions else None,
                                                        ignore_label=self.label_manager.ignore_label)

        # Use nnUNet's native datasets  
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        
        print(f"[Trainer] ✓ Successfully created native nnUNet training transforms: {type(tr_transforms)}")

        # Import nnUNet's unified dataloader
        from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
        dl_tr = nnUNetDataLoader(dataset_tr, self.batch_size,
                                 initial_patch_size,
                                 self.configuration_manager.patch_size,
                                 self.label_manager,
                                 oversample_foreground_percent=self.oversample_foreground_percent,
                                 sampling_probabilities=None, pad_sides=None, transforms=tr_transforms)
        dl_val = nnUNetDataLoader(dataset_val, self.batch_size,
                                  self.configuration_manager.patch_size,
                                  self.configuration_manager.patch_size,
                                  self.label_manager,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  sampling_probabilities=None, pad_sides=None, transforms=val_transforms)

        # Use SingleThreadedAugmenter for Ray compatibility (no multiprocessing)
        from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
        mt_gen_train = SingleThreadedAugmenter(dl_tr, None)  # transforms are already in the dataloader
        mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        
        print("[Trainer] Created Ray-compatible dataloaders with nnUNet's native data pipeline")
        
        return mt_gen_train, mt_gen_val


    def run_training_round(self, num_local_epochs: int):
        """Run Kaapana-style federated training loop for specified number of epochs."""
        try:
            if not self.was_initialized:
                print("[Trainer] Initializing trainer...")
                self.initialize()

            print(f"[Trainer] Dataloader status: train={self.dataloader_train is not None}, val={self.dataloader_val is not None}")
            
            if self.dataloader_train is None:
                print("[Trainer] ERROR: Training dataloader is None!")
                return

            # Validate training setup
            if self.network is None:
                print("[Trainer] ERROR: Network is None!")
                return
                
            if self.optimizer is None:
                print("[Trainer] ERROR: Optimizer is None!")
                return
                
            if self.loss is None:
                print("[Trainer] ERROR: Loss function is None!")
                return

            start_epoch = self.current_epoch
            target_epoch = min(self.current_epoch + num_local_epochs, self.max_num_epochs)
            
            print(f"[Trainer] Running federated training round: epochs {start_epoch} to {target_epoch-1}")
            print(f"[Trainer] Network has {sum(p.numel() for p in self.network.parameters())} parameters")
            print(f"[Trainer] Network on device: {next(self.network.parameters()).device}")
            
            # Record initial parameters for comparison
            initial_params = {name: param.clone().detach() for name, param in self.network.named_parameters()}
            
            # Train for the specified number of epochs  
            epoch_losses = []
            for epoch in range(start_epoch, target_epoch):
                print(f"[Trainer] Starting federated epoch {epoch + 1}/{target_epoch}...")
                
                # Set learning rate if using scheduler
                if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
                    print(f"[Trainer] Current learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
                
                epoch_loss = self.run_one_epoch()
                epoch_losses.append(epoch_loss)
                self.all_train_losses.append(epoch_loss)
                self.current_epoch = epoch + 1
                
                print(f"[Trainer] Completed epoch {self.current_epoch}: loss={epoch_loss:.4f}")
                
                # Update learning rate if using scheduler
                if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                    
            # Check if parameters actually changed
            param_changes = {}
            total_change = 0.0
            for name, param in self.network.named_parameters():
                if name in initial_params:
                    change = torch.norm(param.data - initial_params[name]).item()
                    param_changes[name] = change
                    total_change += change
                    
            print(f"[Trainer] Training round completed!")
            print(f"[Trainer] - Epochs completed: {len(epoch_losses)}")
            print(f"[Trainer] - Average loss: {np.mean(epoch_losses):.4f}")
            print(f"[Trainer] - Total parameter change: {total_change:.6f}")
            print(f"[Trainer] - Current total epochs: {self.current_epoch}")
            
            if total_change < 1e-8:
                print("[Trainer] WARNING: Very small parameter changes detected - training may not be working properly")
            else:
                print(f"[Trainer] Parameters updated successfully (change magnitude: {total_change:.6f})")
            
        except Exception as e:
            print(f"[Trainer] ERROR in run_training_round: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _save_checkpoint(self):
        """Save model checkpoint (optional - can be implemented for recovery)."""
        pass

    def run_one_epoch(self) -> float:
        """Train exactly one epoch using nnUNet's native training approach."""
        try:
            # Use nnUNet's native epoch start method
            self.on_train_epoch_start()
            
            train_outputs = []
            batch_count = 0
            max_batches = 10  # Reasonable number for federated training
            
            print(f"[Trainer] Starting epoch {self.current_epoch} with up to {max_batches} batches...")
            
            for batch_data in self.dataloader_train:
                    
                print(f"[Trainer] Processing batch {batch_count + 1}/{max_batches}...")
                
                try:
                    # Use nnUNet's native training step
                    batch_result = self.train_step(batch_data)
                    train_outputs.append(batch_result)
                    batch_count += 1
                    
                    # Extract and print loss
                    if isinstance(batch_result, dict) and 'loss' in batch_result:
                        loss_val = batch_result['loss']
                        if hasattr(loss_val, 'item'):
                            loss_val = loss_val.item()
                        print(f"[Trainer] Batch {batch_count} loss: {loss_val:.4f}")
                    
                    if batch_count >= max_batches:
                        print(f"[Trainer] Completed {max_batches} batches, ending epoch")
                        break
                        
                except Exception as batch_error:
                    print(f"[Trainer] Error processing batch {batch_count + 1}: {batch_error}")
                    import traceback
                    traceback.print_exc()
                    # Continue with next batch instead of failing completely
                    continue
            
            # Use nnUNet's native epoch end method
            if train_outputs:
                self.on_train_epoch_end(train_outputs)
                
                # Calculate average loss for return
                losses = []
                for output in train_outputs:
                    if isinstance(output, dict) and 'loss' in output:
                        loss_val = output['loss']
                        if hasattr(loss_val, 'item'):
                            loss_val = loss_val.item()
                        losses.append(loss_val)
                
                avg_loss = float(np.mean(losses)) if losses else 0.0
                print(f"[Trainer] Epoch {self.current_epoch} completed. Processed {len(train_outputs)} batches. Average loss: {avg_loss:.4f}")
                return avg_loss
            else:
                print(f"[Trainer] No successful batches processed in epoch {self.current_epoch}")
                return 0.0
            
        except Exception as e:
            print(f"[Trainer] ERROR in run_one_epoch: {e}")
            import traceback
            traceback.print_exc()
            raise

    def run_iteration(self, data_dict) -> float:
        """Process a single training batch using nnUNet's native train_step method."""
        try:
            # Use nnUNet's native training step which handles all the complexity
            # This ensures we follow nnUNet's proven training pipeline
            result = self.train_step(data_dict)
            
            # Extract loss value from result
            if isinstance(result, dict) and 'loss' in result:
                loss_value = result['loss']
                if hasattr(loss_value, 'item'):
                    return float(loss_value.item())
                elif isinstance(loss_value, (int, float)):
                    return float(loss_value)
                elif hasattr(loss_value, '__len__') and len(loss_value) == 1:
                    return float(loss_value[0])
                else:
                    return float(loss_value)
            else:
                print(f"[Trainer] Warning: Unexpected train_step result format: {type(result)}")
                return 0.0
            
        except Exception as e:
            print(f"[Trainer] Error in run_iteration: {e}")
            print(f"[Trainer] Data dict keys: {list(data_dict.keys()) if isinstance(data_dict, dict) else 'not a dict'}")
            if isinstance(data_dict, dict):
                if 'data' in data_dict:
                    data = data_dict['data']
                    print(f"[Trainer] Data shape: {data.shape if hasattr(data, 'shape') else type(data)}")
                if 'target' in data_dict:
                    target = data_dict['target']
                    print(f"[Trainer] Target type: {type(target)}")
                    if hasattr(target, 'shape'):
                        print(f"[Trainer] Target shape: {target.shape}")
                    elif isinstance(target, list) and len(target) > 0:
                        print(f"[Trainer] Target list length: {len(target)}, first shape: {target[0].shape if hasattr(target[0], 'shape') else 'unknown'}")
            raise

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

    def compute_dice_score(self, pred_seg: torch.Tensor, gt_seg: torch.Tensor, labels: list) -> dict:
        """
        Compute per-label Dice scores following nnUNet approach
        
        Args:
            pred_seg: Predicted segmentation tensor
            gt_seg: Ground truth segmentation tensor  
            labels: List of foreground labels to compute Dice for
            
        Returns:
            Dictionary with per-label dice scores and mean
        """
        dice_scores = {}
        
        for label in labels:
            # Create binary masks
            pred_mask = (pred_seg == label).float()
            gt_mask = (gt_seg == label).float()
            
            # Compute TP, FP, FN
            tp = torch.sum(pred_mask * gt_mask).item()
            fp = torch.sum(pred_mask * (1 - gt_mask)).item()
            fn = torch.sum((1 - pred_mask) * gt_mask).item()
            
            # Compute Dice coefficient: 2*TP / (2*TP + FP + FN)
            if tp + fp + fn == 0:
                dice = float('nan')  # No ground truth and no prediction
            else:
                dice = (2 * tp) / (2 * tp + fp + fn)
            
            dice_scores[str(label)] = dice
        
        # Calculate mean dice (excluding NaN values)
        valid_scores = [score for score in dice_scores.values() if not np.isnan(score)]
        dice_scores['mean'] = np.mean(valid_scores) if valid_scores else float('nan')
        
        return dice_scores

    def run_validation_round(self) -> dict:
        """
        Run online validation using nnUNet's native validation pipeline with batched data
        
        Returns:
            Dictionary with validation metrics including Dice scores
        """
        if not self.was_initialized:
            self.initialize()
        
        print("[Trainer] Running online validation...")
        
        # Check if validation dataloader exists
        if not hasattr(self, 'dataloader_val') or self.dataloader_val is None:
            print("[Trainer] Warning: No validation dataloader available")
            return {"per_label": {}, "mean": 0.0, "num_cases": 0}
        
        # Set network to evaluation mode and disable deep supervision like nnUNet does
        self.network.eval()
        self.set_deep_supervision_enabled(False)
        
        print(f"[Trainer] Processing validation batches...")
        
        val_outputs = []
        batch_count = 0
        max_val_batches = 10  # Limit validation batches for faster feedback
        
        try:
            with torch.no_grad():
                for batch_data in self.dataloader_val:
                    # Use nnUNet's native validation_step method
                    val_result = self.validation_step(batch_data)
                    val_outputs.append(val_result)
                    batch_count += 1
                    
                    if batch_count >= max_val_batches:
                        print(f"[Trainer] Processed {max_val_batches} validation batches")
                        break
                        
                # Use nnUNet's native aggregation method
                if val_outputs:
                    # Fix for federated learning: ensure logger has consistent state
                    # Store the current epoch and temporarily set it to 0 for validation
                    original_epoch = self.current_epoch
                    
                    # Reset logger state for federated validation to prevent index errors
                    if hasattr(self, 'logger') and hasattr(self.logger, 'my_fantastic_logging'):
                        # Ensure ema_fg_dice has at least one element if current_epoch > 0
                        if self.current_epoch > 0 and len(self.logger.my_fantastic_logging['ema_fg_dice']) == 0:
                            # Initialize with a default value for the first validation in federated learning
                            self.logger.my_fantastic_logging['ema_fg_dice'] = [0.0] * self.current_epoch
                    
                    self.on_validation_epoch_end(val_outputs)
                    
                    # Restore original epoch
                    self.current_epoch = original_epoch
                    
                    # Extract the computed metrics from the logger
                    if hasattr(self, 'logger') and hasattr(self.logger, 'my_fantastic_logging'):
                        logging_dict = self.logger.my_fantastic_logging
                        if 'dice_per_class_or_region' in logging_dict and logging_dict['dice_per_class_or_region']:
                            dice_per_class = logging_dict['dice_per_class_or_region'][-1]  # Latest values
                            mean_fg_dice = np.nanmean(dice_per_class)
                            
                            # Create per-label results
                            per_label_dict = {}
                            for i, dice_val in enumerate(dice_per_class):
                                if not np.isnan(dice_val):
                                    per_label_dict[str(i + 1)] = float(dice_val)  # Labels start from 1
                            
                            validation_results = {
                                "per_label": per_label_dict,
                                "mean": float(mean_fg_dice),
                                "num_batches": batch_count
                            }
                            
                            print(f"[Trainer] Online validation complete: mean Dice = {mean_fg_dice:.4f} ({batch_count} batches)")
                            return validation_results
                    
                    # Fallback: manually aggregate if logger method fails
                    from nnunetv2.utilities.collate_outputs import collate_outputs
                    outputs_collated = collate_outputs(val_outputs)
                    tp = np.sum(outputs_collated['tp_hard'], 0)
                    fp = np.sum(outputs_collated['fp_hard'], 0)  
                    fn = np.sum(outputs_collated['fn_hard'], 0)
                    
                    # Compute Dice scores manually
                    dice_per_class = [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]
                    mean_fg_dice = np.nanmean(dice_per_class)
                    
                    per_label_dict = {}
                    for i, dice_val in enumerate(dice_per_class):
                        if not np.isnan(dice_val):
                            per_label_dict[str(i + 1)] = float(dice_val)
                    
                    validation_results = {
                        "per_label": per_label_dict,
                        "mean": float(mean_fg_dice),
                        "num_batches": batch_count
                    }
                    
                    print(f"[Trainer] Online validation complete: mean Dice = {mean_fg_dice:.4f} ({batch_count} batches)")
                    return validation_results
                else:
                    print("[Trainer] Warning: No validation batches processed")
                    return {"per_label": {}, "mean": 0.0, "num_batches": 0}
                    
        except Exception as e:
            print(f"[Trainer] Error during validation: {e}")
            import traceback
            traceback.print_exc()
            return {"per_label": {}, "mean": 0.0, "num_batches": 0}
            
        finally:
            # Always restore training mode and deep supervision
            self.network.train()
            self.set_deep_supervision_enabled(True)


    def validation_step(self, batch: dict) -> dict:
        """
        nnUNet-compatible validation step that returns tp_hard, fp_hard, fn_hard for Dice calculation
        
        Args:
            batch: Dictionary with 'data' and 'target' keys
            
        Returns:
            Dictionary with loss, tp_hard, fp_hard, fn_hard for nnUNet validation
        """
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Run forward pass
        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else torch.no_grad():
            output = self.network(data)
            del data
            
            # Ensure both output and target are in list format for deep supervision loss
            if not isinstance(output, (list, tuple)):
                output = [output]
            if not isinstance(target, (list, tuple)):
                target = [target]
                
            loss = self.loss(output, target)

        # Handle deep supervision - use highest resolution output
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        # Calculate TP, FP, FN for Dice computation
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # Standard softmax case
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        # Handle ignore label if present
        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

        # Import nnUNet's tp/fp/fn calculation function
        from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        
        # Remove background class for standard training (not region-based)
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]  # Remove background
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {
            'loss': loss.detach().cpu().numpy(),
            'tp_hard': tp_hard,
            'fp_hard': fp_hard,
            'fn_hard': fn_hard
        }

    def save_best_checkpoint_pytorch(self, output_dir: str, round_num: int, validation_dice: float, 
                                   is_best: bool = False) -> str:
        """
        Save model checkpoint in nnUNet's PyTorch format (.pth)
        
        Args:
            output_dir: Directory to save the checkpoint
            round_num: Current federated learning round
            validation_dice: Current validation Dice score
            is_best: Whether this is the best model so far
            
        Returns:
            Path to saved checkpoint file
        """
        import os
        from pathlib import Path
        
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Only save best models - simplified filename
        if is_best:
            filename = output_path / "model_best.pt"
        else:
            # Don't save regular checkpoints, only best models
            print(f"[Trainer] Skipping regular checkpoint save, only saving best models")
            return None
        
        # Prepare checkpoint data following nnUNet format
        if hasattr(self, 'network') and self.network is not None:
            checkpoint = {
                'network_weights': self.network.state_dict(),
                'optimizer_state': self.optimizer.state_dict() if hasattr(self, 'optimizer') and self.optimizer else None,
                'current_epoch': getattr(self, 'current_epoch', round_num),
                'validation_dice': validation_dice,
                'federated_round': round_num,
                'trainer_name': self.__class__.__name__,
                'plans': getattr(self, 'plans', None),
                'configuration': getattr(self, 'configuration_name', '3d_fullres'),
                'fold': getattr(self, 'fold', 0),
                'dataset_json': getattr(self, 'dataset_json', None),
                'inference_allowed_mirroring_axes': getattr(self, 'inference_allowed_mirroring_axes', None),
                'is_federated_model': True
            }
            
            # Save checkpoint
            torch.save(checkpoint, filename)
            print(f"[Trainer] Saved best PyTorch model: {filename}")
            print(f"[Trainer] Best validation Dice: {validation_dice:.4f}")
            print(f"[Trainer] Round: {round_num}")
            
            return str(filename)
        else:
            print(f"[Trainer] Warning: Cannot save checkpoint - network not initialized")
            return None

    def load_pytorch_checkpoint(self, checkpoint_path: str) -> dict:
        """
        Load a PyTorch checkpoint saved by save_best_checkpoint_pytorch
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            Dictionary containing checkpoint metadata
        """
        import os
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"[Trainer] Loading PyTorch checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if hasattr(self, 'network') and self.network is not None:
            # Load network weights
            self.network.load_state_dict(checkpoint['network_weights'])
            
            # Load optimizer state if available
            if checkpoint.get('optimizer_state') and hasattr(self, 'optimizer') and self.optimizer:
                self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            
            print(f"[Trainer] Loaded checkpoint from round {checkpoint.get('federated_round', 'unknown')}")
            print(f"[Trainer] Validation Dice: {checkpoint.get('validation_dice', 'unknown')}")
            
            return {
                'validation_dice': checkpoint.get('validation_dice', 0.0),
                'federated_round': checkpoint.get('federated_round', 0),
                'current_epoch': checkpoint.get('current_epoch', 0)
            }
        else:
            print(f"[Trainer] Warning: Cannot load checkpoint - network not initialized")
            return {}


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
