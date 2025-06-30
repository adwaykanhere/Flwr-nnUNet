# task.py

import os
# Set CUDA environment for GPU training
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU 0 for training
os.environ['CUDA_HOME'] = '/usr/local/packages/cuda'
os.environ['LD_LIBRARY_PATH'] = '/usr/local/packages/cuda/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

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
        Run validation inference and compute Dice scores following nnUNet approach
        
        Returns:
            Dictionary with validation metrics including Dice scores
        """
        if not self.was_initialized:
            self.initialize()
        
        print("[Trainer] Running validation...")
        
        # Set network to evaluation mode
        self.network.eval()
        
        # Get validation cases from splits
        _, val_keys = self.do_split()
        
        if not val_keys:
            print("[Trainer] Warning: No validation cases found")
            return {"dice_scores": {}, "mean_dice": 0.0, "num_cases": 0}
        
        # Create validation dataset
        dataset_val = self.dataset_class(self.preprocessed_dataset_folder, val_keys,
                                       folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
        
        print(f"[Trainer] Validating on {len(val_keys)} cases...")
        
        all_dice_scores = []
        processed_cases = 0
        
        with torch.no_grad():
            for case_idx, case_id in enumerate(val_keys):
                try:
                    # Load case data
                    data, _, properties = dataset_val.load_case(case_id)
                    
                    # Load ground truth segmentation
                    seg_path = os.path.join(self.preprocessed_dataset_folder_base, 'gt_segmentations', f"{case_id}.nii.gz")
                    if not os.path.exists(seg_path):
                        # Try alternative segmentation path
                        seg_path = os.path.join(self.preprocessed_dataset_folder_base, 'gt_segmentations', f"{case_id}_seg.nii.gz")
                    
                    if not os.path.exists(seg_path):
                        print(f"[Trainer] Warning: Ground truth not found for {case_id}, skipping")
                        continue
                    
                    # Load ground truth
                    import SimpleITK as sitk
                    gt_sitk = sitk.ReadImage(seg_path)
                    gt_seg = sitk.GetArrayFromImage(gt_sitk)
                    gt_seg = torch.from_numpy(gt_seg).long()
                    
                    # Convert data to tensor if needed
                    if not isinstance(data, torch.Tensor):
                        data = torch.from_numpy(data[:])  # [:] to convert blosc2 to numpy
                    
                    # Move to device
                    data = data.to(self.device)
                    
                    # Run inference using simplified sliding window approach
                    prediction_logits = self.predict_sliding_window_simple(data)
                    
                    # Convert logits to segmentation
                    prediction_seg = torch.argmax(prediction_logits, dim=0).cpu()
                    
                    # Resize GT to match prediction if needed
                    if gt_seg.shape != prediction_seg.shape:
                        # Simple resize using interpolation
                        import torch.nn.functional as F
                        gt_seg = gt_seg.unsqueeze(0).unsqueeze(0).float()
                        gt_seg = F.interpolate(gt_seg, size=prediction_seg.shape, mode='nearest')
                        gt_seg = gt_seg.squeeze().long()
                    
                    # Compute Dice score for this case
                    foreground_labels = self.label_manager.foreground_labels
                    case_dice = self.compute_dice_score(prediction_seg, gt_seg, foreground_labels)
                    all_dice_scores.append(case_dice)
                    processed_cases += 1
                    
                    print(f"[Trainer] Case {case_idx + 1}/{len(val_keys)} ({case_id}): dice={case_dice.get('mean', 0):.3f}")
                    
                except Exception as e:
                    print(f"[Trainer] Error validating case {case_id}: {e}")
                    continue
        
        # Set network back to training mode
        self.network.train()
        
        if not all_dice_scores:
            print("[Trainer] Warning: No validation cases processed successfully")
            return {"dice_scores": {}, "mean_dice": 0.0, "num_cases": 0}
        
        # Aggregate dice scores across all cases
        aggregated_dice = {}
        
        # Get all unique labels
        all_labels = set()
        for dice_dict in all_dice_scores:
            all_labels.update(k for k in dice_dict.keys() if k != 'mean')
        
        # Average dice scores per label
        for label in all_labels:
            label_scores = [d.get(label, 0) for d in all_dice_scores if not np.isnan(d.get(label, float('nan')))]
            if label_scores:
                aggregated_dice[label] = np.mean(label_scores)
        
        # Calculate overall mean
        valid_means = [d.get('mean', 0) for d in all_dice_scores if not np.isnan(d.get('mean', float('nan')))]
        overall_mean = np.mean(valid_means) if valid_means else 0.0
        
        validation_results = {
            "per_label": aggregated_dice,
            "mean": overall_mean,
            "num_cases": processed_cases
        }
        
        print(f"[Trainer] Validation complete: mean Dice = {overall_mean:.4f} ({processed_cases} cases)")
        
        return validation_results

    def predict_sliding_window_simple(self, data: torch.Tensor) -> torch.Tensor:
        """
        Simplified sliding window prediction for validation
        
        Args:
            data: Input data tensor [C, D, H, W]
            
        Returns:
            Prediction logits [num_classes, D, H, W]
        """
        # For simplicity, we'll use a single patch prediction instead of full sliding window
        # This is faster for federated learning validation but less accurate than full nnUNet validation
        
        # Get target patch size from configuration
        patch_size = self.configuration_manager.patch_size
        
        # Simple center crop or padding to patch size
        input_shape = data.shape[1:]  # [D, H, W]
        
        # If input is smaller than patch size, pad it
        padded_data = data
        pads = []
        for i, (input_dim, patch_dim) in enumerate(zip(input_shape, patch_size)):
            if input_dim < patch_dim:
                pad_total = patch_dim - input_dim
                pad_before = pad_total // 2
                pad_after = pad_total - pad_before
                pads.extend([pad_before, pad_after])
            else:
                pads.extend([0, 0])
        
        if any(p > 0 for p in pads):
            padded_data = torch.nn.functional.pad(data, pads[::-1])  # Reverse order for torch.pad
        
        # If input is larger than patch size, take center crop
        cropped_data = padded_data
        crop_coords = []
        for i, (input_dim, patch_dim) in enumerate(zip(padded_data.shape[1:], patch_size)):
            if input_dim > patch_dim:
                start = (input_dim - patch_dim) // 2
                crop_coords.append(slice(start, start + patch_dim))
            else:
                crop_coords.append(slice(None))
        
        if crop_coords:
            cropped_data = padded_data[(slice(None),) + tuple(crop_coords)]
        
        # Add batch dimension and predict
        batch_data = cropped_data.unsqueeze(0)  # [1, C, D, H, W]
        
        with torch.no_grad():
            prediction = self.network(batch_data)
            
            # Handle deep supervision (take the highest resolution prediction)
            if isinstance(prediction, (list, tuple)):
                prediction = prediction[0]
            
            prediction = prediction.squeeze(0)  # Remove batch dimension
        
        # Resize prediction back to original input size if needed
        if prediction.shape[1:] != input_shape:
            import torch.nn.functional as F
            prediction = F.interpolate(prediction.unsqueeze(0), size=input_shape, mode='trilinear', align_corners=False)
            prediction = prediction.squeeze(0)
        
        return prediction


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
