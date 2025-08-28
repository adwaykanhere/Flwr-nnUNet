# wandb_integration.py

import os
import numpy as np
import torch
from typing import Dict, Optional, Union, List, Tuple
import warnings

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    warnings.warn("wandb not available. Install with: pip install wandb")


class WandBLogger:
    """
    Utility class for Weights & Biases integration in federated nnUNet.
    Provides environment-based configuration and federated learning specific logging.
    """
    
    def __init__(self, 
                 run_type: str = "client",  # "client", "server", "global"
                 client_id: Optional[int] = None,
                 dataset_name: Optional[str] = None,
                 modality: Optional[str] = None,
                 project_suffix: str = ""):
        """
        Initialize wandb logger with federated learning context.
        
        Args:
            run_type: Type of run ("client", "server", "global")
            client_id: Client identifier for client runs
            dataset_name: Dataset name for tagging
            modality: Imaging modality for tagging
            project_suffix: Additional project name suffix
        """
        self.enabled = self._check_wandb_enabled()
        self.run_type = run_type
        self.client_id = client_id
        self.dataset_name = dataset_name
        self.modality = modality
        self.run = None
        
        if self.enabled and WANDB_AVAILABLE:
            self._initialize_wandb(project_suffix)
    
    def _check_wandb_enabled(self) -> bool:
        """Check if wandb is enabled through environment variables."""
        if not WANDB_AVAILABLE:
            return False
        
        # Check if explicitly disabled
        if os.environ.get("WANDB_DISABLED", "false").lower() == "true":
            return False
        
        # Check if wandb mode is disabled
        if os.environ.get("WANDB_MODE", "online") == "disabled":
            return False
            
        return True
    
    def _initialize_wandb(self, project_suffix: str):
        """Initialize wandb run with appropriate configuration."""
        try:
            # Get project configuration
            project_name = os.environ.get("WANDB_PROJECT", "federated-nnunet")
            if project_suffix:
                project_name = f"{project_name}-{project_suffix}"
            
            entity = os.environ.get("WANDB_ENTITY", None)
            
            # Create run name
            if self.run_type == "client" and self.client_id is not None:
                run_name = f"client-{self.client_id}"
                if self.dataset_name:
                    run_name += f"-{self.dataset_name}"
                if self.modality:
                    run_name += f"-{self.modality}"
            elif self.run_type == "server":
                run_name = "federation-server"
            else:
                run_name = f"{self.run_type}-run"
            
            # Create tags
            tags = [self.run_type, "federated-learning", "nnunet"]
            if self.dataset_name:
                tags.append(self.dataset_name)
            if self.modality:
                tags.append(self.modality)
            
            # Initialize wandb run
            self.run = wandb.init(
                project=project_name,
                entity=entity,
                name=run_name,
                tags=tags,
                group=f"federation-{os.environ.get('FEDERATION_ID', 'default')}",
                job_type=self.run_type,
                reinit=True,
                settings=wandb.Settings(start_method="thread")
            )
            
            print(f"[WandB] Initialized {self.run_type} run: {run_name}")
            
        except Exception as e:
            print(f"[WandB] Failed to initialize: {e}")
            self.enabled = False
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None):
        """Log metrics to wandb."""
        if not self.enabled or not self.run:
            return
        
        try:
            self.run.log(metrics, step=step)
        except Exception as e:
            print(f"[WandB] Failed to log metrics: {e}")
    
    def log_medical_images(self, 
                          images: np.ndarray,
                          masks: Optional[np.ndarray] = None,
                          predictions: Optional[np.ndarray] = None,
                          prefix: str = "medical",
                          step: Optional[int] = None,
                          max_images: int = 4):
        """
        Log medical images with optional segmentation masks and predictions.
        
        Args:
            images: Input medical images [B, C, H, W, D] or [B, C, H, W]
            masks: Ground truth segmentation masks [B, H, W, D] or [B, H, W]
            predictions: Predicted segmentation [B, H, W, D] or [B, H, W]
            prefix: Prefix for wandb image keys
            step: Training step
            max_images: Maximum number of images to log
        """
        if not self.enabled or not self.run:
            return
        
        # Check if image logging is enabled
        if not os.environ.get("WANDB_LOG_IMAGES", "true").lower() == "true":
            return
        
        try:
            log_dict = {}
            batch_size = min(images.shape[0], max_images)
            
            for i in range(batch_size):
                # Process image
                img = self._prepare_medical_image(images[i])
                if img is not None:
                    log_dict[f"{prefix}_image_{i}"] = wandb.Image(img)
                
                # Process mask
                if masks is not None:
                    mask = self._prepare_segmentation_mask(masks[i])
                    if mask is not None:
                        log_dict[f"{prefix}_mask_{i}"] = wandb.Image(mask)
                
                # Process predictions
                if predictions is not None:
                    pred = self._prepare_segmentation_mask(predictions[i])
                    if pred is not None:
                        log_dict[f"{prefix}_prediction_{i}"] = wandb.Image(pred)
                
                # Create overlay if both image and mask/prediction available
                if img is not None and masks is not None:
                    overlay = self._create_segmentation_overlay(
                        images[i], masks[i], predictions[i] if predictions is not None else None
                    )
                    if overlay is not None:
                        log_dict[f"{prefix}_overlay_{i}"] = wandb.Image(overlay)
            
            if log_dict:
                self.run.log(log_dict, step=step)
                
        except Exception as e:
            print(f"[WandB] Failed to log medical images: {e}")
    
    def _prepare_medical_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Prepare medical image for wandb logging."""
        try:
            # Handle different image formats
            if len(image.shape) == 4:  # [C, H, W, D] - 3D volume
                # Take middle slice
                image = image[0, :, :, image.shape[3] // 2]
            elif len(image.shape) == 3:  # [C, H, W] - 2D image or single slice
                if image.shape[0] > 1:  # Multi-channel
                    image = image[0]  # Take first channel
                else:
                    image = image.squeeze(0)
            elif len(image.shape) == 2:  # [H, W] - already 2D
                pass
            else:
                return None
            
            # Normalize for visualization
            image = self._normalize_for_display(image)
            
            # Convert to uint8
            image = (image * 255).astype(np.uint8)
            
            return image
            
        except Exception as e:
            print(f"[WandB] Error preparing medical image: {e}")
            return None
    
    def _prepare_segmentation_mask(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """Prepare segmentation mask for wandb logging."""
        try:
            # Handle different mask formats
            if len(mask.shape) == 3:  # [H, W, D] - 3D mask
                # Take middle slice
                mask = mask[:, :, mask.shape[2] // 2]
            elif len(mask.shape) == 2:  # [H, W] - already 2D
                pass
            else:
                return None
            
            # Convert to colored mask for better visualization
            colored_mask = self._apply_segmentation_colors(mask)
            
            return colored_mask
            
        except Exception as e:
            print(f"[WandB] Error preparing segmentation mask: {e}")
            return None
    
    def _normalize_for_display(self, image: np.ndarray) -> np.ndarray:
        """Normalize image for display based on modality."""
        # Get modality-specific normalization
        if self.modality == "CT":
            # CT window/level normalization
            window_center = float(os.environ.get("CT_WINDOW_CENTER", "40"))
            window_width = float(os.environ.get("CT_WINDOW_WIDTH", "400"))
            image = np.clip((image - (window_center - window_width/2)) / window_width, 0, 1)
        elif self.modality == "MR":
            # MR normalization (percentile-based)
            p2, p98 = np.percentile(image, (2, 98))
            image = np.clip((image - p2) / (p98 - p2), 0, 1)
        else:
            # Generic normalization
            if image.min() < image.max():
                image = (image - image.min()) / (image.max() - image.min())
            else:
                image = np.zeros_like(image)
        
        return image
    
    def _apply_segmentation_colors(self, mask: np.ndarray) -> np.ndarray:
        """Apply colors to segmentation mask."""
        # Define color map for different classes
        colors = [
            [0, 0, 0],       # Background - black
            [255, 0, 0],     # Class 1 - red
            [0, 255, 0],     # Class 2 - green
            [0, 0, 255],     # Class 3 - blue
            [255, 255, 0],   # Class 4 - yellow
            [255, 0, 255],   # Class 5 - magenta
            [0, 255, 255],   # Class 6 - cyan
            [128, 128, 128], # Class 7 - gray
        ]
        
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        
        for class_id in range(min(len(colors), int(mask.max()) + 1)):
            colored_mask[mask == class_id] = colors[class_id]
        
        return colored_mask
    
    def _create_segmentation_overlay(self, 
                                   image: np.ndarray, 
                                   mask: np.ndarray,
                                   prediction: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Create overlay of image with segmentation mask."""
        try:
            # Prepare base image
            base_img = self._prepare_medical_image(image)
            if base_img is None:
                return None
            
            # Convert to RGB
            if len(base_img.shape) == 2:
                base_img = np.stack([base_img, base_img, base_img], axis=-1)
            
            # Prepare mask
            colored_mask = self._prepare_segmentation_mask(mask)
            if colored_mask is None:
                return None
            
            # Create overlay
            alpha = 0.3  # Transparency for overlay
            overlay = base_img.astype(float) * (1 - alpha) + colored_mask.astype(float) * alpha
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)
            
            return overlay
            
        except Exception as e:
            print(f"[WandB] Error creating overlay: {e}")
            return None
    
    def log_model_artifact(self, 
                          model_path: str, 
                          artifact_name: str,
                          artifact_type: str = "model",
                          metadata: Optional[Dict] = None):
        """Log model as wandb artifact."""
        if not self.enabled or not self.run:
            return
        
        try:
            artifact = wandb.Artifact(
                name=artifact_name,
                type=artifact_type,
                metadata=metadata or {}
            )
            artifact.add_file(model_path)
            self.run.log_artifact(artifact)
            
            print(f"[WandB] Logged model artifact: {artifact_name}")
            
        except Exception as e:
            print(f"[WandB] Failed to log model artifact: {e}")
    
    def log_federated_round_metrics(self,
                                  round_num: int,
                                  client_metrics: List[Dict],
                                  aggregation_info: Optional[Dict] = None):
        """Log federated learning round metrics."""
        if not self.enabled or not self.run:
            return
        
        try:
            # Aggregate client metrics
            if client_metrics:
                avg_loss = np.mean([m.get('loss', 0) for m in client_metrics])
                avg_dice = np.mean([m.get('validation_dice_mean', 0) for m in client_metrics if 'validation_dice_mean' in m])
                
                metrics = {
                    f"federation/round": round_num,
                    f"federation/avg_client_loss": avg_loss,
                    f"federation/num_participating_clients": len(client_metrics),
                }
                
                if avg_dice > 0:
                    metrics[f"federation/avg_client_dice"] = avg_dice
                
                # Add aggregation info
                if aggregation_info:
                    for key, value in aggregation_info.items():
                        if isinstance(value, (int, float)):
                            metrics[f"federation/{key}"] = value
                
                self.run.log(metrics, step=round_num)
                
        except Exception as e:
            print(f"[WandB] Failed to log federated round metrics: {e}")
    
    def finish(self):
        """Finish wandb run."""
        if self.enabled and self.run:
            try:
                self.run.finish()
                print(f"[WandB] Finished {self.run_type} run")
            except Exception as e:
                print(f"[WandB] Error finishing run: {e}")


def get_wandb_logger(run_type: str = "client",
                    client_id: Optional[int] = None,
                    dataset_name: Optional[str] = None,
                    modality: Optional[str] = None,
                    project_suffix: str = "") -> WandBLogger:
    """
    Factory function to create WandB logger with environment-based configuration.
    
    Args:
        run_type: Type of run ("client", "server", "global")
        client_id: Client identifier for client runs
        dataset_name: Dataset name for tagging
        modality: Imaging modality for tagging
        project_suffix: Additional project name suffix
    
    Returns:
        WandBLogger instance
    """
    return WandBLogger(
        run_type=run_type,
        client_id=client_id,
        dataset_name=dataset_name,
        modality=modality,
        project_suffix=project_suffix
    )