# client_app.py

import os
import json
import flwr as fl
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, NDArrays

from task import FedNnUNetTrainer
import warnings
warnings.filterwarnings("ignore")


def get_nnunet_preprocessed_path():
    """Get nnUNet preprocessed path from environment variable with fallback."""
    # Try environment variable first
    nnunet_path = os.environ.get("nnUNet_preprocessed")
    if nnunet_path and os.path.exists(nnunet_path):
        print(f"[Client] Using nnUNet preprocessed path from environment: {nnunet_path}")
        return nnunet_path
    
    # Fallback to hardcoded path with warning
    fallback_path = "/local/projects-t3/isaiahlab/nnUNet_preprocessed/"
    if os.path.exists(fallback_path):
        print(f"[Client] WARNING: nnUNet_preprocessed environment variable not set.")
        print(f"[Client] Using fallback path: {fallback_path}")
        print(f"[Client] Please set: export nnUNet_preprocessed=/path/to/your/nnUNet_preprocessed")
        return fallback_path
    
    # Error if no valid path found
    raise RuntimeError(
        f"nnUNet preprocessed directory not found. Please set the nnUNet_preprocessed environment variable:\n"
        f"export nnUNet_preprocessed=/path/to/your/nnUNet_preprocessed\n"
        f"Or ensure the directory exists at: {fallback_path}"
    )

class NnUNet3DFullresClient(NumPyClient):
    """
    Flower NumPyClient that uses FedNnUNetTrainer in 3D fullres mode.
    """

    def __init__(
        self,
        client_id: int,
        plans_path: str,
        dataset_json: str,
        configuration: str,
        dataset_fingerprint: str | None = None,
        max_total_epochs: int = 50,
        local_epochs_per_round: int = 2,
    ):
        super().__init__()
        self.client_id = client_id
        
        # Load JSON files
        import json
        import torch
        with open(plans_path, "r") as f:
            plans_dict = json.load(f)
        with open(dataset_json, "r") as f:
            dataset_dict = json.load(f)
        
        self.trainer = FedNnUNetTrainer(
            plans=plans_dict,
            configuration=configuration,
            fold=0,
            dataset_json=dataset_dict,
            device=torch.device("cuda:0"),
        )
        self.trainer.max_num_epochs = max_total_epochs
        self.local_epochs_per_round = local_epochs_per_round
        self.param_keys = None
        self.dataset_fingerprint_path = dataset_fingerprint or os.path.join(
            os.path.dirname(dataset_json),
            "dataset_fingerprint.json",
        )
        self.dataset_json_path = dataset_json
        self.num_training_cases = self._count_training_cases()
        self.local_fingerprint = self._load_local_fingerprint()
        
        # Best validation tracking for PyTorch model saving
        self.best_validation_dice = 0.0
        self.best_round = 0
        self.output_dir = None

    def _load_local_fingerprint(self) -> dict:
        if self.dataset_fingerprint_path and os.path.exists(self.dataset_fingerprint_path):
            try:
                with open(self.dataset_fingerprint_path, "r") as f:
                    return json.load(f)
            except Exception as exc:
                print(f"[Client {self.client_id}] Could not load fingerprint: {exc}")
        else:
            print(
                f"[Client {self.client_id}] Fingerprint file not found at {self.dataset_fingerprint_path}"
            )
        return {}

    def _count_training_cases(self) -> int:
        """Return number of training cases listed in dataset.json."""
        try:
            with open(self.dataset_json_path, "r") as f:
                data = json.load(f)
            
            # nnUNet dataset.json uses "numTraining" field for total training cases
            num_training = data.get("numTraining", 0)
            print(f"[Client {self.client_id}] Found {num_training} total training cases in dataset.json")
            
            # However, for federated learning, we should return the actual number of cases
            # this client will train on (which depends on the cross-validation split)
            # For now, return the total and we'll get the actual split count from the trainer
            return num_training
        except Exception as exc:
            print(f"[Client {self.client_id}] Could not parse dataset.json: {exc}")
            return 1

    def _get_actual_training_count(self) -> int:
        """Get the actual number of training cases this client is using from the trainer's split."""
        try:
            if hasattr(self.trainer, 'was_initialized') and self.trainer.was_initialized:
                # Get the training split from the trainer
                tr_keys, val_keys = self.trainer.do_split()
                actual_count = len(tr_keys)
                print(f"[Client {self.client_id}] Trainer split: {actual_count} training cases, {len(val_keys)} validation cases")
                return actual_count
            else:
                # Fallback to dataset.json count if trainer not initialized
                print(f"[Client {self.client_id}] Trainer not initialized, falling back to dataset.json count")
                return self.num_training_cases
        except Exception as exc:
            print(f"[Client {self.client_id}] Error getting training count from trainer: {exc}")
            # Fallback to the original dataset count
            return self.num_training_cases if self.num_training_cases > 0 else 25  # Default for prostate dataset

    def _apply_global_fingerprint(self, global_fingerprint: dict):
        """Apply global fingerprint to local dataset_fingerprint.json file."""
        try:
            # Update local fingerprint file with global statistics
            fingerprint_path = self.dataset_fingerprint_path
            if os.path.exists(fingerprint_path):
                with open(fingerprint_path, "w") as f:
                    json.dump(global_fingerprint, f, indent=2)
                print(f"[Client {self.client_id}] Applied global fingerprint to {fingerprint_path}")
            else:
                print(f"[Client {self.client_id}] Warning: fingerprint file not found at {fingerprint_path}")
        except Exception as exc:
            print(f"[Client {self.client_id}] Error applying global fingerprint: {exc}")

    def set_output_directory(self, output_dir: str):
        """Set the output directory for saving PyTorch model checkpoints"""
        self.output_dir = output_dir
        # Create client-specific subdirectory
        import os
        client_output_dir = os.path.join(output_dir, f"client_{self.client_id}")
        os.makedirs(client_output_dir, exist_ok=True)
        self.client_output_dir = client_output_dir
        print(f"[Client {self.client_id}] Output directory set to: {client_output_dir}")

    def get_parameters(self, config) -> NDArrays:
        """
        Return current model parameters as a list of numpy arrays
        (ordered consistently for Flower).
        """
        if not self.trainer.was_initialized:
            self.trainer.initialize()

        weights_dict = self.trainer.get_weights()
        self.param_keys = list(weights_dict.keys())
        return list(weights_dict.values())

    def fit(self, parameters: NDArrays, config):
        """
        Receive global model params, do local partial training, return updated params + metrics.
        Implements kaapana-style federated training with fingerprint handling.
        """
        federated_round = config.get("server_round", 1)
        
        # Handle preprocessing round (federated_round = -2) - share fingerprint only
        if federated_round == -2:
            print(f"[Client {self.client_id}] Preprocessing round - sharing fingerprint")
            if not self.trainer.was_initialized:
                self.trainer.initialize()
            
            # Get initial parameters for consistency
            if self.param_keys is None:
                local_sd = self.trainer.get_weights()
                self.param_keys = list(local_sd.keys())
                
            initial_params = [local_sd[k] for k in self.param_keys]
            
            # Create a simple fingerprint summary for metrics
            fp_summary = {}
            if self.local_fingerprint:
                fp_summary["num_cases"] = len(self.local_fingerprint.get("spacings", []))
                # Get first modality stats if available
                if "foreground_intensity_properties_per_channel" in self.local_fingerprint:
                    first_mod = list(self.local_fingerprint["foreground_intensity_properties_per_channel"].keys())[0] if self.local_fingerprint["foreground_intensity_properties_per_channel"] else "0"
                    if first_mod in self.local_fingerprint["foreground_intensity_properties_per_channel"]:
                        fp_summary["mean_intensity"] = float(self.local_fingerprint["foreground_intensity_properties_per_channel"][first_mod].get("mean", 0.0))
            
            # Get actual training count for preprocessing phase too
            actual_training_cases = self._get_actual_training_count()
            
            metrics = {
                "client_id": self.client_id,
                "loss": 0.0,
                "preprocessing_complete": True,
                "fingerprint_cases": fp_summary.get("num_cases", 0),
                "fingerprint_mean": fp_summary.get("mean_intensity", 0.0),
                "actual_training_cases": actual_training_cases
            }
            return initial_params, actual_training_cases, metrics
        
        # Handle initialization round (federated_round = -1) - apply global fingerprint
        if federated_round == -1:
            print(f"[Client {self.client_id}] Initialization round")
            # global_fingerprint = config.get("global_fingerprint", {})
            # if global_fingerprint:
            #     self._apply_global_fingerprint(global_fingerprint)
            
            if not self.trainer.was_initialized:
                self.trainer.initialize()
                
            if self.param_keys is None:
                local_sd = self.trainer.get_weights()
                self.param_keys = list(local_sd.keys())
                
            # Apply received global parameters
            if parameters:
                new_sd = {}
                for k, arr in zip(self.param_keys, parameters):
                    new_sd[k] = arr
                self.trainer.set_weights(new_sd)
                
            updated_dict = self.trainer.get_weights()
            updated_params = [updated_dict[k] for k in self.param_keys]
            
            # Get actual training count for initialization phase too
            actual_training_cases = self._get_actual_training_count()
            
            metrics = {
                "client_id": self.client_id,
                "loss": 0.0,
                "initialization_complete": True,
                "actual_training_cases": actual_training_cases
            }
            return updated_params, actual_training_cases, metrics

        # Regular training rounds (federated_round >= 0)
        print(f"[Client {self.client_id}] Training round {federated_round}")
        
        if self.param_keys is None:
            local_sd = self.trainer.get_weights()
            self.param_keys = list(local_sd.keys())

        # Convert list->dict and apply received parameters
        if parameters:
            new_sd = {}
            for k, arr in zip(self.param_keys, parameters):
                new_sd[k] = arr
            self.trainer.set_weights(new_sd)

        # Local training - use minimal epochs for testing
        local_epochs = config.get("local_epochs", 1)  # Reduced from self.local_epochs_per_round for faster testing
        self.trainer.run_training_round(local_epochs)

        updated_dict = self.trainer.get_weights()
        updated_params = [updated_dict[k] for k in self.param_keys]

        # Get actual number of training cases from the trainer's split
        actual_training_cases = self._get_actual_training_count()
        print(f"[Client {self.client_id}] Using {actual_training_cases} training cases for FedAvg aggregation")

        # Training metrics
        final_loss = (
            self.trainer.all_train_losses[-1] if self.trainer.all_train_losses else 0.0
        )

        metrics = {
            "client_id": self.client_id,
            "loss": final_loss,
            "federated_round": federated_round,
            "local_epochs_completed": local_epochs,
            "actual_training_cases": actual_training_cases
        }
        
        # Run validation if requested
        should_validate = config.get("validate", False)
        if should_validate:
            try:
                print(f"[Client {self.client_id}] Running validation...")
                validation_results = self.trainer.run_validation_round()
                metrics["validation_dice"] = validation_results
                current_dice = validation_results.get('mean', 0)
                print(f"[Client {self.client_id}] Validation Dice: {current_dice:.4f}")
                
                # Save PyTorch model if validation improved and output directory is set
                if hasattr(self, 'client_output_dir') and self.client_output_dir:
                    is_best = current_dice > self.best_validation_dice
                    if is_best:
                        self.best_validation_dice = current_dice
                        self.best_round = federated_round
                        print(f"[Client {self.client_id}] New best validation Dice: {current_dice:.4f}")
                    
                    # Save PyTorch checkpoint only if it's the best model
                    if is_best:
                        try:
                            checkpoint_path = self.trainer.save_best_checkpoint_pytorch(
                                output_dir=self.client_output_dir,
                                round_num=federated_round,
                                validation_dice=current_dice,
                                is_best=True
                            )
                            if checkpoint_path:
                                metrics["pytorch_checkpoint_saved"] = checkpoint_path
                                metrics["best_model_updated"] = True
                                print(f"[Client {self.client_id}] Saved best model checkpoint: {checkpoint_path}")
                        except Exception as save_e:
                            print(f"[Client {self.client_id}] Failed to save PyTorch checkpoint: {save_e}")
                    else:
                        print(f"[Client {self.client_id}] Validation Dice ({current_dice:.4f}) <= best ({self.best_validation_dice:.4f}), not saving model")
                        
            except Exception as e:
                print(f"[Client {self.client_id}] Validation failed: {e}")
                # Continue without validation metrics
        
        return updated_params, actual_training_cases, metrics

    def evaluate(self, parameters: NDArrays, config):
        """
        Evaluate global model on local validation set (optional).
        """
        if self.param_keys is None:
            local_sd = self.trainer.get_weights()
            self.param_keys = list(local_sd.keys())

        # Convert list->dict, set local model
        new_sd = {}
        for k, arr in zip(self.param_keys, parameters):
            new_sd[k] = arr
        self.trainer.set_weights(new_sd)

        # Example local validation
        val_loss = 0.5  # or run actual inference
        return val_loss, self.num_training_cases, {"val_loss": val_loss}


def client_fn(context: Context):
    """
    This callback is used by Flower 1.13 to create the client instance.
    Typically you read environment variables or config to set up the trainer.
    """
    client_id = context.node_config.get("partition-id", 0)

    task_name = os.environ.get("TASK_NAME", "Dataset005_Prostate") # Default to Dataset005_Prostate, change as needed
    preproc_root = get_nnunet_preprocessed_path()
    plans_path = os.path.join(preproc_root, task_name, "nnUNetPlans.json")
    dataset_json = os.path.join(preproc_root, task_name, "dataset.json")
    dataset_fp = os.path.join(preproc_root, task_name, "dataset_fingerprint.json")
    configuration = os.environ.get("NNUNET_CONFIG", "3d_fullres")
    out_root = os.environ.get("OUTPUT_ROOT", "/local/projects-t3/isaiahlab/nnunet_output")
    output_folder = os.path.join(out_root, f"client_{client_id}")

    # Create the client
    return NnUNet3DFullresClient(
        client_id=client_id,
        plans_path=plans_path,
        dataset_json=dataset_json,
        configuration=configuration,
        dataset_fingerprint=dataset_fp,
        max_total_epochs=50,
        local_epochs_per_round=2,
    ).to_client()


# Flower 1.13+ recommended usage: a ClientApp
app = ClientApp(client_fn=client_fn)

if __name__ == "__main__":
    print("This is a Flower ClientApp. Typically run with:")
    print("flower-supernode --client-app=client_app.py:app")
