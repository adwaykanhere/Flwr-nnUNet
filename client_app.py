# client_app.py

import os
import json
import flwr as fl
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, FitRes, EvaluateRes, NDArrays, Status, Code

from task import FedNnUNetTrainer
import warnings
warnings.filterwarnings("ignore")

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
            unpack_dataset=True,
            device=torch.device("cpu"),
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
            return len(data.get("training", []))
        except Exception as exc:
            print(f"[Client {self.client_id}] Could not parse dataset.json: {exc}")
            return 1

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

    def fit(self, parameters: NDArrays, config) -> FitRes:
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
            
            metrics = {
                "client_id": self.client_id,
                "loss": 0.0,
                "fingerprint": self.local_fingerprint,
                "preprocessing_complete": True
            }
            return FitRes(parameters=initial_params, num_examples=self.num_training_cases, metrics=metrics)
        
        # Handle initialization round (federated_round = -1) - apply global fingerprint
        if federated_round == -1:
            print(f"[Client {self.client_id}] Initialization round - applying global fingerprint")
            global_fingerprint = config.get("global_fingerprint", {})
            if global_fingerprint:
                self._apply_global_fingerprint(global_fingerprint)
            
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
            
            metrics = {
                "client_id": self.client_id,
                "loss": 0.0,
                "initialization_complete": True
            }
            return FitRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=updated_params, 
                num_examples=self.num_training_cases, 
                metrics=metrics
            )

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

        # Local training
        local_epochs = config.get("local_epochs", self.local_epochs_per_round)
        self.trainer.run_training_round(local_epochs)

        updated_dict = self.trainer.get_weights()
        updated_params = [updated_dict[k] for k in self.param_keys]

        # Training metrics
        final_loss = (
            self.trainer.all_train_losses[-1] if self.trainer.all_train_losses else 0.0
        )

        metrics = {
            "client_id": self.client_id,
            "loss": final_loss,
            "federated_round": federated_round,
            "local_epochs_completed": local_epochs
        }
        
        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=updated_params, 
            num_examples=self.num_training_cases, 
            metrics=metrics
        )

    def evaluate(self, parameters: NDArrays, config) -> EvaluateRes:
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
        return EvaluateRes(
            status=Status(code=Code.OK, message="Success"),
            loss=val_loss,
            num_examples=self.num_training_cases,
            metrics={"val_loss": val_loss},
        )


def client_fn(context: Context):
    """
    This callback is used by Flower 1.13 to create the client instance.
    Typically you read environment variables or config to set up the trainer.
    """
    client_id = context.node_config.get("partition-id", 0)

    task_name = os.environ.get("TASK_NAME", "Dataset005_Prostate") # Default to Dataset005_Prostate, change as needed
    preproc_root = os.environ.get("nnUNet_preprocessed", "/mnt/c/Users/adway/Documents/nnUNet_preprocessed")
    plans_path = os.path.join(preproc_root, task_name, "nnUNetPlans.json")
    dataset_json = os.path.join(preproc_root, task_name, "dataset.json")
    dataset_fp = os.path.join(preproc_root, task_name, "dataset_fingerprint.json")
    configuration = os.environ.get("NNUNET_CONFIG", "3d_fullres")
    out_root = os.environ.get("OUTPUT_ROOT", "/mnt/c/Users/adway/Documents/nnunet_output")
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
