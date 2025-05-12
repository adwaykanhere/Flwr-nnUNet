# client_app.py

import os
import flwr as fl
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, FitRes, EvaluateRes, NDArrays

from flowernnunet.task import FedNnUNetTrainer
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
        output_folder: str,
        max_total_epochs: int = 50,
        local_epochs_per_round: int = 2,
    ):
        super().__init__()
        self.client_id = client_id
        self.trainer = FedNnUNetTrainer(
            plans=plans_path,
            configuration=configuration,
            fold=0,
            dataset_json=dataset_json,
            output_folder=output_folder,
            max_num_epochs=max_total_epochs,
        )
        self.local_epochs_per_round = local_epochs_per_round
        self.param_keys = None

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
        """
        if self.param_keys is None:
            local_sd = self.trainer.get_weights()
            self.param_keys = list(local_sd.keys())

        # Convert list->dict
        new_sd = {}
        for k, arr in zip(self.param_keys, parameters):
            new_sd[k] = arr
        self.trainer.set_weights(new_sd)

        # Local training
        local_epochs = config.get("local_epochs", self.local_epochs_per_round)
        self.trainer.run_training_round(local_epochs)

        updated_dict = self.trainer.get_weights()
        updated_params = [updated_dict[k] for k in self.param_keys]

        # Example local metrics
        final_loss = self.trainer.all_train_losses[-1] if self.trainer.all_train_losses else 0.0
        # Example local “fingerprint” (just dummy data)
        local_fp = {"mean": 40.0 + self.client_id, "std": 7.5, "n": 5000}

        metrics = {
            "client_id": self.client_id,
            "loss": final_loss,
            "fingerprint": local_fp
        }
        # Put number of local training samples (estimated)
        num_examples = 100
        return FitRes(parameters=updated_params, num_examples=num_examples, metrics=metrics)

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
            loss=val_loss,
            num_examples=50,
            metrics={"val_loss": val_loss},
        )


def client_fn(context: Context):
    """
    This callback is used by Flower 1.13 to create the client instance.
    Typically you read environment variables or config to set up the trainer.
    """
    client_id = context.node_config.get("partition-id", 0)

    task_name = os.environ.get("TASK_NAME", "Dataset005_Prostate")
    # Build your paths accordingly:
    plans_path = f"/Users/akanhere/Documents/nnUNet/nnUNet_preprocessed/{task_name}/nnUNetPlans.json"
    dataset_json = f"/Users/akanhere/Documents/nnUNet/nnUNet_preprocessed/{task_name}/dataset.json"
    configuration = "3d_fullres"  # or read from env
    output_folder = f"/Users/akanhere/Documents/nnUNet/output_client_{client_id}"

    # Create the client
    return NnUNet3DFullresClient(
        client_id=client_id,
        plans_path=plans_path,
        dataset_json=dataset_json,
        configuration=configuration,
        output_folder=output_folder,
        max_total_epochs=50,
        local_epochs_per_round=2,
    ).to_client()


# Flower 1.13+ recommended usage: a ClientApp
app = ClientApp(client_fn=client_fn)

if __name__ == "__main__":
    print("This is a Flower ClientApp. Typically run with:")
    print("flower-supernode --client-app=client_app.py:app")
