# server_app.py

import os
import flwr as fl
from flwr.common import Context, FitRes
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy

from flowernnunet.task import merge_local_fingerprints


class KaapanaStyleStrategy(FedAvg):
    """FedAvg strategy that also aggregates dataset fingerprints."""

    def __init__(self, expected_num_clients: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.expected_num_clients = expected_num_clients
        self.fingerprints_collected: list[dict] = []
        self.global_fingerprint: dict | None = None

    def aggregate_fit(
        self,
        rnd: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[BaseException],
    ):
        print(f"[Server] Round {rnd} results: {len(results)} successes, {len(failures)} failures.")

        if rnd == 1:
            # Gather local fingerprint from each client
            for _, fitres in results:
                fp = fitres.metrics.get("fingerprint", None)
                if fp:
                    self.fingerprints_collected.append(fp)
            print(
                f"[Server] Collected {len(self.fingerprints_collected)}/"
                f"{self.expected_num_clients} fingerprints in round 1."
            )
            return super().aggregate_fit(rnd, results, failures)

        if rnd == 2 and self.fingerprints_collected:
            # Merge them into a global fingerprint once all are received
            if len(self.fingerprints_collected) >= self.expected_num_clients:
                self.global_fingerprint = merge_local_fingerprints(
                    self.fingerprints_collected
                )
                print("[Server] Merged fingerprint =>", self.global_fingerprint)
            else:
                print(
                    f"[Server] Only {len(self.fingerprints_collected)} fingerprints received; skipping merge"
                )
            return super().aggregate_fit(rnd, results, failures)

        return super().aggregate_fit(rnd, results, failures)


def server_fn(context: Context):
    expected_clients = int(os.environ.get("NUM_CLIENTS", 2))
    num_rounds = int(os.environ.get("NUM_ROUNDS", 5))
    strategy = KaapanaStyleStrategy(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_available_clients=expected_clients,
        expected_num_clients=expected_clients,
    )
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)

if __name__ == "__main__":
    print("This is a Flower ServerApp. Typically run with:")
    print("flower-supernode --server-app=server_app.py:app")
