# server_app.py

import flwr as fl
from flwr.common import Context, FitRes
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy

from flowernnunet.task import merge_local_fingerprints


class KaapanaStyleStrategy(FedAvg):
    """
    Example FedAvg strategy that collects local fingerprints in round 1,
    merges them in round 2, then does normal FedAvg from round 3 onward.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fingerprints_collected = []
        self.global_fingerprint = None

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
            print("[Server] Collected local fingerprints in round 1.")
            return super().aggregate_fit(rnd, results, failures)

        if rnd == 2 and self.fingerprints_collected:
            # Merge them into a global fingerprint
            self.global_fingerprint = merge_local_fingerprints(self.fingerprints_collected)
            print("[Server] Merged fingerprint =>", self.global_fingerprint)
            return super().aggregate_fit(rnd, results, failures)

        return super().aggregate_fit(rnd, results, failures)


def server_fn(context: Context):
    strategy = KaapanaStyleStrategy(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_available_clients=2,
    )
    config = ServerConfig(num_rounds=5)
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)

if __name__ == "__main__":
    print("This is a Flower ServerApp. Typically run with:")
    print("flower-supernode --server-app=server_app.py:app")
