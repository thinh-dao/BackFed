"""
Bulyan server implementation for FL.

Bulyan combines a selection phase (Multi-Krum) with a robust aggregation phase (Trimmed Mean).
Selection: choose theta = n - 2f client updates with the smallest Multi-Krum scores.
Aggregation: for each parameter coordinate, compute the average of the middle values after
trimming f smallest and f largest among the selected theta updates.

References:
- "The Hidden Vulnerability of Distributed Learning in Byzantium" (El Mhamdi, Guerraoui, Rouault)
- "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent" (Blanchard et al.)
"""
import torch
from typing import List, Tuple

from backfed.servers.fedavg_server import FedAvgServer
from backfed.utils.logging_utils import log
from backfed.const import ModelUpdate, client_id, num_examples
from logging import INFO, WARNING

class BulyanServer(FedAvgServer):
    """
    Server that implements Bulyan aggregation:
      - Multi-Krum selection of theta = n - 2f candidates
      - Coordinate-wise trimmed mean that removes f smallest and f largest values
    """

    def __init__(self, server_config, server_type="bulyan", num_malicious_clients=None, oracle=False, eta=1.0):
        """
        Initialize the Bulyan server.

        Args:
            server_config: Hydra config
            server_type: Type name
            num_malicious_clients: Number of malicious clients (f) when oracle=False
            oracle: If True, infer f each round from client_manager
        """
        super(BulyanServer, self).__init__(server_config, server_type)
        self.eta = eta
        self.oracle = oracle

        if self.config.num_clients_per_round <= 4*num_malicious_clients + 3:
            log(WARNING, f"n({self.config.num_clients_per_round}) <= 4f({num_malicious_clients}) + 3, which violates assumption of Bulyan.")

        # 0 < f < min(n-2, n//2)
        if num_malicious_clients < 0 or num_malicious_clients > min(self.config.num_clients_per_round - 2, self.config.num_clients_per_round // 2):
            raise ValueError(f"Invalid number of malicious clients: {num_malicious_clients}. Must be in [0, {min(self.config.num_clients_per_round - 2, self.config.num_clients_per_round // 2)}]")

        if self.oracle:
            self.num_malicious_clients = None
            log(INFO, "Initialized Bulyan server with oracle=True (f inferred each round)")
        else:
            self.num_malicious_clients = num_malicious_clients if num_malicious_clients is not None else 0
            log(INFO, f"Initialized Bulyan server with oracle=False, f={self.num_malicious_clients}")

    def aggregate_client_updates(self, client_updates: List[Tuple[client_id, num_examples, ModelUpdate]]) -> bool:
        """
        Aggregate client updates using Bulyan:
          1) Multi-Krum selection of theta = n - 2f candidates
          2) Coordinate-wise trimmed mean that trims f values on each side
        """
        if len(client_updates) == 0:
            return False

        # Determine f (number of malicious clients)
        if self.oracle:
            f = len(self.client_manager.malicious_clients_per_round[self.current_round])
        else:
            f = self.num_malicious_clients

        n = len(client_updates)
        theta = n - 2 * f

        # Extract client parameters
        flattened_params = []
        client_ids = []
        updates = []

        for cid, _, update in client_updates:
            updates.append(update)
            flattened_params.append(self.parameters_dict_to_vector(update).cpu())
            client_ids.append(cid)

        # Pairwise squared Euclidean distances
        distances = torch.zeros(n, n)
        for i in range(n):
            for j in range(i + 1, n):
                dist = torch.linalg.norm(flattened_params[i] - flattened_params[j]) ** 2
                distances[i, j] = dist
                distances[j, i] = dist

        # Multi-Krum scores: sum of closest (n - f - 2) neighbor distances
        selected_indices = []
        available = list(range(n))
        for _ in range(theta):
            cur_n = len(available)
            cur_neighbors = max(cur_n - f - 2, 1)  # Krum on the current pool

            scores = torch.zeros(cur_n)
            for idx_i, i in enumerate(available):
                row = distances[i][available]  # distances to current pool
                vals, _ = torch.sort(row)
                # first value is self; take the next cur_neighbors
                scores[idx_i] = torch.sum(vals[1:1+cur_neighbors])

            best_local = torch.argmin(scores).item()
            selected_indices.append(available.pop(best_local))

        selected_client_ids = [client_ids[i] for i in selected_indices]
        log(INFO, f"Bulyan selection (iterative Multi-Krum) chose clients: {selected_client_ids}")

        if theta - 2 * f <= 0:
            log(WARNING, f"Bulyan trimmed mean invalid: theta - 2f = {theta} - 2*{f} <= 0. Falling back to mean over selected.")
            return super().aggregate_client_updates(
                [client_updates[i] for i in selected_indices]
            )
    
        # Accumulate gradient updates with coordinate-wise trimmed mean 
        weight_accumulator = {
            name: torch.zeros_like(param, device=self.device, dtype=torch.float32)
            for name, param in self.global_model.state_dict().items()
        }    

        # Coordinate-wise trimmed mean over selected set
        for name, param in self.global_model.state_dict().items():
            if any(pattern in name for pattern in self.ignore_weights):
                continue

            # We only perform trimmed-mean on trainable params
            if name not in self.trainable_names:
                for i in selected_indices:
                    param_update = updates[i][name].to(device=self.device, dtype=torch.float32)
                    weight_accumulator[name].add_(param_update * 1/theta)
            else:
                # Stack the selected updates for this layer
                layer_updates = torch.stack([
                    updates[i][name].to(device=self.device, dtype=torch.float32)
                    for i in selected_indices
                ])

                # Sort along client dimension and trim f on each side
                sorted_updates, _ = torch.sort(layer_updates, dim=0)
                low = f
                high = theta - f
                trimmed = sorted_updates[low:high]  # shape: [theta - 2f, ...]
                mean_update = torch.mean(trimmed, dim=0)

                # Update weight_accumulator
                weight_accumulator[name].copy_(mean_update.to(param.device))
        
        # Update global model with learning rate
        for name, param in self.global_model.state_dict().items():
            if any(pattern in name for pattern in self.ignore_weights):
                continue
            param.data.add_(weight_accumulator[name] * self.eta)
        return True

    def __repr__(self) -> str:
        f = self.num_malicious_clients if not self.oracle else "oracle"
        return f"Bulyan(f={f})"