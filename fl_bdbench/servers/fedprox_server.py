"""
FedProx server implementation for FL.

Reference: "Federated Optimization in Heterogeneous Networks"
https://arxiv.org/abs/1812.06127
"""
from fl_bdbench.servers.defense_categories import ClientSideDefenseServer
from fl_bdbench.utils.logging_utils import log
from logging import INFO
from typing import Tuple, Dict, Any

class FedProxServer(ClientSideDefenseServer):
    """
    FedProx server implementation.

    FedProx adds a proximal term to the client's local optimization problem to limit
    the impact of client drift in heterogeneous networks.

    The proximal term is added to the client's loss function:
    L_i(w) = F_i(w) + (μ/2) * ||w - w_t||^2

    where:
    - F_i(w) is the original loss function
    - μ is the proximal term coefficient
    - w_t is the global model parameters at round t
    """

    def __init__(self, server_config, server_type="fedprox", proximal_mu=0.01):
        """
        Initialize the FedProx server.

        Args:
            server_config: Dictionary containing configuration
            server_type: Type of server
            proximal_mu: Coefficient for the proximal term (default: 0.01)
        """
        super(FedProxServer, self).__init__(server_config, server_type)
        self.proximal_mu = proximal_mu
        log(INFO, f"Initialized FedProx server with proximal_mu={proximal_mu}")

    def train_package(self, client_type: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Send the init_args and train_package to ClientApp based on the client type.
        Extends the base implementation to include the proximal term.
        """
        init_args, train_package = super().train_package(client_type)

        # Add proximal term to train_package
        train_package["proximal_mu"] = self.proximal_mu

        return init_args, train_package
