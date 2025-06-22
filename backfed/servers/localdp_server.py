"""
FedProx server implementation for FL.

Reference: https://www.ndss-symposium.org/wp-content/uploads/2022-54-paper.pdf
"""
from backfed.servers.defense_categories import ClientSideDefenseServer
from backfed.utils.logging_utils import log
from logging import INFO
from typing import Tuple, Dict, Any

class LocalDPServer(ClientSideDefenseServer):

    def __init__(self, server_config, server_type="localdp", std_dev=0.01, clipping_norm=1.0):
        """
        Initialize the LocalDP server.

        Args:
            server_config: Dictionary containing configuration
            server_type: Type of server
            std_dev: Standard deviation of Gaussian noise
            clipping_norm: Norm threshold of gradients at each epoch
        """
        super(LocalDPServer, self).__init__(server_config, server_type)
        self.std_dev = std_dev
        log(INFO, f"Initialized LocalDPServer server with std_dev={std_dev} and clipping_norm={clipping_norm}")

    def train_package(self, client_type: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Send the init_args and train_package to ClientApp based on the client type.
        Extends the base implementation to include the std_dev and clipping_norm.
        """
        init_args, train_package = super().train_package(client_type)

        train_package["std_dev"] = self.std_dev
        train_package["clipping_norm"] = self.clipping_norm

        return init_args, train_package
