"""
Main entry point.
"""
import hydra
import omegaconf
import torch
import os
import ray

from rich.traceback import install
from fl_bdbench.servers.base_server import BaseServer
from fl_bdbench.utils import system_startup
from omegaconf import DictConfig
from hydra.utils import instantiate

@hydra.main(config_path="config", config_name="defaults.yaml", version_base=None)
def main(config: DictConfig):
    system_startup(config)
    aggregator = config["aggregator"]
    server : BaseServer = instantiate(config.aggregator_config[aggregator], server_config=config, _recursive_=False)
    server.run_experiment()

if __name__ == "__main__":
    # Rich traceback and suppress traceback from hydra, omegaconf, and torch
    install(show_locals=False, suppress=[hydra, omegaconf, torch, ray])
    os.environ["HYDRA_FULL_ERROR"] = "1" # For detailed error messages
    main()
