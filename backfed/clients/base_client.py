"""
Base client implementation for FL.
"""
import torch
import torch.nn as nn

from typing import Dict, Any, Tuple, List
from torch.utils.data import DataLoader, Dataset, random_split
from omegaconf import DictConfig
from backfed.const import ModelUpdate, Metrics
from hydra.utils import instantiate

class BaseClient:
    """
    Base class for all FL clients.
    Handles data partitioning, model setup, optimizer, and training logic.
    """
    def __init__(
        self,
        client_id: int,
        dataset: Dataset,
        model: nn.Module,
        client_config: DictConfig,
        client_type: str = "base",
        verbose: bool = True,
    ):
        """
        Initialize the client.
        Args:
            client_id: Unique identifier
            dataset: Client dataset
            model: Training model
            client_config: Dictionary containing training configuration
            client_type: String for client type identification
        """
        self.client_id = client_id
        self.client_config = client_config
        self.client_type = client_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose

        # Set up model, dataloader, optimizer, criterion
        self.model = model.to(self.device)
        self._set_dataloader(dataset)
        self._set_optimizer()
        self._set_criterion()

    def _set_optimizer(self):
        """
        Set up the optimizer for the client. Uses the optimizer specified in the config.
        """
        self.optimizer = instantiate(self.client_config.optimizer, params=self.model.parameters())

    def _set_criterion(self):
        """
        Set up the loss criterion for the client. Defaults to CrossEntropyLoss.
        """
        self.criterion = nn.CrossEntropyLoss()

    def _set_dataloader(self, dataset):
        """
        Set up train and validation data loaders for the client.
        """
        if self.client_config.val_split > 0.0:
            num_val = int(len(dataset) * self.client_config.val_split)
            num_train = len(dataset) - num_val

            self.train_dataset, self.val_dataset = random_split(dataset, [num_train, num_val])
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.client_config["batch_size"], shuffle=True, pin_memory=False)
            self.val_loader = DataLoader(self.val_dataset, batch_size=self.client_config["batch_size"], shuffle=True, pin_memory=False)
        else:
            self.train_dataset = dataset
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.client_config["batch_size"], shuffle=True, pin_memory=False)

    def _check_required_keys(self, train_package: Dict[str, Any], required_keys: List[str] = ["global_state_dict", "server_round"]):
        """
        Check if the required keys are present in the train_package.
        """
        for key in required_keys:
            assert key in train_package, f"{key} not found in train_package for {self.client_type} client"

    def weight_diff_dict(self, client_state_dict: Dict[str, torch.Tensor], global_state_dict: Dict[str, torch.Tensor]) -> ModelUpdate:
        """
        Compute the weight difference between the current model and the provided state_dict.
        Args:
            state_dict: ModelUpdate containing model parameters
        """
        return {name: param - global_state_dict[name] for name, param in client_state_dict.items()}

    def train(self, train_package: Dict[str, Any]) -> Tuple[int, ModelUpdate, Metrics]:
        """
        Train the model for a number of epochs.

        Args:
            train_package: Data package received from server to train the model (e.g., global model weights, learning rate, etc.)

        Returns:
            num_examples (int): number of examples in the training dataset
            state_dict (ModelUpdate): updated model parameters
            training_metrics (Dict[str, float]): training metrics
        """
        raise NotImplementedError("Train method must be implemented by subclasses")

    def evaluate(self, test_package: Dict[str, Any]) -> Tuple[int, Metrics]:
        """
        Evaluate the model on test data.
        Args:
            test_package: Data package received from server to evaluate the model (e.g., global model weights, learning rate, etc.)
        Returns:
            num_examples (int): number of examples in the test dataset
            evaluation_metrics (Dict[str, float]): evaluation metrics
        """
        raise NotImplementedError("Evaluate method must be implemented by subclasses")

    # An utility function to calculate the L2 distance between client model parameters and global parameters
    def model_dist(self, global_params_tensor: torch.Tensor, client_model=None, gradient_calc=False):
        """Calculate the L2 distance between client model parameters and global parameters"""
        if client_model is None:
            client_model = self.model

        client_params_tensor = torch.cat([param.view(-1) for param in client_model.parameters()]).to(self.device)
        global_params_tensor = global_params_tensor.to(self.device)
        if gradient_calc:
            return torch.linalg.norm(client_params_tensor - global_params_tensor, ord=2).item()
        else:
            return torch.linalg.norm(client_params_tensor - global_params_tensor, ord=2)

    def get_client_info(self):
        """
        Get client information.
        Returns:
            Dictionary containing client information
        """
        return {
            "client_id": self.client_id,
            "client_type": self.client_type,
            "device": str(self.device),
            "dataset_size": len(self.train_dataset)
        }

    def get_client_type(self):
        """
        Get client type.
        Returns:
            String for client type identification
        """
        return self.client_type
