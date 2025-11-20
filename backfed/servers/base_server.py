"""
Base server implementation.
"""
import torch
import ray
import wandb
import os
import glob
import time
import copy

from hydra.utils import get_class
from torch.utils.data import DataLoader
from logging import ERROR
from ray.actor import ActorHandle
from rich.progress import track
from hydra.utils import instantiate
from backfed.client_manager import ClientManager
from backfed.datasets import FL_DataLoader, nonIID_Dataset, check_download
from backfed.utils import (
    pool_size_from_resources,
    log, get_console, log_metrics,
    get_model,
    get_normalization,
    init_wandb,
    init_csv_logger,
    CSVLogger,
    test_lstm_reddit, test_classifier,
    save_model_to_wandb_artifact,
    format_time_hms
)
from backfed.context_actor import ContextActor
from backfed.clients import BenignClient, MaliciousClient, LocalDPClient
from backfed.client_app import ClientApp
from backfed.poisons import Poison
from backfed.const import ModelUpdate, Metrics, client_id, num_examples
from logging import INFO, WARNING
from typing import Dict, Any, List, Tuple, Callable, Optional
from collections import deque

class BaseServer:
    """
    Base class for all FL servers.
    """
    ignore_weights = ["num_batches_tracked"] # Ignore non-differentiable parameters during aggregation
    
    def __init__(self, server_config, server_type="base", **kwargs):
        """
        Initialize the server.

        Args:
            server_config: Dictionary containing configuration
        """
        # Basic setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.server_type = server_type
        self.start_round = 1
        self.config = server_config
        
        # Normalization
        if self.config.dataset.upper() not in ["SENTIMENT140", "REDDIT"] and self.config.normalize:
            self.normalization = get_normalization(dataset_name=server_config.dataset)
        else:
            self.normalization = None

        assert self.config.training_mode == "parallel" or self.config.training_mode == "sequential", "Invalid running mode"

        # Prepare dataset
        self._prepare_dataset()

        # Get initial model
        self._init_model()
        self.current_round = self.start_round

        # Initialize the client_manager and get the rounds_selection
        self._init_client_manager(config=server_config, start_round=self.start_round)

        # Initialize the trainer
        self._init_trainer()
        
        # Initialize poison module to share among malicious clients
        self._init_poison_module()

        # Initialize tracking
        if self.config.save_logging in ["wandb", "both"]:
            init_wandb(server_config)

        elif self.config.save_logging in ["csv", "both"]:
            # Import here to avoid circular imports
            from .anomaly_detection import AnomalyDetectionServer
            if isinstance(self, AnomalyDetectionServer):
                self.csv_logger : CSVLogger = init_csv_logger(server_config, detection=True)
            else:
                self.csv_logger : CSVLogger = init_csv_logger(server_config)

        elif self.config.save_logging == None:
            log(WARNING, "The logging is not saved!")

        # Visualization
        if self.config.plot_client_selection:
            self.client_manager.visualize_client_selection(save_path=self.config.output_dir)

        if self.config.plot_data_distribution:
            self.fl_dataloader.visualize_dataset_distribution(malicious_clients=self.client_manager.get_malicious_clients(), 
                                                              save_path=self.config.output_dir
                                                            )
        
        self.trainable_names = set(name for name, _ in self.global_model.named_parameters())

    def _init_model(self):
        """
        Get the initial model.
        """
        if self.config.checkpoint:
            checkpoint = self._load_checkpoint()
            self.global_model = get_model(model_name=self.config.model, num_classes=self.config.num_classes, dataset_name=self.config.dataset)
            self.global_model.load_state_dict(checkpoint['model_state'], strict=True)
            self.start_round = checkpoint['server_round'] + 1

        elif self.config.pretrain_model_path != None:
            self.global_model = get_model(model_name=self.config.model, num_classes=self.config.num_classes, dataset_name=self.config.dataset, pretrain_model_path=self.config.pretrain_model_path)

        else:
            self.global_model = get_model(model_name=self.config.model, num_classes=self.config.num_classes, dataset_name=self.config.dataset)

        self.global_model = self.global_model.to(self.device)

        if self.config.wandb.save_model == True and self.config.wandb.save_model_round == -1:
            self.config.wandb.save_model_round = self.start_round + self.config.num_rounds
            
    def _init_client_manager(self, config, start_round):
        log(INFO, "Initializing client manager...")
        
        # Get benign_client_class from config
        if self.server_type == "localdp":
            benign_client_class = LocalDPClient
        else:
            benign_client_class = BenignClient
        
        log(INFO, f"Loaded benign client class: {benign_client_class.__name__}")

        # Get malicious_client_class from config
        if not self.config.no_attack:
            model_poison_method = config.atk_config.model_poison_config[config.atk_config.model_poison_method]
            malicious_client_class = get_class(model_poison_method._target_)
            log(INFO, f"Loaded malicious client class: {malicious_client_class.__name__}")
        else:
            malicious_client_class = None
            log(INFO, "No attack mode enabled, malicious client class set to None")
        
        self.client_manager = ClientManager(config, benign_client_class=benign_client_class, malicious_client_class=malicious_client_class, start_round=start_round)
        self.rounds_selection = self.client_manager.get_rounds_selection()

    def _init_trainer(self):
        if self.config.training_mode == "parallel":
            model_ref = ray.put(self.global_model)
            client_config_ref = ray.put(self.config.client_config)
            dataset_ref = ray.put(self.trainset)
            dataset_indices_ref = ray.put(self.client_data_indices)
            secret_dataset_indices_ref = ray.put(self.secret_dataset_indices)

            self.trainer : FLTrainer = FLTrainer(server_config=self.config,
                server=self,
                mode=self.config.training_mode,
                clientapp_init_args=dict(
                    client_config=client_config_ref,
                    model=model_ref,
                    dataset=dataset_ref,
                    dataset_partition=dataset_indices_ref,
                    secret_dataset_indices=secret_dataset_indices_ref
                )
            )

        else:
            self.trainer : FLTrainer = FLTrainer(server_config=self.config,
                server=self,
                mode=self.config.training_mode,
                clientapp_init_args=dict(
                    client_config=self.config.client_config,
                    model=copy.deepcopy(self.global_model),
                    dataset=self.trainset,
                    dataset_partition=self.client_data_indices,
                    secret_dataset_indices=self.secret_dataset_indices
                )
            )

    def _init_poison_module(self):
        # Initialize poison module (for poisoning) and ContextActor (for resource synchronization between malicious clients)
        if self.config.no_attack == False:
            self.atk_config = self.config.atk_config
            data_poison_method = self.atk_config.data_poison_method
            self.poison_module : Poison = instantiate(
                config=self.atk_config.data_poison_config[data_poison_method],
                params=self.atk_config,
                _recursive_=False # Avoid recursive instantiation
            )

            if self.config.training_mode == "parallel":
                self.context_actor = ContextActor.remote()
                self.poison_module.set_device(self.device) # Set device for poison module since it is initialized on the server
            else:
                self.context_actor = None

        else:
            self.atk_config = None
            self.poison_module = None
            self.context_actor = None

    def _prepare_dataset(self):
        self.fl_dataloader = FL_DataLoader(config=self.config)
        if self.config.dataset.upper() in ["REDDIT", "FEMNIST", "SENTIMENT140"]:
            check_download(self.config.dataset.upper())
            self.trainset, self.client_data_indices, self.secret_dataset_indices = None, None, None 
            self.testset = nonIID_Dataset(dataset_name=self.config.dataset, config=self.config, client_id=-1)
        else:
            self.trainset, self.client_data_indices, self.secret_dataset_indices, self.testset = self.fl_dataloader.prepare_dataset() 
        
        self.test_loader = DataLoader(self.testset, 
                                    batch_size=self.config.test_batch_size, 
                                    num_workers=self.config.num_workers,
                                    pin_memory=self.config.pin_memory,
                                    shuffle=False
                                )

    def _load_checkpoint(self):
        """
        Three ways to load checkpoint:
        1. From W&B
        2. From a specific round
        3. From local path
        """
        if self.config.checkpoint == "wandb":
            # Fetch the model from W&B
            api = wandb.Api()
            artifact = api.artifact(f"{self.config.wandb.entity}/{self.config.wandb.project}/{self.config.dataset}_{self.config.model.lower()}:latest")
            local_path = artifact.download()
            log(INFO, f"{self.config.model.lower()} checkpoint from W&B is downloaded to: {local_path}")
            resume_model_dict = torch.load(os.path.join(local_path, "model.pth"))

        elif isinstance(self.config.checkpoint, int): # Load from specific round
            # Load from checkpoint
            eta = self.config.aggregator_config[self.config.aggregator]['eta']
            save_dir = os.path.join(os.getcwd(), "checkpoints", f"{self.config.dataset.upper()}_{self.config.aggregator}_{eta}")
            if self.config.partitioner == "uniform":
                model_path = f"{self.config.model.lower()}_round_{self.config.checkpoint}_uniform.pth"
            else:
                # Look for the model with the correct round_number and alpha. If correct alpha is not found, take the model with the highest alpha.
                model_path = os.path.join(save_dir, f"{self.config.model.lower().lower()}_round_{self.config.checkpoint}_dir_{self.config.alpha}.pth")
                if not os.path.exists(model_path):
                    model_path_pattern = os.path.join(save_dir, f"{self.config.model.lower()}_round_{self.config.checkpoint}_dir_*.pth")
                    model_paths = glob.glob(model_path_pattern)
                    if len(model_paths) == 0:
                        raise FileNotFoundError(f"No checkpoint found for {self.config.model.lower()} at round {self.config.checkpoint} with any alpha in {save_dir}")
                    model_path = max(model_paths, key=lambda p: float(p.split('_')[-1].replace('.pth', '')))
                    highest_alpha = float(model_path.split('_')[-1].replace('.pth', ''))
                    log(WARNING, f"No checkpoint found for alpha {self.config.alpha} at round {self.config.checkpoint}. Loading model with highest alpha: {highest_alpha}")

            save_path = os.path.join(save_dir, model_path)
            if not os.path.exists(save_path):
                raise FileNotFoundError(f"No checkpoint found for {self.config.model.lower()} at round {self.config.checkpoint} in {save_dir}")

            resume_model_dict = torch.load(save_path)
            save_paths = glob.glob(os.path.join(save_dir, f"{self.config.model.lower()}_round_{self.config.checkpoint}*.pth"))
            if not save_paths:
                raise FileNotFoundError(f"No checkpoint found for {self.config.model.lower()} at round {self.config.checkpoint} in {save_dir}")
            save_path = save_paths[0]  # Assuming we take the first match if multiple files are found
            resume_model_dict = torch.load(save_path)

        else: # Load from local path
            if not os.path.exists(self.config.checkpoint):
                raise FileNotFoundError(f"Checkpoint not found at {self.config.checkpoint}. Have you downloaded the checkpoints?")
            resume_model_dict = torch.load(self.config.checkpoint)

        # Update current round
        start_round = resume_model_dict['server_round']
        log(INFO, f"Loaded checkpoint from round {start_round} with metrics: {resume_model_dict['metrics']}")

        return resume_model_dict

    def _get_save_dict(self):
        return {
            'metrics': self.best_metrics,
            'model_state': self.best_model_state,
            'server_round': self.current_round,
            'model_name': self.config.model.lower(),
        }

    def _save_checkpoint(self, server_metrics):
        if self.config.partitioner == "dirichlet":
            model_filename = f"{self.config.model.lower()}_round_{self.current_round}_dir_{self.config.alpha}.pth"
        else:
            model_filename = f"{self.config.model.lower()}_round_{self.current_round}_uniform.pth"

        # Create a dictionary with metrics, model state, server_round, and model_name, etc.
        save_dict = self._get_save_dict()
        
        # Save the full checkpoint
        if self.config.save_checkpoint:
            eta = self.config.aggregator_config[self.config.aggregator]['eta']
            save_dir = os.path.join(os.getcwd(), "checkpoints", f"{self.config.dataset.upper()}_{self.config.aggregator}_{eta}")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, model_filename)
        
            # Save the dictionary
            torch.save(save_dict, save_path)

            if self.config.dataset.upper() == "REDDIT":
                log(INFO, f"Checkpoint with {self.best_metrics['test_perplexity']:.2f} perplexity saved to {save_path}.")
            else:
                log(INFO, f"Checkpoint with {self.best_metrics['test_clean_acc'] * 100:.2f}% ACC saved to {save_path}.")

        # Save only the model state
        if self.config.save_model:
            save_dir = os.path.join(self.config.output_dir, "models")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, model_filename)

            torch.save(self.best_model_state, save_path) # include only model state

            if self.config.dataset.upper() == "REDDIT":
                log(INFO, f"Model with {self.best_metrics['test_perplexity']:.2f} perplexity saved to {save_path}.")
            else:
                log(INFO, f"Model with {self.best_metrics['test_clean_acc'] * 100:.2f}% ACC saved to {save_path}.")

        # Save full checkpoint to W&B as artifact
        if self.config.save_logging in ["wandb", "both"] \
            and self.config.wandb.save_model == True \
            and self.current_round == self.config.wandb.save_model_round:
            save_model_to_wandb_artifact(self.config, self.current_round, save_dict, model_filename)

    def aggregate_client_updates(self, client_updates: List[Tuple[client_id, num_examples, ModelUpdate]]) -> bool:
        """
        Aggregates client updates to update global model parameters (self.global_model)

        Args:
            client_updates: List of (client_id, num_examples, model_updates)
        Returns:
            True if the global model parameters are updated, False otherwise
        """
        pass

    def aggregate_client_metrics(self, client_metrics: List[Tuple[int, int, Metrics]]) -> Metrics:
        """
        Aggregates client metrics using a weighted average based on num_examples.

        Args:
            client_metrics: List of (client_id, num_examples, metrics)
        Returns:
            aggregated_metrics: Dict of weighted average metrics
        """
        aggregated_metrics = {}
        metrics_num_examples = {}

        for client_id, num_examples, metric in client_metrics:
            for metric_name, metric_value in metric.items():

                if metric_name not in aggregated_metrics:
                    aggregated_metrics[metric_name] = 0
                    metrics_num_examples[metric_name] = 0

                aggregated_metrics[metric_name] += num_examples * metric_value
                metrics_num_examples[metric_name] += num_examples


        # Aggregate and return custom metrics (weighted average)
        for metric_name in aggregated_metrics.keys():
            aggregated_metrics[metric_name] /= metrics_num_examples[metric_name]

        return aggregated_metrics

    def fit_round(self, clients_mapping: Dict[Any, List[int]]) -> Metrics:
        """Perform one round of FL training.

        Args:
            clients_mapping: Mapping of client types to list of client IDs
        Returns:
            aggregated_metrics: Dict of aggregated metrics from clients training
        """
        train_time_start = time.time()
        client_packages = self.trainer.train(clients_mapping)
        train_time_end = time.time()
        train_time = train_time_end - train_time_start
        log(INFO, f"Clients training time: {train_time:.2f} seconds")

        client_metrics = []
        client_updates = []

        for client_id, package in client_packages.items():
            num_examples, model_updates, metrics = package
            client_metrics.append((client_id, num_examples, metrics))
            client_updates.append((client_id, num_examples, model_updates))

        aggregate_time_start = time.time()

        if self.aggregate_client_updates(client_updates):
            aggregated_metrics = self.aggregate_client_metrics(client_metrics)
        else:
            log(WARNING, "No client updates to aggregate. Global model parameters are not updated.")
            aggregated_metrics = {}

        aggregate_time_end = time.time()
        aggregate_time = aggregate_time_end - aggregate_time_start
        log(INFO, f"Server aggregate time: {aggregate_time:.2f} seconds")

        return aggregated_metrics

    def evaluate_round(self, clients_mapping: Dict[Any, List[int]]) -> Metrics:
        """Perform one round of FL evaluation on the client side.

        Args:
            clients_mapping: Mapping of client types to list of client IDs
        Returns:
            aggregated_metrics: Dict of aggregated metrics from clients evaluation
        """
        client_packages = self.trainer.test(clients_mapping)
        client_metrics = []

        for client_id, package in client_packages.items():
            num_examples, metrics = package
            client_metrics.append((client_id, num_examples, metrics))

        return self.aggregate_client_metrics(client_metrics)

    @torch.inference_mode()
    def server_evaluate(self, round_number: Optional[int] = None, test_poisoned: bool = True, model: Optional[torch.nn.Module] = None) -> Metrics:
        """Perform one round of FL evaluation on the server side."""
        if model is None:
            model = self.global_model

        if self.config.dataset.upper() == "REDDIT":
            clean_loss, perplexity = test_lstm_reddit(model=model,
                                        test_loader=self.test_loader,
                                        device=self.device,
                                        normalization=self.normalization,
                                    )
            metrics = {
                "test_clean_loss": clean_loss,
                "test_perplexity": perplexity
            }
        else:
            clean_total_samples, clean_loss, clean_accuracy = test_classifier(dataset=self.config.dataset, 
                                        model=model,
                                        test_loader=self.test_loader,
                                        device=self.device,
                                        normalization=self.normalization,
                                    )

            metrics = {
                "test_clean_samples": clean_total_samples,
                "test_clean_loss": clean_loss,
                "test_clean_acc": clean_accuracy
            }

        if test_poisoned and self.poison_module is not None and (round_number is None or round_number > self.atk_config.poison_start_round - 1): # Evaluate the backdoor performance starting from the round before the poisoning starts
            self.poison_module.set_client_id(-1) # Set poison module to server
            backdoor_total_samples, backdoor_loss, backdoor_accuracy= self.poison_module.poison_test(net=self.global_model,
                                                            test_loader=self.test_loader,
                                                            loss_fn=torch.nn.CrossEntropyLoss(),
                                                            normalization=self.normalization
                                                        )
            metrics.update({
                "test_backdoor_samples": backdoor_total_samples,
                "test_backdoor_loss": backdoor_loss,
                "test_backdoor_acc": backdoor_accuracy
            })

        return metrics

    def run_one_round(self, round_number: int):
        """Run a full FL round: clients training, clients evaluation, and server evaluation.
        By default, clients that are selected for training are also selected for evaluation.

        Args:
            round_number: The round number to run
        Returns:
            server_metrics: Dict of server evaluation metrics
            client_fit_metrics: Dict of client training metrics
            client_evaluation_metrics: Dict of client evaluation metrics
        """
        clients_mapping = self.rounds_selection[round_number]

        log(INFO, f"ClientManager: Selected clients")
        log(INFO, clients_mapping)

        time_start = time.time()
        client_fit_metrics = self.fit_round(clients_mapping)
        time_end = time.time()
        time_fit = time_end - time_start

        log(INFO, f"Server fit time: {time_fit:.2f} seconds")

        # Update poison module with update_poison_module if:
        # 1. training_mode is parallel, and
        # 2. no_attack is False, and
        # 3. current round is a poison round, and
        # 4. poison_module.sync_poison is True (IBA, A3FL)
        # Note: In sequential mode, poison_module is shared between server and clients, so no update is needed.
        update_poison_module = (self.config.training_mode == "parallel" and \
                                self.config.no_attack == False and \
                                self.poison_module.sync_poison and \
                                round_number in self.client_manager.get_poison_rounds() \
                                )
        if update_poison_module:
            resource_package = ray.get(self.context_actor.wait_for_resource.remote(round_number=round_number))
            self.poison_module.update_shared_resources(resource_package)

        client_eval_time_start = time.time()
        if self.config.federated_evaluation:
            client_evaluation_metrics = self.evaluate_round(clients_mapping)
        else:
            client_evaluation_metrics = None
        client_eval_time_end = time.time()
        client_eval_time = client_eval_time_end - client_eval_time_start
        log(INFO, f"Clients evaluation time: {client_eval_time:.2f} seconds")

        if round_number % self.config.test_every == 0:
            server_eval_time_start = time.time()
            server_evaluation_metrics = self.server_evaluate(round_number)
            server_eval_time_end = time.time()
            server_eval_time = server_eval_time_end - server_eval_time_start
            log(INFO, f"Server evaluation time: {server_eval_time:.2f} seconds")
        else:
            server_evaluation_metrics = {}

        return server_evaluation_metrics, client_fit_metrics, client_evaluation_metrics

    def run_experiment(self):
        """Run the full FL experiment loop."""
        experiment_start_time = time.time()

        if self.config.progress_bar:
            train_progress_bar = track(
                range(self.start_round, self.start_round + self.config.num_rounds),
                "[bold green]Training...",
                console=get_console(),
            )
        else:
            train_progress_bar = range(self.start_round, self.start_round + self.config.num_rounds)

        self.best_metrics = {}
        self.best_model_state = {name: param.detach().clone() for name, param in self.global_model.state_dict().items()}

        poison_rounds = self.client_manager.get_poison_rounds()
        for self.current_round in train_progress_bar:
            separator = "=" * 30
            round_start_time = time.time()

            if self.current_round in poison_rounds:
                log(INFO, f"{separator} POISONING ROUND: {self.current_round} {separator}")
            else:
                log(INFO, f"{separator} TRAINING ROUND: {self.current_round} {separator}")

            server_metrics, client_fit_metrics, client_evaluation_metrics = self.run_one_round(round_number=self.current_round)

            # Initialize or update best metrics based on dataset type
            if len(self.best_metrics) == 0:
                self.best_metrics = server_metrics
                self.best_model_state = {name: param.detach().clone() for name, param in self.global_model.state_dict().items()}
            elif self.config.dataset.upper() == "REDDIT":
                # For language modeling, use perplexity (derived from loss)
                if server_metrics.get("test_perplexity", float('inf')) < self.best_metrics.get("test_perplexity", float('inf')):
                    self.best_metrics = server_metrics
                    self.best_model_state = {name: param.detach().clone() for name, param in self.global_model.state_dict().items()}
            else:
                # For classification tasks, use accuracy
                if server_metrics.get("test_clean_acc", 0) > self.best_metrics.get("test_clean_acc", 0):
                    self.best_metrics = server_metrics
                    self.best_model_state = {name: param.detach().clone() for name, param in self.global_model.state_dict().items()}

            round_end_time = time.time()
            round_time = round_end_time - round_start_time
            log(INFO, f"Round {self.current_round} completed in {round_time:.2f} seconds")

            # Centralized metrics
            log(INFO, "═══ Centralized Metrics ═══")
            log_metrics(server_metrics)
            log(INFO, "═══ Client Fit Metrics ═══")
            log_metrics(client_fit_metrics)
            if client_evaluation_metrics is not None:
                log(INFO, "═══ Client Evaluation Metrics ═══")
                log_metrics(client_evaluation_metrics)

            if self.config.save_logging in ["wandb", "both"]:
                wandb.log({**server_metrics, "round": self.current_round})
                wandb.log({**client_fit_metrics}, step=self.current_round)

            elif self.config.save_logging in ["csv", "both"]:
                self.csv_logger.log({**server_metrics}, step=self.current_round)
                self.csv_logger.log({**client_fit_metrics}, step=self.current_round)

            if self.current_round in self.config.save_checkpoint_rounds:
                self._save_checkpoint(server_metrics=server_metrics)

        experiment_end_time = time.time()
        experiment_time = experiment_end_time - experiment_start_time
        log(INFO, f"{separator} TRAINING COMPLETED {separator}")
        log(INFO, f"Total experiment time: {format_time_hms(experiment_time)}")

        # Clear up logging resources
        if self.config.save_logging in ["csv", "both"]:
            self.csv_logger.finish()
        elif self.config.save_logging in ["wandb", "both"]:
            wandb.finish()
        
        # Finalize poison module
        if self.config.no_attack == False:
            self.poison_module.poison_finish()
        
        # Shutdown Ray if in parallel mode
        if self.config.training_mode == "parallel":
            ray.shutdown()

    def train_package(self, client_type: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Send the init_args and train_package to ClientApp based on the client type.
        ClientApp use these arguments to initialize client and train the client.
        """
        if issubclass(client_type, BenignClient):
            init_args = {}
            train_package = {
                "global_state_dict": self.global_model.state_dict(),
                "server_round": self.current_round,
                "normalization": self.normalization
            }
        elif issubclass(client_type, MaliciousClient):
            assert self.poison_module is not None, "Poison module is not initialized"
            if self.config.training_mode == "parallel":
                assert self.context_actor is not None, "Context actor is not initialized"

            model_poison_method = self.atk_config.model_poison_method
            model_poison_kwargs = {k:v for k,v in self.atk_config.model_poison_config[model_poison_method].items() if k != "_target_"}

            init_args = {
                "poison_module": self.poison_module,
                "atk_config": self.atk_config,
                "context_actor": self.context_actor,
                **model_poison_kwargs
            }
            train_package = {
                "global_state_dict": self.global_model.state_dict(),
                "server_round": self.current_round,
                "normalization": self.normalization,
                "selected_malicious_clients": self.rounds_selection[self.current_round][client_type],
            }
        else:
            raise ValueError(f"Unsupported client type: {client_type}")

        return init_args, train_package

    def test_package(self, client_type: Any) -> Dict[str, Any]:
        """
        Send the test_package to ClientApp based on the client type.
        ClientApp use this package to test the client.
        """
        if issubclass(client_type, BenignClient):
            test_package = {
                "global_state_dict": self.global_model.state_dict(),
                "server_round": self.current_round,
                "normalization": self.normalization
            }
        elif issubclass(client_type, MaliciousClient):
            test_package = {
                "global_state_dict": self.global_model.state_dict(),
                "server_round": self.current_round,
                "normalization": self.normalization
            }
        else:
            raise ValueError(f"Unsupported client type: {client_type}")

        return test_package

    def get_clients_info(self, round_number: int) -> Dict[str, Any]:
        """
        Get the clients info for the given round number.
        """
        assert round_number in self.rounds_selection, "Round number is not in the rounds selection"
        clients_info = {
            "benign_clients": [],
            "malicious_clients": []
        }

        for cls in self.rounds_selection[round_number].keys():
            if issubclass(cls, BenignClient):
                clients_info["benign_clients"] = self.rounds_selection[round_number][cls]
            elif issubclass(cls, MaliciousClient):
                clients_info["malicious_clients"] = self.rounds_selection[round_number][cls]

        return clients_info

    #========= Some utility functions =========

    @torch.no_grad()
    def compute_client_distance(self, client_update: ModelUpdate) -> float:
        """
        Compute L2 distance between client model and global model for differentiable parameters only.
        """
        update_tensor = self.parameters_dict_to_vector(client_update)
        weight_diff_norm = torch.linalg.norm(update_tensor, ord=2)
        return weight_diff_norm.item()
    
    def parameters_dict_to_vector(self, state_dict: ModelUpdate) -> torch.Tensor:
        """Convert parameters dictionary to flat vector, excluding BatchNorm buffers."""
        vec = []
        for name, param in state_dict.items():
            if name in self.trainable_names:
                vec.append(param.detach().clone().view(-1))
        return torch.cat(vec)
    
    @property
    def global_parameters_vector(self) -> torch.Tensor:
        """Get global model parameters as a flat vector, excluding BatchNorm buffers and ignored weights."""
        vec = []
        for _, param in self.global_model.named_parameters():  # Only trainable params
            vec.append(param.detach().clone().view(-1))
        return torch.cat(vec)
    
    #========= End utility functions =========

class FLTrainer:
    def __init__(self,
        server_config: Dict,
        server: BaseServer,
        clientapp_init_args: dict,
        mode: str,
    ):
        """
        FLTrainer coordinates training and evaluation between the server and clients.

        Args:
            server_config: Configuration dictionary for the server
            server: BaseServer instance
            clientapp_init_args: Dictionary containing initialization arguments for ClientApp
            mode: Training mode (sequential or parallel)
        """
        self.config = server_config
        self.server = server
        self.mode = mode
        self.check_for_nan = True # Whether to check for NaN/Inf values in client updates

        if self.mode == "sequential":
            self.worker : ClientApp = ClientApp(**clientapp_init_args)
        elif self.mode == "parallel":
            ray_client = ray.remote(ClientApp).options(
                num_cpus=self.config.num_cpus,
                num_gpus=self.config.num_gpus
            )

            client_ressource = dict(num_cpus=self.config.num_cpus, num_gpus=self.config.num_gpus)
            self.num_workers = pool_size_from_resources(client_ressource)
            self.workers : List[ActorHandle] = [
                ray_client.remote(**clientapp_init_args) for _ in range(self.num_workers)
            ]
        else:
            raise ValueError(f"Unrecongnized running mode.")

        if self.mode == "sequential":
            self.train = self._serial_train
            self.test = self._serial_test
            self.exec = self._serial_exec
        else:
            self.train = self._parallel_train
            self.test = self._parallel_test
            self.exec = self._parallel_exec

    def _serial_train(self, clients_mapping: Dict[Any, List[int]]) -> Dict[int, Tuple[int, ModelUpdate, Metrics]]:
        """Trains clients sequentially.

        Args:
            clients_mapping: Mapping of client types to list of client IDs

        Returns:
            client_packages (dict): client_id -> (num_examples, model_updates, metrics)
        """
        client_packages = {}
        num_failures = 0
        train_times = []

        for client_cls in clients_mapping.keys():
            init_args, train_package = self.server.train_package(client_cls)
            for client_id in clients_mapping[client_cls]:
                train_time, client_package = self.worker.train(client_cls=client_cls,
                    client_id=client_id,
                    init_args=init_args,
                    train_package=train_package,
                    timeout=self.config.client_timeout
                )
                train_times.append((train_time, client_id))

                # Check if the client failed
                if isinstance(client_package, dict) and client_package.get("status") == "failure":
                    num_failures += 1
                    error_msg = client_package['error']
                    error_tb = client_package.get('traceback', 'No traceback available')
                    log(ERROR, f"Client [{client_id}] failed during training:\n{error_msg}\n{error_tb}")
                    continue
                
                if self.check_for_nan:
                    # Check for NaN/Inf values in model updates
                    _, model_updates, _ = client_package
                    has_nan = False
                    nan_param_name = None
                    for name, param in model_updates.items():
                        if torch.isnan(param).any() or torch.isinf(param).any():
                            has_nan = True
                            nan_param_name = name
                            break

                    if has_nan:
                        log(WARNING, f"Client [{client_id}] has NaN/Inf values in parameter '{nan_param_name}'. Skipping this client from aggregation.")
                        continue

                # If not failed and no NaN, add the client package to the client_packages
                client_packages[client_id] = client_package

        if num_failures > 0:
            log(WARNING, f"Number of failures: {num_failures}")

        if self.config.selection_threshold is not None:
            num_select = int(self.config.selection_threshold * self.config.num_clients_per_round)
            train_times.sort(key=lambda x: x[0]) # Sort by training time
            selected_client_ids = set([client_id for _, client_id in train_times[:num_select]])
            client_packages = {client_id: package for client_id, package in client_packages.items() if client_id in selected_client_ids}
            avg_time = sum([t for t, _ in train_times if _ in selected_client_ids]) / len(client_packages)
            log(INFO, f"Selected {len(client_packages)} clients based on selection_threshold {self.config.selection_threshold}. Average training time: {avg_time:.2f} seconds.")
            log(INFO, f"Selected client IDs: {selected_client_ids}")

        return client_packages

    def _parallel_train(self, clients_mapping: Dict[Any, List[int]]) -> Dict[int, Tuple[int, ModelUpdate, Metrics]]:
        """Trains clients in parallel.

        Args:
            clients_mapping: Mapping of client types to list of client IDs

        Returns:
            client_packages (dict): client_id -> (num_examples, model_updates, metrics)
        """
        # Prepare all clients and their corresponding packages
        all_clients = []
        for client_cls, clients in clients_mapping.items():
            init_args, train_package = self.server.train_package(client_cls)

            init_args_ref = ray.put(init_args)
            train_package_ref = ray.put(train_package)
            for client_id in clients:
                all_clients.append((client_cls, client_id, init_args_ref, train_package_ref))

        idle_workers = deque(range(self.num_workers))
        futures = []
        job_map = {}
        train_times = []
        client_packages = {}
        num_failures = 0
        i = 0
        while i < len(all_clients) or len(futures) > 0:
            # Launch new tasks while we have idle workers and clients to process
            while i < len(all_clients) and len(idle_workers) > 0:
                worker_id = idle_workers.popleft()
                client_cls, client_id, init_args, train_package = all_clients[i]
                future = self.workers[worker_id].train.remote(
                    client_cls=client_cls,
                    client_id=client_id,
                    init_args=init_args,
                    train_package=train_package,
                    timeout=self.config.client_timeout
                )
                job_map[future] = (client_id, worker_id)
                futures.append(future)
                i += 1

            # Process completed tasks
            if len(futures) > 0:
                all_finished, futures = ray.wait(futures)
                for finished in all_finished:
                    client_id, worker_id = job_map[finished]
                    train_time, client_package = ray.get(finished)
                    idle_workers.append(worker_id)
                    train_times.append((train_time, client_id))

                    # Check if the client failed
                    if isinstance(client_package, dict) and client_package.get("status") == "failure":
                        num_failures += 1
                        error_msg = client_package['error']
                        error_tb = client_package.get('traceback', 'No traceback available')
                        log(ERROR, f"Client [{client_id}] failed during training:\n{error_msg}\n{error_tb}")
                        continue
                    
                    if self.check_for_nan:
                        # Check for NaN/Inf values in model updates
                        num_examples, model_updates, metrics = client_package
                        has_nan = False
                        nan_param_name = None
                        for name, param in model_updates.items():
                            if torch.isnan(param).any() or torch.isinf(param).any():
                                has_nan = True
                                nan_param_name = name
                                break

                        if has_nan:
                            log(WARNING, f"Client [{client_id}] has NaN/Inf values in parameter '{nan_param_name}'. Skipping this client from aggregation.")
                            continue

                    # If not failed and no NaN, add the client package to the client_packages
                    client_packages[client_id] = client_package

        if num_failures > 0:
            log(WARNING, f"Number of failures: {num_failures}")


        if self.config.selection_threshold is not None:
            num_select = int(self.config.selection_threshold * self.config.num_clients_per_round)
            train_times.sort(key=lambda x: x[0]) # Sort by training time
            selected_client_ids = set([client_id for _, client_id in train_times[:num_select]])
            client_packages = {client_id: package for client_id, package in client_packages.items() if client_id in selected_client_ids}
            avg_time = sum([t for t, _ in train_times if _ in selected_client_ids]) / len(client_packages)
            log(INFO, f"Selected {len(client_packages)} clients based on selection_threshold {self.config.selection_threshold}. Average training time: {avg_time:.2f} seconds.")
            log(INFO, f"Selected client IDs: {selected_client_ids}")

        return client_packages

    def _serial_test(self, clients_mapping: Dict[Any, List[int]]) -> Dict[int, Tuple[int, Metrics]]:
        """Evaluates clients sequentially.

        Args:
            clients_mapping: Mapping of client types to list of client IDs

        Returns:
            client_packages (dict): client_id -> (num_examples, eval_metrics)
        """
        client_packages = {}
        num_failures = 0
        for client_cls in clients_mapping.keys():
            test_package = self.server.test_package(client_cls)
            for client_id in clients_mapping[client_cls]:
                eval_time, client_package = self.worker.evaluate(test_package=test_package, timeout=self.config.client_timeout)

                # Check if the client failed
                if isinstance(client_package, dict) and client_package.get("status") == "failure":
                    num_failures += 1
                    error_msg = client_package['error']
                    error_tb = client_package.get('traceback', 'No traceback available')
                    log(WARNING, f"Client [{client_id}] failed during evaluation:\n{error_msg}\n{error_tb}")
                    continue

                # If not failed, add the client package to the client_packages
                client_packages[client_id] = client_package

        if num_failures > 0:
            log(WARNING, f"Number of failures: {num_failures}")

        return client_packages

    def _parallel_test(self, clients_mapping: Dict[Any, List[int]]) -> Dict[int, Tuple[int, Metrics]]:
        """Evaluates clients in parallel.

        Args:
            clients_mapping: Mapping of client types to list of client IDs

        Returns:
            client_packages (dict): client_id -> (num_examples, eval_metrics)
        """
        # Prepare all clients and their corresponding packages
        all_clients = []
        for client_cls, clients in clients_mapping.items():
            test_package = self.server.test_package(client_cls)
            test_package_ref = ray.put(test_package)
            for client_id in clients:
                all_clients.append((client_id, test_package_ref))

        idle_workers = deque(range(self.num_workers))
        futures = []
        job_map = {}
        client_packages = {}
        num_failures = 0
        i = 0
        while i < len(all_clients) or len(futures) > 0:
            # Launch new tasks while we have idle workers and clients to process
            while i < len(all_clients) and len(idle_workers) > 0:
                worker_id = idle_workers.popleft()
                client_id, test_package = all_clients[i]
                future = self.workers[worker_id].evaluate.remote(test_package=test_package, timeout=self.config.client_timeout)
                job_map[future] = (client_id, worker_id)
                futures.append(future)
                i += 1

            # Process completed tasks
            if len(futures) > 0:
                all_finished, futures = ray.wait(futures)
                for finished in all_finished:
                    client_id, worker_id = job_map[finished]
                    eval_time, client_package = ray.get(finished)
                    idle_workers.append(worker_id)

                    # Check if the client failed
                    if isinstance(client_package, dict) and client_package.get("status") == "failure":
                        num_failures += 1
                        error_msg = client_package['error']
                        error_tb = client_package.get('traceback', 'No traceback available')
                        log(ERROR, f"Client [{client_id}] failed during evaluation:\n{error_msg}\n{error_tb}")
                        continue

                    # If not failed, add the client package to the client_packages
                    client_packages[client_id] = client_package

        if num_failures > 0:
            log(WARNING, f"Number of failures: {num_failures}")

        return client_packages

    def _serial_exec(
        self,
        clients_mapping: Dict[Any, List[int]],
        func_name: str,
        package_func: Optional[Callable[[Any], Tuple[Dict[str, Any], Dict[str, Any]]]] = None,
    ) -> Dict[int, Any]:
        """Executes arbitrary function on clients sequentially.

        Args:
            func_name: Name of function to execute on clients
            clients_mapping: Mapping of client types to list of client IDs
            package_func: Function that returns initialization args and execution package

        Returns:
            client_packages: OrderedDict mapping client IDs to function execution results
        """
        client_packages = {}
        num_failures = 0
        for client_cls in clients_mapping.keys():
            init_args, exec_package = package_func(client_cls)
            for client_id in clients_mapping[client_cls]:
                package = getattr(self.worker, func_name)(
                    client_cls=client_cls,
                    client_id=client_id,
                    init_args=init_args,
                    exec_package=exec_package
                )

                # Check if the client failed
                if isinstance(package, dict) and package.get("status") == "failure":
                    num_failures += 1
                    error_msg = package['error']
                    error_tb = package.get('traceback', 'No traceback available')
                    log(ERROR, f"Client [{client_id}] failed during execution:\n{error_msg}\n{error_tb}")
                    continue

                # If not failed, add the client package to the client_packages
                client_packages[client_id] = package

        if num_failures > 0:
            log(WARNING, f"Number of failures: {num_failures}")

        return client_packages

    def _parallel_exec(
        self,
        clients_mapping: Dict[Any, List[int]],
        func_name: str,
        package_func: Optional[Callable[[Any], Tuple[Dict[str, Any], Dict[str, Any]]]] = None,
    ) -> Dict[int, Any]:
        """Executes arbitrary function on clients in parallel.

        Args:
            func_name: Name of function to execute on clients
            clients_mapping: Mapping of client types to list of client IDs
            package_func: Function that returns initialization args and execution package

        Returns:
            client_packages: OrderedDict mapping client IDs to function execution results
        """
        # Prepare all clients and their corresponding packages
        all_clients = []
        for client_cls, clients in clients_mapping.items():
            init_args, exec_package = package_func(client_cls)
            init_args_ref = ray.put(init_args)
            exec_package_ref = ray.put(exec_package)
            for client_id in clients:
                all_clients.append((client_cls, client_id, init_args_ref, exec_package_ref))

        idle_workers = deque(range(self.num_workers))
        futures = []
        job_map = {}
        client_packages = {}
        num_failures = 0
        i = 0
        while i < len(all_clients) or len(futures) > 0:
            # Launch new tasks while we have idle workers and clients to process
            while i < len(all_clients) and len(idle_workers) > 0:
                worker_id = idle_workers.popleft()
                client_cls, client_id, init_args, exec_package = all_clients[i]
                future = getattr(self.workers[worker_id], func_name).remote(
                    client_cls=client_cls,
                    client_id=client_id,
                    init_args=init_args,
                    exec_package=exec_package
                )
                job_map[future] = (client_id, worker_id)
                futures.append(future)
                i += 1

            # Process completed tasks
            if len(futures) > 0:
                all_finished, futures = ray.wait(futures)
                for finished in all_finished:
                    client_id, worker_id = job_map[finished]
                    package = ray.get(finished)
                    idle_workers.append(worker_id)

                    # Check if the client failed
                    if isinstance(package, dict) and package.get("status") == "failure":
                        num_failures += 1
                        error_msg = package['error']
                        error_tb = package.get('traceback', 'No traceback available')
                        log(ERROR, f"Client [{client_id}] failed during execution:\n{error_msg}\n{error_tb}")
                        continue

                    # If not failed, add the client package to the client_packages
                    client_packages[client_id] = package

        if num_failures > 0:
            log(WARNING, f"Number of failures: {num_failures}")

        return client_packages
