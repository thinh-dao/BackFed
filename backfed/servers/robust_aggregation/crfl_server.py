import torch

from logging import INFO
from typing import Optional
from tqdm import tqdm
from backfed.const import Metrics
from backfed.servers.robust_aggregation.weakdp_server import WeakDPServer
from backfed.utils.logging_utils import log

class CRFLServer(WeakDPServer):
    """Certified Robust Federated Learning aggregation (post-update clipping + DP noise + smoothing eval)."""
    def __init__(self, server_config, server_type="weakdp", strategy="crfl",
                 std_dev=0.025, clipping_norm=5.0, test_sigma: float = 0.01, N_m: int = 1000, eta=0.1):

        """
        Args:
            server_config: Configuration for the server.
            server_type: Type of server.
            strategy: Strategy for the server.
            std_dev: Standard deviation for the Gaussian noise.
            clipping_norm: Clipping norm for the Gaussian noise.
            test_sigma: Standard deviation for smoothing evaluation.
            N_m: Number of models for smoothing evaluation.
            eta: Learning rate for the server.
        """
        if std_dev < 0:
            raise ValueError("The std_dev should be a non-negative value.")
        if clipping_norm <= 0:
            raise ValueError("The clipping norm should be a positive value.")
        
        super().__init__(server_config, server_type, strategy=strategy, std_dev=std_dev, 
                        clipping_norm=clipping_norm, eta=eta)

        if self.config.dataset.upper() in {"REDDIT", "SENTIMENT140"}:
            raise ValueError("CRFLServer does not support smoothing evaluation for Reddit or Sentiment140 datasets.")
        
        self.test_sigma = test_sigma
        self.N_m = N_m
        self.smooth_on_last = True  # Only smooth evaluation on the last round
        log(INFO, f"Initialized CRFL server with std_dev={std_dev}, clipping_norm={clipping_norm}, test_sigma={test_sigma}, N_m={N_m}")

    def _add_smoothing_noise(self, model: torch.nn.Module, sigma: float) -> None:
        """Add Gaussian noise to model parameters for smoothed evaluation."""
        for name, param in model.named_parameters():
            if any(pattern in name for pattern in self.ignore_weights) or name not in self.trainable_names:
                continue
            noise = torch.normal(0, sigma, size=param.shape, device=param.device, dtype=param.dtype)
            param.data.add_(noise)

    def _resolve_test_sample_count(self) -> int:
        dataset = getattr(self.test_loader, "dataset", None)
        if dataset is not None:
            try:
                return len(dataset)
            except TypeError:
                pass
        total_samples = 0
        for batch in self.test_loader:
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                labels = batch[1]
            else:
                raise ValueError("Unsupported batch format for smoothed evaluation")
            total_samples += len(labels)
        return total_samples
    
    def _smoothed_classifier_metrics(self, model: torch.nn.Module, num_models: int, sigma: float) -> Metrics:
        """
        Perform randomized smoothing evaluation by running multiple forward passes with Gaussian noise.
        
        Args:
            model: The model to evaluate.
            num_models: Number of noisy model instances to average predictions over.
            sigma: Standard deviation of Gaussian noise added to model parameters.
            
        Returns:
            Metrics dictionary with test_clean_samples, test_clean_loss, and test_clean_acc.
        """
        total_samples = self._resolve_test_sample_count()
        vote_counts = torch.zeros(total_samples, self.config.num_classes, dtype=torch.int32)
        labels_buffer = torch.empty(total_samples, dtype=torch.long)

        base_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        model = model.to(self.device)
        normalization = self.normalization

        for m in tqdm(range(num_models), desc="Smoothed evaluation", leave=False):
            model.load_state_dict(base_state, strict=True)
            self._add_smoothing_noise(model, sigma)
            model.eval()

            offset = 0
            for batch in self.test_loader:
                if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                    raise ValueError("Unsupported batch format for smoothed evaluation")
                inputs, labels = batch[0], batch[1]

                if normalization is not None:
                    inputs = normalization(inputs)

                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)

                batch_size = labels.size(0)
                if m == 0:
                    labels_buffer[offset:offset + batch_size] = labels.cpu()

                batch_indices = torch.arange(offset, offset + batch_size, dtype=torch.long)
                vote_counts[batch_indices, preds.cpu()] += 1
                offset += batch_size

            if offset != total_samples:
                raise ValueError("Mismatch between accumulated samples and dataset size during smoothed evaluation")

        # Restore model to original state (base_state already contains the original parameters)
        model.load_state_dict(base_state, strict=True)
        probs = vote_counts.float() / num_models
        eps = 1e-12
        sample_indices = torch.arange(total_samples, dtype=torch.long)
        target_probs = probs[sample_indices, labels_buffer].clamp_min(eps)
        clean_loss = (-torch.log(target_probs)).mean().item()
        clean_accuracy = (torch.argmax(vote_counts, dim=1) == labels_buffer).float().mean().item()

        return {
            "test_clean_samples": total_samples,
            "test_clean_loss": clean_loss,
            "test_clean_acc": clean_accuracy,
        }

    @torch.inference_mode()
    def server_evaluate(self, round_number: Optional[int] = None, test_poisoned: bool = True,
                        model: Optional[torch.nn.Module] = None) -> Metrics:

        if round_number is None:
            raise ValueError("CRFLServer requires round_number for smoothing evaluation.")
        
        if self.smooth_on_last and (round_number < self.start_round + self.config.num_rounds - 1):
            return super().server_evaluate(round_number=round_number, test_poisoned=test_poisoned, model=model)

        if model is None:
            model = self.global_model

        metrics = self._smoothed_classifier_metrics(model=model, num_models=self.N_m, sigma=self.test_sigma)

        if test_poisoned and self.poison_module is not None and (round_number is None or round_number > self.atk_config.poison_start_round - 1):
            self.poison_module.set_client_id(-1)
            backdoor_total_samples, backdoor_loss, backdoor_accuracy = self.poison_module.poison_test(
                net=self.global_model,
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

    def __repr__(self) -> str:
        return f"CRFL(strategy={self.strategy}, std_dev={self.std_dev}, clipping_norm={self.clipping_norm}, test_sigma={self.test_sigma}, N_m={self.N_m})"
