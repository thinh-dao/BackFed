"""AugmentServer: A Server implementing test-time augmentation (TTA) for robust evaluation."""

import torch
import torch.nn as nn
import torchvision.transforms as T
import random
import numpy as np
import cv2

from tqdm import tqdm
from typing import Optional
from backfed.servers.fedavg_server import UnweightedFedAvgServer
from backfed.const import Metrics
from backfed.utils.logging_utils import log, INFO


# ============================================================
# --- Helper functions for image conversion and denoising ---
# ============================================================

def tensor_to_uint8_numpy(img_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert tensor [C,H,W] float in [0,1] -> uint8 HxWxC (RGB) for OpenCV.
    """
    img = img_tensor.detach().cpu().clamp(0.0, 1.0).mul(255).byte().numpy()
    img = np.transpose(img, (1, 2, 0))  # [H, W, C]
    return img


def uint8_numpy_to_tensor(img_np: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convert uint8 HxWxC (RGB) -> tensor [C,H,W] float in [0,1].
    """
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    img = torch.from_numpy(img_np.astype(np.float32) / 255.0)  # [H, W, C]
    img = img.permute(2, 0, 1).contiguous().to(device)         # [C, H, W]
    return img


def denoise_nlm_pytorch(img_tensor: torch.Tensor, h_color: float = 10.0,
                        templateWindowSize: int = 7, searchWindowSize: int = 21) -> torch.Tensor:
    """
    Non-local Means denoising via OpenCV.
    """
    device = img_tensor.device
    img_np = tensor_to_uint8_numpy(img_tensor)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    denoised_bgr = cv2.fastNlMeansDenoisingColored(
        img_bgr, None, h=h_color, hColor=h_color,
        templateWindowSize=templateWindowSize, searchWindowSize=searchWindowSize
    )
    denoised_rgb = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)
    return uint8_numpy_to_tensor(denoised_rgb, device)


def denoise_gaussian_blur(img_tensor: torch.Tensor, kernel_size: int = 3, sigma: float = 0.8) -> torch.Tensor:
    """
    Lightweight Gaussian blur denoiser.
    """
    gb = T.GaussianBlur(kernel_size=kernel_size, sigma=(max(0.1, sigma), max(0.1, sigma)))
    return gb(img_tensor)


def add_gaussian_noise(img_tensor: torch.Tensor, sigma: float = 0.03) -> torch.Tensor:
    """
    Add Gaussian pixel noise.
    """
    noise = torch.randn_like(img_tensor) * sigma
    return torch.clamp(img_tensor + noise, 0.0, 1.0)


# ============================================================
# --- Augmentation Function (Affine + Noise + Denoise) ---
# ============================================================
def augment_fn(x: torch.Tensor, denoise_type="nlm") -> torch.Tensor:
    """
    Perform random geometric and photometric augmentations,
    plus noise and optional denoising (Gaussian/NLM).
    """
    # max_deg = 10
    # max_translate = 0.05
    # scale_range = (0.9, 1.1)

    # aug_transforms = T.Compose([
    #     T.RandomAffine(
    #         degrees=max_deg,
    #         translate=(max_translate, max_translate),
    #         scale=scale_range,
    #         shear=(-5, 5),
    #         interpolation=T.InterpolationMode.BILINEAR,
    #         fill=0,
    #     ),
    #     T.RandomResizedCrop(size=x.shape[-1], scale=(0.9, 1.0)),
    #     T.RandomHorizontalFlip(p=0.5),
    #     T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    # ])

    # # Apply base transforms
    # x_aug = aug_transforms(x)

    # # Add Gaussian noise
    # sigma = random.uniform(0.01, 0.5)
    x_aug = add_gaussian_noise(x, sigma=0.05)
    x_aug = torch.stack([denoise_gaussian_blur(img, kernel_size=5, sigma=1.5) for img in x_aug], dim=0)

    # # # Denoise (Gaussian blur or NLM)
    # if denoise_type == "gaussian":
    #     x_aug = torch.stack([denoise_gaussian_blur(img, kernel_size=3, sigma=0.8) for img in x_aug], dim=0)
    # else:
    #     x_aug = torch.stack([denoise_nlm_pytorch(img, h_color=10.0) for img in x_aug], dim=0)

    return x_aug.clamp(0, 1)


# ============================================================
# --- AugmentServer with TTA-based Evaluation ---
# ============================================================

class AugmentServer(UnweightedFedAvgServer):
    """
    Federated server with Test-Time Augmentation (TTA) evaluation.
    Uses spatial + color augmentations, Gaussian noise, and denoising.
    """
    def __init__(self, server_config, server_type="norm_clipping", tta_k=20, eta=0.5, eval_on_last: bool = True):
        """
        Args:
            server_config: Configuration for the server.
            server_type: Type of server.
            tta_k: Number of augmentations per test sample during TTA.
            eta: Learning rate for the server.
        """
        super(AugmentServer, self).__init__(server_config, server_type, eta=eta)
        self.tta_k = tta_k
        self.eval_on_last = eval_on_last
        log(INFO, f"Initialized Augment server with tta_k={tta_k}, eta={eta}")

    @torch.inference_mode()
    def server_evaluate(self, round_number: Optional[int] = None, test_poisoned: bool = True, model: Optional[torch.nn.Module] = None) -> Metrics:
        """
        Evaluate the global model using test-time augmentation (TTA).
        """

        if round_number is None:
            raise ValueError("AugmentServer requires round_number for augmented evaluation.")

        if self.eval_on_last and (round_number < self.start_round + self.config.num_rounds - 1):
            return super().server_evaluate(round_number=round_number, test_poisoned=test_poisoned, model=model)
        
        if model is None:
            model = self.global_model

        model = model.to(self.device)
        model.eval()

        normalization = self.normalization
        loss_fn = nn.CrossEntropyLoss(reduction="sum")

        total_samples = 0
        total_loss = 0.0
        total_correct = 0

        # ---------- TTA forward helper ----------
        def _tta_forward_batch(inputs: torch.Tensor) -> torch.Tensor:
            bsz = inputs.size(0)
            aug_batches = []

            for _ in tqdm(range(self.tta_k), desc="Augmented evaluation", leave=False):
                # apply augment_fn per image
                aug = augment_fn(inputs)
                # Apply normalization to each augmented batch
                if normalization is not None:
                    aug = normalization(aug)
                aug_batches.append(aug)

            # Concatenate all augmented batches
            aug_inputs = torch.cat(aug_batches, dim=0).to(self.device, non_blocking=True)
            logits = model(aug_inputs)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]

            num_classes = logits.size(-1)
            logits = logits.view(self.tta_k, bsz, num_classes)
            logits_agg = logits.mean(dim=0)
            return logits_agg

        # ---------- Clean evaluation ----------
        for batch in self.test_loader:
            if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                raise ValueError("Unsupported batch format in test_loader for AugmentServer evaluation")

            inputs, labels = batch[0], batch[1]
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            logits_agg = _tta_forward_batch(inputs)
            batch_loss = loss_fn(logits_agg, labels).item()
            preds = torch.argmax(logits_agg, dim=1)
            correct = (preds == labels).sum().item()

            bsz = labels.size(0)
            total_samples += bsz
            total_loss += batch_loss
            total_correct += correct

        test_clean_loss = total_loss / max(1, total_samples)
        test_clean_acc = total_correct / max(1, total_samples)

        metrics: Metrics = {
            "test_clean_samples": total_samples,
            "test_clean_loss": test_clean_loss,
            "test_clean_acc": test_clean_acc,
        }

        # ---------- Optional backdoor evaluation ----------
        if (
            test_poisoned
            and getattr(self, "poison_module", None) is not None
            and (round_number is None or round_number > self.atk_config.poison_start_round - 1)
        ):
            self.poison_module.set_client_id(-1)
            backdoor_total_samples, backdoor_loss, backdoor_accuracy = self.poison_module.poison_test(
                net=self.global_model,
                test_loader=self.test_loader,
                loss_fn=nn.CrossEntropyLoss(),
                normalization=self.normalization,
            )
            metrics.update({
                "test_backdoor_samples": backdoor_total_samples,
                "test_backdoor_loss": backdoor_loss,
                "test_backdoor_acc": backdoor_accuracy,
            })

        return metrics
