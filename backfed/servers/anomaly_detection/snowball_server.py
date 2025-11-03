"""
Implementation of Snowball anomaly detection defense for federated learning.
Reference: Snowball paper and implementation
"""

import torch
import numpy as np
import math
import time
import torch.nn as nn
import torch.nn.functional as F

from .anomaly_detection_server import AnomalyDetectionServer
from typing import List, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from torch.nn.init import xavier_normal_, kaiming_normal_
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from logging import INFO, WARNING
from backfed.utils import log
from backfed.const import ModelUpdate, client_id, num_examples


# ============= Helper Functions =============
def cluster(init_ids, data):
    """Perform k-means clustering with specified initial centroids."""
    clusterer = KMeans(n_clusters=len(init_ids), init=[data[i] for i in init_ids], n_init=1)
    cluster_labels = clusterer.fit_predict(data)
    return cluster_labels

kl_loss = lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
recon_loss = torch.nn.MSELoss(reduction='sum')

def _flatten_model(model_update, layer_list=['conv1', 'fc2'], ignore=None):
    """Flatten specified layers from model update into a single vector."""
    k_list = []
    for k in model_update.keys():
        if ignore is not None and ignore in k:
            continue
        for target_k in layer_list:
            # Match only the first part of the layer name (before first dot)
            # e.g., 'conv1.weight' matches 'conv1', but 'layer1.0.conv1.weight' does not
            first_part = k.split('.')[0]
            if first_part == target_k:
                k_list.append(k)
                break
    # Detach to avoid gradient tracking issues
    return torch.concat([model_update[k].flatten() for k in k_list]).detach()

# ============= Dataset and Model Components =============
class MyDST(Dataset):
    """Simple dataset wrapper for VAE training."""
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def _init_weights(model, init_type):
    """Initialize model weights with specified initialization scheme."""
    if init_type not in ['none', 'xavier', 'kaiming']:
        raise ValueError('init must in "none", "xavier" or "kaiming"')

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'xavier':
                xavier_normal_(m.weight.data, gain=1.0)
            elif init_type == 'kaiming':
                kaiming_normal_(m.weight.data, nonlinearity='relu')

    if init_type != 'none':
        model.apply(init_func)
        
def build_dif_set(data):
    """Build difference set from all pairs of data points."""
    dif_set = []
    for i in range(len(data)):
        for j in range(len(data)):
            if i != j:
                # Detach to avoid gradient tracking issues
                dif_set.append((data[i] - data[j]).detach())
    return dif_set

def obtain_dif(base, target):
    """Obtain differences between target and all base points."""
    dif_set = []
    for item in base:
        if torch.sum(item - target) != 0.0:
            # Detach to avoid gradient tracking issues
            dif_set.append((item - target).detach())
            dif_set.append((target - item).detach())
    return dif_set

# ============= VAE Model =============
class VAE(nn.Module):
    """Variational Autoencoder for anomaly detection."""
    def __init__(self, input_dim=784, latent_dim=32, hidden_dim=64):
        super(VAE, self).__init__()
        self.fc_e1 = nn.Linear(input_dim, hidden_dim)
        self.fc_e2 = nn.Linear(hidden_dim, hidden_dim)

        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.fc_d1 = nn.Linear(latent_dim, hidden_dim)
        self.fc_d2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_d3 = nn.Linear(hidden_dim, input_dim)

        self.input_dim = input_dim

    def encoder(self, x_in):
        x = F.relu(self.fc_e1(x_in.view(-1, self.input_dim)))
        x = F.relu(self.fc_e2(x))
        mean = self.fc_mean(x)
        logvar = F.softplus(self.fc_logvar(x))
        return mean, logvar

    def decoder(self, z):
        z = F.relu(self.fc_d1(z))
        z = F.relu(self.fc_d2(z))
        x_out = torch.sigmoid(self.fc_d3(z))
        return x_out.view(-1, self.input_dim)

    def sample_normal(self, mean, logvar):
        sd = torch.exp(logvar * 0.5)
        e = Variable(torch.randn_like(sd))
        z = e.mul(sd).add_(mean)
        return z

    def forward(self, x_in):
        z_mean, z_logvar = self.encoder(x_in)
        z = self.sample_normal(z_mean, z_logvar)
        x_out = self.decoder(z)
        return x_out, z_mean, z_logvar

    def recon_prob(self, x_in, L=10):
        """Calculate reconstruction probability for anomaly detection."""
        with torch.no_grad():
            # Ensure input is on the same device as the model
            device = next(self.parameters()).device
            x_in = x_in.to(device)
            
            x_in = torch.unsqueeze(x_in, dim=0)
            # Clamp input to prevent extreme values
            x_in = torch.clamp(x_in, -10, 10)
            mean, log_var = self.encoder(x_in)

            samples_z = []
            for i in range(L):
                z = self.sample_normal(mean, log_var)
                samples_z.append(z)
            reconstruction_prob = 0.
            for z in samples_z:
                x_logit = self.decoder(z)
                reconstruction_prob += recon_loss(x_logit, x_in).item()
            return reconstruction_prob / L

    def test(self, input_data):
        """Test VAE on input data."""
        running_loss = []
        for single_x in input_data:
            single_x = torch.tensor(single_x).float()

            x_in = Variable(single_x)
            x_out, z_mu, z_logvar = self.forward(x_in)
            x_out = x_out.view(-1)
            x_in = x_in.view(-1)
            bce_loss = F.mse_loss(x_out, x_in, reduction='sum')
            kld_loss = -0.5 * torch.sum(1 + z_logvar - (z_mu ** 2) - torch.exp(z_logvar))
            loss = (bce_loss + kld_loss)

            running_loss.append(loss.item())
        return running_loss


def train_vae(vae, data, num_epoch, device, latent, hidden):
    """Train VAE model on difference data."""    
    data = torch.stack(data, dim=0)
    # Detach data to avoid gradient tracking issues from client training
    data = data.detach()
    data = torch.sigmoid(data)
    
    if vae is None:
        vae = VAE(input_dim=len(data[0]), latent_dim=latent, hidden_dim=hidden).to(device)
        _init_weights(vae, 'kaiming')
        log(INFO, f"Initialized new VAE with input_dim={len(data[0])}, latent_dim={latent}, hidden_dim={hidden}")
    
    vae = vae.to(device)
    vae.train()
    train_loader = DataLoader(MyDST(data), batch_size=8, shuffle=True)
    optimizer = torch.optim.Adam(vae.parameters())
    
    # Training loop with minimal logging
    for epoch in range(num_epoch):
        epoch_loss = 0.0
        epoch_batches = 0
        for _, x in enumerate(train_loader):
            
            optimizer.zero_grad()
            x = x.to(device)
            recon_x, mu, logvar = vae(x)
            recon = recon_loss(recon_x, x)
            kl = kl_loss(mu, logvar)
            kl = torch.mean(kl)

            loss = recon + kl
                
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_batches += 1
        
        # Check for NaN and stop if detected
        if torch.isnan(loss):
            log(WARNING, f"VAE training stopped: NaN loss at epoch {epoch}")
            break
        
        # Print loss at intervals
        if epoch > 50 and (epoch % (num_epoch // 6) == 0 or epoch == num_epoch - 1):
            avg_epoch_loss = epoch_loss / epoch_batches
            log(INFO, f"VAE epoch {epoch}/{num_epoch}: loss={avg_epoch_loss:.4f}")
    
    return vae



# ============= Snowball Server =============

class SnowballServer(AnomalyDetectionServer):
    """
    Snowball server that uses clustering-based initial filtering followed by VAE-based
    anomaly detection to identify and filter malicious updates.
    """

    def __init__(
        self,
        server_config,
        ct: int = 5,
        vae_initial_epochs: int = 50,
        vae_tuning_epochs: int = 10,
        vae_latent_dim: int = 32,
        vae_hidden_dim: int = 64,
        vae_threshold: float = 0.9,
        vae_step: float = 0.1,
        warmup_rounds: int = 0, # We use checkpoint
        eta: float = 0.5,
        server_type: str = "snowball",
        **kwargs,
    ) -> None:
        super().__init__(server_config, server_type=server_type, eta=eta, **kwargs)
        
        # Snowball parameters
        self.ct = ct
        self.vae_initial_epochs = int(vae_initial_epochs)
        self.vae_tuning_epochs = int(vae_tuning_epochs)
        self.vae_latent_dim = int(vae_latent_dim)
        self.vae_hidden_dim = int(vae_hidden_dim)
        self.vae_threshold = float(vae_threshold)
        self.vae_step = float(vae_step)
        
        # Warmup rounds before using VAE (dataset-specific)
        self.warmup_rounds = int(warmup_rounds)
        
        log(
            INFO,
            (
                f"Initialized Snowball server with ct={self.ct}, "
                f"vae_initial_epochs={self.vae_initial_epochs}, vae_tuning_epochs={self.vae_tuning_epochs}, "
                f"vae_latent_dim={self.vae_latent_dim}, vae_hidden_dim={self.vae_hidden_dim}, "
                f"vae_threshold={self.vae_threshold}, warmup_rounds={self.warmup_rounds}, eta={self.eta}"
            ),
        )

    def _get_layer_list(self) -> List[str]:
        """Get layer list based on model architecture."""
        model_name = self.config.model.lower()
        
        if 'resnet' in model_name:
            # ResNet models: first conv layer and final fc layer
            return ['conv1', 'fc']
        elif 'vgg' in model_name:
            # VGG models: first conv layer and classifier layers
            return ['conv1', 'classifier']
        elif 'mnistnet' in model_name or 'mnist' in model_name:
            # MnistNet: conv layers and fc layers
            return ['conv1', 'fc2']
        elif 'gru' in model_name or 'lstm' in model_name:
            # RNN models: input weights and final fc layer
            return ['weight_ih_l0', 'fc2']
        elif 'albert' in model_name or 'transformer' in model_name:
            # Transformer models: embedding and classifier
            return ['encoder', 'decoder', 'classifier']
        else:
            raise NotImplementedError(f"Snowball layer list not defined for model {self.config.model}")

    def detect_anomalies(
        self, client_updates: List[Tuple[client_id, num_examples, ModelUpdate]]
    ) -> Tuple[List[int], List[int]]:
        """
        Detect anomalous updates using Snowball's two-phase approach:
        1. Clustering-based initial filtering using Calinski-Harabasz score
        2. VAE-based refinement (after warmup rounds)
        """
        client_diffs = []
        client_ids = []

        for client_id, _, updates in client_updates:            
            client_diffs.append({key: updates[key] for key in self.trainable_names})
            client_ids.append(client_id)

        num_clients = len(client_ids)
        
        # Phase 1: Clustering-based initial filtering
        benign_idx = self._clustering_phase(client_diffs, client_ids)

        # If in warmup period, return clustering results only
        if self.current_round < self.warmup_rounds:
            log(INFO, f"Snowball: Warmup round {self.current_round}/{self.warmup_rounds}, using clustering only")
            malicious_clients = [client_ids[i] for i in range(num_clients) if i not in benign_idx]
            benign_clients = [client_ids[i] for i in benign_idx]
            return malicious_clients, benign_clients
        
        # Phase 2: VAE-based refinement
        benign_idx = self._vae_refinement_phase(client_diffs, benign_idx, client_ids)

        malicious_clients = [client_ids[i] for i in range(num_clients) if i not in benign_idx]
        benign_clients = [client_ids[i] for i in benign_idx]

        return malicious_clients, benign_clients

    def _clustering_phase(self, model_updates: List[ModelUpdate], chosen_clients: List[int]) -> List[int]:
        """
        Phase 1: Use clustering with Calinski-Harabasz score to identify initial benign set.
        """
        # Extract kernels for each layer
        kernels = []
        layer_names = list(model_updates[0].keys())
        for key in layer_names:
            kernels.append([model_updates[idx_client][key] for idx_client in range(len(model_updates))])
        
        # Score accumulator for each client
        cnt = [0.0 for _ in range(len(model_updates))]
        layer_list = self._get_layer_list()
        
        # Analyze each relevant layer
        for idx_layer, layer_name in enumerate(layer_names):
            # Skip layers not in our target list
            should_analyze = any(target in layer_name for target in layer_list)
            if not should_analyze:
                continue
            
            # Skip reverse layers for RNN models
            if '_reverse' in layer_name:
                continue
            
            log(INFO, f"Snowball: Analyzing layer {layer_name}")
            
            # Flatten kernels for this layer
            updates_kernel = [item.detach().flatten().cpu().numpy() for item in kernels[idx_layer]]
            
            score_list_cur_layer = []
            benign_list_cur_layer = []
            
            for idx_client in range(len(updates_kernel)):
                # Compute differences from this client to all others
                ddif = [updates_kernel[idx_client] - updates_kernel[i] for i in range(len(updates_kernel))]
                norms = np.linalg.norm(ddif, axis=1)
                norm_rank = np.argsort(norms)
                
                # Select most suspicious clients (furthest away)
                suspicious_idx = norm_rank[-self.ct:]
                
                # Create cluster centroids: current client + suspicious clients
                centroid_ids = [idx_client]
                centroid_ids.extend(suspicious_idx.tolist())
                
                # Perform clustering with K-means
                cluster_result = cluster(centroid_ids, ddif)
                
                # Calculate Calinski-Harabasz score (higher is better separation)
                score_ = calinski_harabasz_score(ddif, cluster_result)
                
                # Find benign clients (in same cluster as current client)
                benign_ids = np.argwhere(cluster_result == cluster_result[idx_client]).flatten()
                
                benign_list_cur_layer.append(benign_ids)
                score_list_cur_layer.append(score_)
            
            # Normalize scores
            score_array = np.array(score_list_cur_layer)
            std_, mean_ = np.std(score_array), np.mean(score_array)
            
            # Find effective clients (those with positive scores)
            effective_ids = np.argwhere(score_array > 0).flatten()
            if len(effective_ids) < int(len(score_array) * 0.1):
                log(WARNING, f"Snowball Layer {layer_name}: Less than 10% effective clients, adjusting selection")
                effective_ids = np.argsort(-score_array)[:int(len(score_array) * 0.1)]
            
            # Min-max normalization (avoid division by zero)
            score_range = np.max(score_array) - np.min(score_array)
            if score_range > 0:
                score_array = (score_array - np.min(score_array)) / score_range
            else:
                score_array = np.ones_like(score_array)
            log(INFO, f'Snowball Layer {layer_name}: STD={std_:.4f}, Mean={mean_:.4f}')
            
            # Accumulate scores for benign clients
            for idx_client in effective_ids:
                for idx_b in benign_list_cur_layer[idx_client]:
                    cnt[idx_b] += score_array[idx_client]
        
        # Select top clients based on accumulated scores
        cnt_rank = np.argsort(-np.array(cnt))
        num_selected = max(math.ceil(len(cnt_rank) * 0.1), 2) # Require at least 2 clients
        selected_ids = cnt_rank[:num_selected].tolist()

        log(INFO, f"Snowball clustering phase: Selected {len(selected_ids)} benign clients from initial filtering")
        
        return selected_ids

    def _vae_refinement_phase(
        self, model_updates: List[ModelUpdate], initial_benign_idx: List[int], chosen_clients: List[int]
    ) -> List[int]:
        """
        Phase 2: Use VAE to refine the benign set by iteratively adding clients
        with low reconstruction error.
        """
        # Flatten model updates for VAE processing
        layer_list = self._get_layer_list()
        ignore_pattern = '_reverse' if 'gru' in self.config.model.lower() or 'lstm' in self.config.model.lower() else None
        
        flatten_update_list = [
            _flatten_model(update, layer_list=layer_list, ignore=ignore_pattern)
            for update in model_updates
        ]
        
        selected_ids = initial_benign_idx.copy()
        target_size = int(len(chosen_clients) * self.vae_threshold)
        
        log(INFO, f"Snowball VAE refinement: Starting with {len(selected_ids)} clients, target={target_size}")
        
        # Check if we have enough clients to build difference set
        if len(selected_ids) < 2:
            log(WARNING, f"Snowball VAE: Only {len(selected_ids)} benign client(s) found, need at least 2 for VAE. Returning all clients.")
            return list(range(len(chosen_clients)))
        
        # If already at or above target, return selected ids
        if len(selected_ids) >= target_size:
            log(INFO, f"Snowball VAE: Already have {len(selected_ids)} >= {target_size} clients, skipping VAE refinement")
            return selected_ids
        
        # Train initial VAE on differences from initial benign set
        log(INFO, f"Snowball VAE: Training initial VAE with {self.vae_initial_epochs} epochs...")
        vae_start_time = time.time()
        vae = train_vae(
            None,
            build_dif_set([flatten_update_list[i] for i in selected_ids]),
            self.vae_initial_epochs,
            device=self.device,
            latent=self.vae_latent_dim,
            hidden=self.vae_hidden_dim
        )
        log(INFO, f"Snowball VAE: Initial training completed in {time.time() - vae_start_time:.2f}s")
        
        # Iteratively add benign clients based on reconstruction error
        iteration = 0
        total_refinement_time = 0.0
        while len(selected_ids) < target_size:
            iteration += 1
            iter_start_time = time.time()
            
            # Check if there are remaining clients to add
            rest_ids = [i for i in range(len(flatten_update_list)) if i not in selected_ids]
            if len(rest_ids) == 0:
                log(INFO, f"Snowball VAE: No more clients to add, stopping at {len(selected_ids)} clients")
                break
            
            # Fine-tune VAE on current benign set
            vae = train_vae(
                vae,
                build_dif_set([flatten_update_list[i] for i in selected_ids]),
                self.vae_tuning_epochs,
                device=self.device,
                latent=self.vae_latent_dim,
                hidden=self.vae_hidden_dim
            )
            
            vae.eval()
            
            with torch.no_grad():
                loss_list = []
                
                for idx in rest_ids:
                    m_loss = 0.0
                    loss_cnt = 0
                    
                    # Calculate average reconstruction loss for differences
                    for dif in obtain_dif([flatten_update_list[i] for i in selected_ids], flatten_update_list[idx]):
                        m_loss += vae.recon_prob(dif)
                        loss_cnt += 1
                    
                    if loss_cnt > 0:
                        m_loss /= loss_cnt
                    loss_list.append(m_loss)
            
            # Add clients with lowest reconstruction error
            rank_ = np.argsort(loss_list)
            # log(INFO, f"Snowball VAE iteration {iteration}: Reconstruction loss {loss_list}")

            num_to_add = min(
                math.ceil(len(chosen_clients) * self.vae_step),
                target_size - len(selected_ids)
            )

            new_clients = np.array(rest_ids)[rank_[:num_to_add]].tolist()
            selected_ids.extend(new_clients)
            
            iter_time = time.time() - iter_start_time
            total_refinement_time += iter_time

            log(INFO, f"Snowball VAE iteration {iteration}: Added {len(new_clients)} clients (total_iter_time={iter_time:.2f}s), current_total={len(selected_ids)}")

        log(INFO, f"Snowball VAE refinement complete: {len(selected_ids)} benign clients selected in {iteration} iterations (total_refinement_time={total_refinement_time:.2f}s)")
        return selected_ids
