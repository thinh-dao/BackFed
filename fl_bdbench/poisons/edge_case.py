import torch
import torchvision.transforms.v2 as transforms
import numpy as np
import pickle
import random
import ray

from .base import Poison
from omegaconf import DictConfig
from logging import INFO, WARNING
from flwr.common.logger import log

DEFAULT_TRANSFORMS = {
    "CIFAR10": transforms.Compose([
        transforms.ToImage(), 
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Resize((32, 32))
    ]),
    "MNIST": transforms.Compose([
        transforms.ToImage(), 
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Resize((28, 28))
    ]),
    "NIST": transforms.Compose([
        transforms.ToImage(), 
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Resize((28, 28))
    ])
}

class EdgeCase(Poison):
    def __init__(self, params: DictConfig, client_id: int = -1): 
        super(EdgeCase, self).__init__(params, client_id)
        
        if self.params.attack_type == "all2all":
            raise ValueError(f"Edge-case is not supported for all2all attack")

        dataset = self.params.dataset.upper()
        if dataset in DEFAULT_TRANSFORMS:
            self.transform_edge_case = DEFAULT_TRANSFORMS[dataset]
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
                
        self._load_edge_case()      
    
    def _load_edge_case(self):            
        # Load data from disk if not available in context actor
        if self.params.dataset.upper() == "CIFAR10":
            with open('fl_bdbench/poisons/shared/edge-case/southwest_images_new_train.pkl', 'rb') as train_f:
                saved_southwest_dataset_train = pickle.load(train_f)
            with open('fl_bdbench/poisons/shared/edge-case/southwest_images_new_test.pkl', 'rb') as test_f:
                saved_southwest_dataset_test = pickle.load(test_f)    
            self.edge_case_train = torch.stack([self.transform_edge_case(img) for img in saved_southwest_dataset_train])
            self.edge_case_test = torch.stack([self.transform_edge_case(img) for img in saved_southwest_dataset_test]) 

        elif "NIST" in self.params.dataset.upper():
            # Load ARDIS train dataset
            train_ardis_images=np.loadtxt('./data/ARDIS/ARDIS_train_2828.csv', dtype='float')
            train_ardis_labels=np.loadtxt('./data/ARDIS/ARDIS_train_labels.csv', dtype='float')

            #### reshape to be [samples][width][height]
            train_ardis_images = train_ardis_images.reshape(train_ardis_images.shape[0], 28, 28).astype('float32')

            #### get the test images with label 7
            train_indices_seven = np.where(train_ardis_labels[:,7] == 1)[0] # labels are one-hot encoded
            train_images_seven = train_ardis_images[train_indices_seven,:]
            train_images_seven = torch.tensor(train_images_seven).type(torch.uint8)
            # train_labels_seven = torch.tensor([7 for y in train_ardis_labels])

            # Load ARDIS test dataset
            test_ardis_images=np.loadtxt('./data/ARDIS/ARDIS_test_2828.csv', dtype='float')
            test_ardis_labels=np.loadtxt('./data/ARDIS/ARDIS_test_labels.csv', dtype='float')

            #### reshape to be [samples][width][height]
            test_ardis_images = test_ardis_images.reshape(test_ardis_images.shape[0], 28, 28).astype('float32')

            #### get the test images with label 7
            test_indices_seven = np.where(test_ardis_labels[:,7] == 1)[0]
            test_images_seven = test_ardis_images[test_indices_seven,:]
            test_images_seven = torch.tensor(test_images_seven).type(torch.uint8)
            # test_labels_seven = torch.tensor([7 for y in test_ardis_labels])

            self.edge_case_train = torch.stack([self.transform_edge_case(img) for img in train_images_seven])
            self.edge_case_test = torch.stack([self.transform_edge_case(img) for img in test_images_seven])
        else:
            raise ValueError(f"Dataset {self.params.dataset} is not supported for edge-case attack")

        log(INFO, f"Number of edge case train: {len(self.edge_case_train)} - test: {len(self.edge_case_test)}")     

    def poison_inputs(self, inputs):
        # Replace inputs with edge-case samples
        poison_choice = random.sample(range(len(self.edge_case_train)), inputs.shape[0])
        poison_inputs = self.edge_case_train[poison_choice].to(inputs.device)

        return poison_inputs

    def poison_test(self, net, test_loader, loss_fn=torch.nn.CrossEntropyLoss(), normalization=None):
        """Validate attack success rate on edge-case samples.

        Args:
            net (torch.nn.Module): The model to test
            test_loader (torch.utils.data.DataLoader): The test loader
            loss_fn (torch.nn.Module): The loss function to use
            normalization (torch.utils.transforms.Normalize): The normalization

        Returns:
            backdoor_loss (float): The loss of backdoor target samples
            backdoor_accuracy (float): The accuracy of targeted misclassification
        """
        net.eval()
        if normalization:
            edge_case_test = normalization(self.edge_case_test)
        else:
            edge_case_test = self.edge_case_test
        edge_case_test = edge_case_test.to("cuda")
        target_labels = torch.tensor([self.params.target_class] * len(edge_case_test), device="cuda")

        with torch.no_grad():
            outputs = net(edge_case_test)
            backdoored_preds = (torch.max(outputs.data, 1)[1] == target_labels).sum().item()
            backdoored_loss = loss_fn(outputs, target_labels).item()

        backdoor_accuracy = backdoored_preds / len(edge_case_test)
        return backdoored_loss, backdoor_accuracy
