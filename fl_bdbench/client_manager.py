"""
Client selection strategies.
"""
import random
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

from typing import Dict, List
from logging import WARNING
from fl_bdbench.utils import log
from fl_bdbench.clients import BenignClient
from hydra.utils import get_class


class ClientManager:
    """
    Custom client manager to evaluate attacks. At initialization, malicious clients and poison rounds are selected 
    based on the attack configuration.
    """
    def __init__(self, config, start_round=0):
        self.config = config
        self.atk_config = self.config.atk_config if not self.config.no_attack else None
        self.start_round = start_round
        # Store the selected clients for each round. In each round, we have a dictionary of client classes and the clients selected for that type.
        self.rounds_selection: Dict[int, Dict[type, List[int]]] = {}
        self.poison_rounds = []  # Store all poison rounds
        self.num_clients_per_round = int(round(self.config.fraction_fit * self.config.num_clients))

        # If no attack, we don't need to initialize poison rounds
        if config.no_attack:
            self.malicious_clients = []
            self.benign_clients = list(range(self.config.num_clients))
            # Initialize benign client class for rounds_selection
            self.benign_client_class = BenignClient
            self._initialize_normal_rounds()
        else:
            self.malicious_clients = self.atk_config.malicious_clients
            self.benign_clients = [i for i in range(self.config.num_clients) if i not in self.malicious_clients]
            
            # Initialize malicious and benign client classes for rounds_selection
            model_poison_method = self.atk_config["model_poison_method"]
            self.malicious_client_class = get_class(self.atk_config.model_poison_config[model_poison_method]._target_)
            self.benign_client_class = BenignClient

            if self.start_round >= self.atk_config.poison_end_round or self.start_round + self.config.num_rounds + 1 <= self.atk_config.poison_start_round:
                log(WARNING, f"Training rounds [{self.start_round} - {self.start_round + self.config.num_rounds + 1}] are out of scope for the attack range [{self.atk_config.poison_start_round} - {self.atk_config.poison_end_round}]. No attack will be applied.")
                self._initialize_normal_rounds()
            else:
                self._initialize_poison_rounds()
                self._initialize_normal_rounds()

    def _initialize_poison_rounds(self):
        """Initialize poison rounds based on the attack configuration."""
        selection_scheme = self.atk_config.selection_scheme
        poison_frequency = self.atk_config.poison_frequency

        if poison_frequency == "single-shot":
            selected_poison_rounds = [self.atk_config.poison_start_round]
        elif poison_frequency == "multi-shot":
            start, end, interval = self.atk_config.poison_start_round, self.atk_config.poison_end_round, self.atk_config.poison_interval
            selected_poison_rounds = list(range(start, end + 1, interval))
        else:
            raise ValueError(f"Invalid poison scheme {poison_frequency}")

        if selection_scheme == "single-adversary":
            self._single_adversary_selection(selected_poison_rounds)
        elif selection_scheme == "multi-adversary":
            self._multi_adversary_selection(selected_poison_rounds)
        elif selection_scheme == "all-adversary":
            self._all_adversary_selection(selected_poison_rounds)
        elif selection_scheme == "random":
            self._random_selection(poison_frequency)
        elif selection_scheme == "manual":
            self._manual_selection()
        else:
            raise ValueError(f"Invalid selection scheme {selection_scheme}. Choose between ['random', 'manual', 'all-adversary', 'single-adversary', 'multi-adversary']")

    def _initialize_normal_rounds(self):
        """Initialize normal rounds. Only update rounds that are not poisoned."""
        for r in range(self.start_round, self.start_round + self.config.num_rounds + 1):
            if r not in self.poison_rounds:
                self.rounds_selection[r] = {
                    self.benign_client_class: random.sample(self.benign_clients, self.num_clients_per_round)
                }

    def _single_adversary_selection(self, selected_rounds):
        """Each adversary is selected consecutively for poisoning in each communication round."""
        self.poison_rounds = selected_rounds
        for r in selected_rounds:
            round_idx = selected_rounds.index(r) % len(self.malicious_clients)
            self.rounds_selection[r] = {
                self.malicious_client_class: [self.malicious_clients[round_idx]],
                self.benign_client_class: random.sample(self.benign_clients, self.num_clients_per_round - 1)
            }

    def _multi_adversary_selection(self, selected_rounds):
        """Randomly select {num_adversaries_per_round} adversaries for poisoning in each communication round."""
        num_adversaries_per_round = min(self.atk_config.num_adversaries_per_round, len(self.malicious_clients), self.num_clients_per_round)
        self.poison_rounds = selected_rounds
        for r in selected_rounds:
            self.rounds_selection[r] = {
                self.malicious_client_class: random.sample(self.malicious_clients, num_adversaries_per_round),
                self.benign_client_class: random.sample(self.benign_clients, self.num_clients_per_round - num_adversaries_per_round)
            }

    def _all_adversary_selection(self, selected_rounds):
        """All adversaries are selected for poisoning in each communication round."""
        num_adversaries_per_round = min(len(self.malicious_clients), self.num_clients_per_round)
        self.poison_rounds = selected_rounds
        for r in selected_rounds:
            self.rounds_selection[r] = {
                self.malicious_client_class: random.sample(self.malicious_clients, num_adversaries_per_round),
                self.benign_client_class: random.sample(self.benign_clients, self.num_clients_per_round - num_adversaries_per_round)
            }

    def _random_selection(self, poison_frequency):
        """Randomly select clients. If malicious clients are selected, they will poison the data."""
        start, end = self.atk_config.poison_start_round, self.atk_config.poison_end_round
        for r in range(start, end + 1):
            selected_clients = random.sample(range(self.config.num_clients), self.num_clients_per_round)
            if any(client in self.malicious_clients for client in selected_clients):
                malicious_clients = [client for client in selected_clients if client in self.malicious_clients]
                benign_clients = [client for client in selected_clients if client not in self.benign_clients]

                # Update the rounds_selection dictionary and poison_rounds list
                self.rounds_selection[r] = {
                    self.malicious_client_class: malicious_clients
                }
                if len(benign_clients) > 0: self.rounds_selection[r].update({self.benign_client_class: benign_clients})
                self.poison_rounds.append(r)

                if poison_frequency == "single-shot":
                    break  # Only attack once.
            else:
                self.rounds_selection[r] = {
                    self.benign_client_class: selected_clients
                }

    def _manual_selection(self):
        """Manually specify the poisoning rounds and malicious clients for each poisoning round."""
        manual_poison_rounds = self.atk_config.poison_rounds
        self.poison_rounds = list(manual_poison_rounds.keys())
        for r in manual_poison_rounds.keys():
            self.rounds_selection[r] = {
                self.malicious_client_class: manual_poison_rounds[r],
                self.benign_client_class: random.sample(self.benign_clients, self.num_clients_per_round - len(manual_poison_rounds[r]))
            }
            
    def get_rounds_selection(self):
        """Get the client selection for each round."""
        return self.rounds_selection

    def get_poison_rounds(self):
        """Get the list of poisoning rounds."""
        return self.poison_rounds

    def get_client_info(self):
        """Get info about malicious and benign clients."""
        return dict(malicious=self.malicious_clients, benign=self.benign_clients)

    def get_malicious_clients(self):
        """Get the list of malicious clients."""
        return self.malicious_clients

    def get_benign_clients(self):
        """Get the list of benign clients."""
        return self.benign_clients

    def get_num_clients_per_round(self):
        """Get the number of clients per round."""
        return self.num_clients_per_round

    def plot_client_selection(self, start_round=-1, end_round=-1, interval=2, only_poison_rounds=False, save_path=None):
        """
        Plot a heatmap showing client selection over communication rounds. Malicious clients and poisoned rounds are highlighted in red.
        """
        if end_round == -1:
            end_round = max(list(self.selection_history.keys()))
        if start_round == -1:
            start_round = max(1, end_round - 100)  # Default to 100 rounds before the last round

        if only_poison_rounds:
            selected_rounds = list(self.get_poison_rounds().keys())
        else:
            selected_rounds = list(range(start_round, end_round + 1, interval))

        num_clients = self.config.num_clients
        num_rounds = len(selected_rounds)

        # Initialize a matrix with zeros
        data = np.zeros((num_clients, num_rounds))

        for idx, r in enumerate(selected_rounds):
            if r in self.selection_history:
                selected_clients = np.array(self.selection_history[r])
                data[selected_clients, idx] = 1  # Highlight selected clients

        fig_width = num_rounds   # Width scales with the number of clients
        fig_height = num_clients # Height scales with the number of classes

        if fig_width > fig_height:
            fig_height = fig_height * 24 / fig_width + 8
            fig_width = 24
            scaling_factor = fig_width / (num_clients ** 0.8)
        else:
            fig_width = fig_width * 24 / fig_height + 8
            fig_height = 24
            scaling_factor = fig_height / (num_rounds ** 0.8)

        # Create a DataFrame for better labeling
        df = pd.DataFrame(data, index=[f'Client {i}' for i in range(num_clients)], columns=[f'{i}' for i in selected_rounds])

        # Plot the heatmap using Seaborn
        plt.figure(figsize=(fig_width, fig_height))

        poisoned_rounds = list(self.get_poison_rounds().keys())
        malicious_clients = self.get_malicious_clients()

        # Create a custom colormap
        cmap = matplotlib.colormaps.get_cmap("Blues")
        cmap.set_bad(color='red')

        # Create a mask to highlight poisoned cells where malicious clients are selected
        mask = np.zeros_like(data, dtype=bool)
        for i, client in enumerate(df.index):
            client_id = int(client.split(" ")[1])
            if client_id in malicious_clients:
                for j, round_name in enumerate(df.columns):
                    if int(round_name) in poisoned_rounds and client_id in self.selection_history[int(round_name)]:
                        mask[i, j] = True  # Mark cells to be highlighted in red

        # Plot heatmap with masked cells highlighted in red
        sns.heatmap(df, cmap=cmap, cbar=False, linewidths=.5, linecolor='lightgrey', mask=mask, vmin=0, vmax=1)

        # Customize the plot
        plt.title('Client Selection Over Communication Rounds', fontsize=min(24*scaling_factor, 24))
        plt.xlabel('Communication Round', fontsize=min(20 * scaling_factor, 20))
        plt.ylabel('Client ID', fontsize=min(20 * scaling_factor, 20))
        plt.xticks(fontsize=min(12 * scaling_factor, 12))
        plt.yticks(fontsize=min(12 * scaling_factor, 12), rotation=0)

        # Change the color of the y-axis ticks for malicious clients
        ax = plt.gca()  # Get the current axes
        yticks = ax.get_yticklabels()  # Get the y-axis tick labels
        for tick in yticks:
            client_id = int(tick.get_text().split(" ")[1])
            if client_id in malicious_clients:  # Check if the client is malicious
                tick.set_color('red')  # Set the color to red

        # Change the color of the x-axis ticks for poisoned rounds
        xticks = ax.get_xticklabels()  # Get the x-axis tick labels
        for tick in xticks:
            r = int(tick.get_text())
            if r in poisoned_rounds:  # Check if the round is poisoned
                tick.set_color('red')  # Set the color to red

        if save_path is not None:
            path = os.path.join(save_path, "client_selection.pdf")
            plt.savefig(path, dpi=500, bbox_inches='tight')
        else:
            plt.show()
