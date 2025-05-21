#!/bin/bash

# Run sentiment analysis on Sentiment140 dataset
python main.py \
    dataset=sentiment140 \
    num_clients=100 \
    num_clients_per_round=10 \
    num_rounds=50 \
    client_config.local_epochs=5 \
    no_attack=True \
    save_logging=csv \
    dir_tag=sentiment140_fedavg
