#!/bin/bash

# Run sentiment analysis on Sentiment140 dataset with Albert model
python main.py \
    dataset=sentiment140 \
    model=albert-tiny \
    num_clients=100 \
    num_clients_per_round=10 \
    num_rounds=20 \
    client_config.local_epochs=3 \
    sample_size=10000 \
    no_attack=True \
    save_logging=csv \
    dir_tag=sentiment140_albert
