#!/bin/bash

# Run sentiment analysis on Sentiment140 dataset with backdoor attack
python main.py \
    dataset=sentiment140 \
    num_clients=100 \
    num_clients_per_round=10 \
    num_rounds=50 \
    client_config.local_epochs=5 \
    no_attack=False \
    atk_config=singleshot \
    atk_config.data_poison_method=pattern \
    atk_config.poison_start_round=10 \
    atk_config.poison_end_round=20 \
    atk_config.malicious_clients=[0,1,2,3,4] \
    save_logging=csv \
    dir_tag=sentiment140_attack
