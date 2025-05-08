#!/bin/bash
# Example script for running experiments with hybrid defenses

# FLAME defense (anomaly detection + robust aggregation)
echo "Running experiment with FLAME defense"
python main.py \
    --server_type flame \
    --dataset cifar10 \
    --data_partition dirichlet \
    --alpha 0.5 \
    --attack_type random \
    --attack_mode multi-shot \
    --poison_start_round 200 \
    --poison_end_round 300 \
    --checkpoint 200 \
    --save_logging csv \
    --num_rounds 600 \
    --lamda 0.001

# FLARE defense (anomaly detection + robust aggregation)
echo "Running experiment with FLARE defense"
python main.py \
    --server_type flare \
    --dataset cifar10 \
    --data_partition dirichlet \
    --alpha 0.5 \
    --attack_type random \
    --attack_mode multi-shot \
    --poison_start_round 200 \
    --poison_end_round 300 \
    --checkpoint 200 \
    --save_logging csv \
    --num_rounds 600 \
    --voting_threshold 0.5
