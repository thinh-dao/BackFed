#!/bin/bash
# Example script for running experiments with client-side defenses

# FedProx defense
echo "Running experiment with FedProx defense"
python main.py \
    --server_type fedprox \
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
    --proximal_mu 0.01

# WeakDP defense
echo "Running experiment with WeakDP defense"
python main.py \
    --server_type weakdp \
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
    --noise_multiplier 0.1 \
    --max_norm 1.0
