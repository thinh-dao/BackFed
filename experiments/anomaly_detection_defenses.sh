#!/bin/bash
# Example script for running experiments with anomaly detection defenses

# FoolsGold defense
echo "Running experiment with FoolsGold defense"
python main.py \
    --server_type foolsgold \
    --dataset cifar10 \
    --data_partition dirichlet \
    --alpha 0.5 \
    --attack_type random \
    --attack_mode multi-shot \
    --poison_start_round 200 \
    --poison_end_round 300 \
    --checkpoint 200 \
    --save_logging csv \
    --num_rounds 600

# DeepSight defense
echo "Running experiment with DeepSight defense"
python main.py \
    --server_type deepsight \
    --dataset cifar10 \
    --data_partition dirichlet \
    --alpha 0.5 \
    --attack_type random \
    --attack_mode multi-shot \
    --poison_start_round 200 \
    --poison_end_round 300 \
    --checkpoint 200 \
    --save_logging csv \
    --num_rounds 600

# RFLBAT defense
echo "Running experiment with RFLBAT defense"
python main.py \
    --server_type rflbat \
    --dataset cifar10 \
    --data_partition dirichlet \
    --alpha 0.5 \
    --attack_type random \
    --attack_mode multi-shot \
    --poison_start_round 200 \
    --poison_end_round 300 \
    --checkpoint 200 \
    --save_logging csv \
    --num_rounds 600

# FLDetector defense
echo "Running experiment with FLDetector defense"
python main.py \
    --server_type fldetector \
    --dataset cifar10 \
    --data_partition dirichlet \
    --alpha 0.5 \
    --attack_type random \
    --attack_mode multi-shot \
    --poison_start_round 200 \
    --poison_end_round 300 \
    --checkpoint 200 \
    --save_logging csv \
    --num_rounds 600
