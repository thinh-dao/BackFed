#!/bin/bash
# Example script for running experiments with robust aggregation defenses

# TrimmedMean defense
echo "Running experiment with TrimmedMean defense"
python main.py \
    --server_type trimmed_mean \
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
    --trim_ratio 0.2

# MultiKrum defense
echo "Running experiment with MultiKrum defense"
python main.py \
    --server_type multi_krum \
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
    --num_malicious 2

# GeometricMedian defense
echo "Running experiment with GeometricMedian defense"
python main.py \
    --server_type geometric_median \
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

# CoordinateMedian defense
echo "Running experiment with CoordinateMedian defense"
python main.py \
    --server_type coordinate_median \
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

# NormClipping defense
echo "Running experiment with NormClipping defense"
python main.py \
    --server_type norm_clipping \
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
    --max_norm 1.0

# FLTrust defense
echo "Running experiment with FLTrust defense"
python main.py \
    --server_type fltrust \
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

# RobustLR defense
echo "Running experiment with RobustLR defense"
python main.py \
    --server_type robustlr \
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
    --robustLR_threshold 0.5
