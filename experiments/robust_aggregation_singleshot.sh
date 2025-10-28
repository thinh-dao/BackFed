######## Robust Aggregation singleshot attack ########

############## CIFAR10 ################

# One-line argument using Hydra --multirun 
# For efficiency, you can run attacks in different processes
python main.py -m -cn cifar10 \
    aggregator=coordinate_median,geometric_median,trimmed_mean,krum,foolsgold,robustlr,norm_clipping \
    no_attack=True \
    num_rounds=300 \
    save_checkpoint=True \
    "save_checkpoint_rounds=[2300]" \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"0,1,2,4,5\" \
    dir_tag=cifar10_pretrain_robust_aggregation && \
python main.py -m -cn cifar10 \
    aggregator=coordinate_median,geometric_median,trimmed_mean,krum,foolsgold,robustlr,norm_clipping \
    checkpoint=2300 \
    num_rounds=300 \
    atk_config=cifar10_singleshot \
    atk_config.data_poison_method=pattern,edge_case,a3fl,iba,distributed \
    atk_config.poison_start_round=2301 \
    atk_config.poison_end_round=2301 \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"0,1,2,4,5\" \
    dir_tag=cifar10_robust_aggregation



############## EMNIST ################
python main.py -m -cn emnist \
    aggregator=coordinate_median,geometric_median,trimmed_mean,krum,foolsgold,robustlr,norm_clipping \
    no_attack=True \
    num_rounds=300 \
    save_checkpoint=True \
    "save_checkpoint_rounds=[1300]" \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"0,1,2,4,5\" \
    dir_tag=emnist_pretrain_robust_aggregation && \
python main.py -m -cn emnist \
    aggregator=coordinate_median,geometric_median,trimmed_mean,krum,foolsgold,robustlr,norm_clipping \
    checkpoint=1300 \
    num_rounds=300 \
    atk_config=emnist_singleshot \
    atk_config.data_poison_method=pattern,edge_case,a3fl,iba,distributed \
    atk_config.poison_start_round=1301 \
    atk_config.poison_end_round=1301 \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"0,1,2,4,5\" \
    dir_tag=emnist_robust_aggregation
