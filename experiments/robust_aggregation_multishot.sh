######## Robust Aggregation defense against multishot attack ########

############## CIFAR10 ################
# One-line argument using Hydra --multirun 
# For efficiency, you may run attacks in different processes

python main.py -m -cn cifar10 \
    aggregator=coordinate_median,geometric_median,trimmed_mean,krum,foolsgold,robustlr,norm_clipping,weakdp \
    no_attack=True \
    num_rounds=300 \
    save_checkpoint=True \
    "save_checkpoint_rounds=[2300]" \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"0,1,2,3,4\" \
    dir_tag=cifar10_pretrain_robust_aggregation





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
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern,edge_case,a3fl,iba,distributed \
    atk_config.poison_start_round=2301 \
    atk_config.poison_end_round=2600 \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"0,1,2,4,5\" \
    dir_tag=cifar10_robust_aggregation


python main.py -m -cn cifar10 \
    aggregator=coordinate_median,geometric_median,trimmed_mean,krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern,distributed,edge_case,iba \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=200 \
    dir_tag=cifar10_robust_aggregation


python main.py -m -cn femnist \
    aggregator=coordinate_median,geometric_median,trimmed_mean,krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=pattern,distributed,edge_case,iba \
    cuda_visible_devices=\"1,2,3\" \
    num_rounds=200 \
    dir_tag=femnist_robust_aggregation



#### edge-case only
python main.py -m -cn cifar10 \
    aggregator=coordinate_median,geometric_median,trimmed_mean,krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=edge_case \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=200 \
    dir_tag=cifar10_robust_aggregation_edgecase


python main.py -m -cn femnist \
    aggregator=coordinate_median,geometric_median,trimmed_mean,krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=edge_case \
    cuda_visible_devices=\"0,2,1,4,5\" \
    num_rounds=200 \
    dir_tag=femnist_robust_aggregation_edgecase

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
    num_rounds=600 \
    atk_config=emnist_multishot \
    atk_config.data_poison_method=pattern,edge_case,a3fl,iba,distributed \
    atk_config.poison_start_round=1301 \
    atk_config.poison_end_round=1600 \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"0,1,2,4,5\" \
    dir_tag=emnist_robust_aggregation
