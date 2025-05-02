python main.py -m \
    aggregator=coordinate_median,trimmed_mean,multi_krum,weakdp,foolsgold,robustlr,norm_clipping \
    atk_config.data_poison_method=pattern \
    atk_config.selection_scheme=all-adversary,random \
    atk_config.poison_start_round=2000 \
    atk_config.poison_end_round=2300 \
    atk_config.scale_weights=False,True \
    checkpoint=checkpoints/CIFAR10_unweighted_fedavg/ResNet18_round_2000_dir_0.5.pth \
    save_logging=csv \
    num_rounds=300 \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"3,4,5,6,7\"

python main.py -m \
    aggregator=coordinate_median,trimmed_mean,multi_krum,weakdp,foolsgold,robustlr,norm_clipping \
    atk_config.data_poison_method=distributed \
    atk_config.selection_scheme=all-adversary,random \
    atk_config.poison_start_round=2000 \
    atk_config.poison_end_round=2300 \
    checkpoint=checkpoints/CIFAR10_unweighted_fedavg/ResNet18_round_2000_dir_0.5.pth \
    save_logging=csv \
    num_rounds=300 \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"3,4,5,6,7\"

python main.py -m \
    aggregator=coordinate_median,trimmed_mean,multi_krum,weakdp,foolsgold,robustlr,norm_clipping \
    atk_config.data_poison_method=a3fl \
    atk_config.selection_scheme=all-adversary,random \
    atk_config.poison_start_round=2000 \
    atk_config.poison_end_round=2300 \
    checkpoint=checkpoints/CIFAR10_unweighted_fedavg/ResNet18_round_2000_dir_0.5.pth \
    save_logging=csv \
    num_rounds=300 \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"1,7,6,3,5\"

python main.py -m \
    aggregator=coordinate_median,trimmed_mean,multi_krum,weakdp,foolsgold,robustlr,norm_clipping \
    atk_config.data_poison_method=iba,edge_case \
    atk_config.selection_scheme=all-adversary,random \
    atk_config.poison_start_round=2000 \
    atk_config.poison_end_round=2300 \
    checkpoint=checkpoints/CIFAR10_unweighted_fedavg/ResNet18_round_2000_dir_0.5.pth \
    save_logging=csv \
    num_rounds=300 \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"2,6,7,4,1\"