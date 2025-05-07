python main.py -m \
    aggregator=coordinate_median,trimmed_mean,multi_krum,weakdp,foolsgold,robustlr,norm_clipping \
    atk_config=multishot \
    atk_config.data_poison_method=pattern \
    atk_config.poison_start_round=2000 \
    atk_config.poison_end_round=2300 \
    checkpoint=checkpoints/CIFAR10_unweighted_fedavg/ResNet18_round_2000_dir_0.5.pth \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"0,1,2,4,5\" \
    dir_tag=robust_aggregation


python main.py -m \
    aggregator=coordinate_median,trimmed_mean,multi_krum,weakdp,foolsgold,robustlr,norm_clipping \
    atk_config=multishot \
    atk_config.data_poison_method=distributed \
    atk_config.poison_start_round=2000 \
    atk_config.poison_end_round=2300 \
    checkpoint=checkpoints/CIFAR10_unweighted_fedavg/ResNet18_round_2000_dir_0.5.pth \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"0,1,2,4,5\" \
    dir_tag=robust_aggregation


python main.py -m \
    aggregator=coordinate_median,trimmed_mean,multi_krum,weakdp,foolsgold,robustlr,norm_clipping \
    atk_config=multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.poison_start_round=2000 \
    atk_config.poison_end_round=2300 \
    checkpoint=checkpoints/CIFAR10_unweighted_fedavg/ResNet18_round_2000_dir_0.5.pth \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"0,1,2,4,5\" \
    dir_tag=robust_aggregation


python main.py -m \
    aggregator=coordinate_median,trimmed_mean,multi_krum,weakdp,foolsgold,robustlr,norm_clipping \
    atk_config=multishot \
    atk_config.data_poison_method=a3fl \
    atk_config.poison_start_round=2000 \
    atk_config.poison_end_round=2300 \
    checkpoint=checkpoints/CIFAR10_unweighted_fedavg/ResNet18_round_2000_dir_0.5.pth \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"0,1,2,3,4\" \
    dir_tag=robust_aggregation

python main.py -m \
    aggregator=coordinate_median,trimmed_mean,multi_krum,weakdp,foolsgold,robustlr,norm_clipping \
    atk_config=multishot \
    atk_config.data_poison_method=iba \
    atk_config.poison_start_round=2000 \
    atk_config.poison_end_round=2300 \
    checkpoint=checkpoints/CIFAR10_unweighted_fedavg/ResNet18_round_2000_dir_0.5.pth \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"5,4,2,1,0\" \
    dir_tag=robust_aggregation

