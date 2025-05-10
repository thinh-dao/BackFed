python main.py -m \
    aggregator=flame \
    atk_config=multishot \
    atk_config.data_poison_method=pattern,distributed,edge_case,a3fl,iba \
    atk_config.poison_start_round=2000 \
    atk_config.poison_end_round=2300 \
    checkpoint=checkpoints/CIFAR10_unweighted_fedavg/ResNet18_round_2000_dir_0.5.pth \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"5,4,2,1,0\" \
    dir_tag=anomaly_detection_cifar10

python main.py -m \
    aggregator=deepsight \
    atk_config=multishot \
    atk_config.data_poison_method=pattern,distributed,edge_case,a3fl,iba \
    atk_config.poison_start_round=2000 \
    atk_config.poison_end_round=2300 \
    checkpoint=checkpoints/CIFAR10_unweighted_fedavg/ResNet18_round_2000_dir_0.5.pth \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"5,4,2,1,0\" \
    dir_tag=anomaly_detection_cifar10

python main.py -m \
    aggregator=rflbat \
    atk_config=multishot \
    atk_config.data_poison_method=pattern,distributed,edge_case,a3fl,iba \
    atk_config.poison_start_round=2000 \
    atk_config.poison_end_round=2300 \
    checkpoint=checkpoints/CIFAR10_unweighted_fedavg/ResNet18_round_2000_dir_0.5.pth \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"5,4,2,1,0\" \
    dir_tag=anomaly_detection_cifar10
