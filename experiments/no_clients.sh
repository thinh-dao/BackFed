python main.py -m -cn cifar10 \
    aggregator=unweighted_fedavg \
    aggregator_config.unweighted_fedavg.eta=0.5 \
    atk_config=cifar10_multishot \
    atk_config.poison_start_round=2000 \
    atk_config.poison_end_round=2200 \
    num_clients_per_round=10,20,30,40,50 \
    checkpoint=checkpoints/CIFAR10_unweighted_fedavg_0.5/resnet18_round_2000_dir_0.9.pth \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"0,1,2,3,4\" \
    dir_tag=num_clients_per_round

python main.py -cn cifar10 \
    aggregator=unweighted_fedavg \
    aggregator_config.unweighted_fedavg.eta=0.5 \
    atk_config=cifar10_multishot \
    atk_config.poison_start_round=2000 \
    atk_config.poison_end_round=2200 \
    num_clients_per_round=10 \
    checkpoint=checkpoints/CIFAR10_unweighted_fedavg_0.5/resnet18_round_2000_dir_0.9.pth \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"0,1,2,3,4\" \
    dir_tag=num_clients_per_round

python main.py -cn cifar10 \
    aggregator=unweighted_fedavg \
    aggregator_config.unweighted_fedavg.eta=0.5 \
    atk_config=cifar10_multishot \
    atk_config.poison_start_round=2000 \
    atk_config.poison_end_round=2200 \
    num_clients_per_round=20 \
    checkpoint=checkpoints/CIFAR10_unweighted_fedavg_0.5/resnet18_round_2000_dir_0.9.pth \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"0,1,2,3,4\" \
    dir_tag=num_clients_per_round

python main.py -cn cifar10 \
    aggregator=unweighted_fedavg \
    aggregator_config.unweighted_fedavg.eta=0.5 \
    atk_config=cifar10_multishot \
    atk_config.poison_start_round=2000 \
    atk_config.poison_end_round=2200 \
    num_clients_per_round=30 \
    checkpoint=checkpoints/CIFAR10_unweighted_fedavg_0.5/resnet18_round_2000_dir_0.9.pth \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"0,1,2,3,4\" \
    dir_tag=num_clients_per_round


python main.py -cn cifar10 \
    aggregator=unweighted_fedavg \
    aggregator_config.unweighted_fedavg.eta=0.5 \
    atk_config=cifar10_multishot \
    atk_config.poison_start_round=2000 \
    atk_config.poison_end_round=2200 \
    num_clients_per_round=40 \
    checkpoint=checkpoints/CIFAR10_unweighted_fedavg_0.5/resnet18_round_2000_dir_0.9.pth \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"0,1,2,3,4\" \
    dir_tag=num_clients_per_round

python main.py -cn cifar10 \
    aggregator=unweighted_fedavg \
    aggregator_config.unweighted_fedavg.eta=0.5 \
    atk_config=cifar10_multishot \
    atk_config.poison_start_round=2000 \
    atk_config.poison_end_round=2200 \
    num_clients_per_round=50 \
    checkpoint=checkpoints/CIFAR10_unweighted_fedavg_0.5/resnet18_round_2000_dir_0.9.pth \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"0,1,2,3,4\" \
    dir_tag=num_clients_per_round