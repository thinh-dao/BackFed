# Scaled + Continuous attack
python main.py -cn cifar10 \
    aggregator=unweighted_fedavg \
    aggregator_config.unweighted_fedavg.eta=0.1 \
    atk_config=cifar10_multishot \
    atk_config.model_poison_method=base \
    atk_config.data_poison_method=pattern \
    atk_config.poison_start_round=2001 \
    atk_config.poison_end_round=2100 \
    atk_config.adversary_selection=fixed \
    atk_config.selection_scheme=single-adversary \
    atk_config.scale_poison=True \
    atk_config.scale_factor=100 \
    checkpoint=checkpoints/CIFAR10_unweighted_fedavg_0.5/resnet18_round_2000_dir_0.9.pth \
    num_rounds=300 \
    save_logging=csv \
    dir_tag=multisht_dba_compare \
    cuda_visible_devices=\"5,4,3,2,1,0\" &&
python main.py -cn cifar10 \
    aggregator=unweighted_fedavg \
    aggregator_config.unweighted_fedavg.eta=0.1 \
    atk_config=cifar10_multishot \
    atk_config.model_poison_method=base \
    atk_config.data_poison_method=distributed \
    atk_config.poison_start_round=2001 \
    atk_config.poison_end_round=2100 \
    atk_config.adversary_selection=fixed \
    atk_config.selection_scheme=single-adversary \
    atk_config.scale_poison=True \
    atk_config.scale_factor=100 \
    checkpoint=checkpoints/CIFAR10_unweighted_fedavg_0.5/resnet18_round_2000_dir_0.9.pth \
    num_rounds=300 \
    num_gpus=0.5 \
    num_cpus=1 \
    save_logging=csv \
    dir_tag=multisht_dba_compare \
    cuda_visible_devices=\"5,4,3,2,1,0\"

# Continuous
python main.py -cn cifar10 \
    aggregator=unweighted_fedavg \
    aggregator_config.unweighted_fedavg.eta=0.1 \
    atk_config=cifar10_multishot \
    atk_config.model_poison_method=base \
    atk_config.data_poison_method=pattern \
    atk_config.poison_start_round=2001 \
    atk_config.poison_end_round=2100 \
    atk_config.adversary_selection=fixed \
    atk_config.selection_scheme=single-adversary \
    checkpoint=checkpoints/CIFAR10_unweighted_fedavg_0.5/resnet18_round_2000_dir_0.9.pth \
    num_rounds=300 \
    num_gpus=0.5 \
    num_cpus=1 \
    save_logging=csv \
    dir_tag=multisht_dba_compare \
    cuda_visible_devices=\"5,4,3,2,1,0\" &&
python main.py -cn cifar10 \
    aggregator=unweighted_fedavg \
    aggregator_config.unweighted_fedavg.eta=0.1 \
    atk_config=cifar10_multishot \
    atk_config.model_poison_method=base \
    atk_config.data_poison_method=distributed \
    atk_config.poison_start_round=2001 \
    atk_config.poison_end_round=2100 \
    atk_config.adversary_selection=fixed \
    atk_config.selection_scheme=single-adversary \
    checkpoint=checkpoints/CIFAR10_unweighted_fedavg_0.5/resnet18_round_2000_dir_0.9.pth \
    num_rounds=300 \
    num_gpus=0.5 \
    num_cpus=1 \
    save_logging=csv \
    dir_tag=multisht_dba_compare \
    cuda_visible_devices=\"5,4,3,2,1,0\"

# Scaled
python main.py -cn cifar10 \
    aggregator=unweighted_fedavg \
    aggregator_config.unweighted_fedavg.eta=0.1 \
    atk_config=cifar10_multishot \
    atk_config.model_poison_method=base \
    atk_config.data_poison_method=pattern \
    atk_config.poison_start_round=2001 \
    atk_config.poison_end_round=2100 \
    atk_config.scale_poison=True \
    atk_config.scale_factor=100 \
    checkpoint=checkpoints/CIFAR10_unweighted_fedavg_0.5/resnet18_round_2000_dir_0.9.pth \
    num_rounds=300 \
    num_gpus=0.5 \
    num_cpus=1 \
    save_logging=csv \
    dir_tag=multisht_dba_compare \
    cuda_visible_devices=\"5,4,3,2,1,0\" &&
python main.py -cn cifar10 \
    aggregator=unweighted_fedavg \
    aggregator_config.unweighted_fedavg.eta=0.1 \
    atk_config=cifar10_multishot \
    atk_config.model_poison_method=base \
    atk_config.data_poison_method=distributed \
    atk_config.poison_start_round=2001 \
    atk_config.poison_end_round=2100 \
    atk_config.scale_poison=True \
    atk_config.scale_factor=100 \
    checkpoint=checkpoints/CIFAR10_unweighted_fedavg_0.5/resnet18_round_2000_dir_0.9.pth \
    num_rounds=300 \
    num_gpus=0.5 \
    num_cpus=1 \
    save_logging=csv \
    dir_tag=multisht_dba_compare \
    cuda_visible_devices=\"5,4,3,2,1,0\"



##### Tiny-ImageNet


# Scaled + Continuous attack
python main.py -cn tiny \
    aggregator=unweighted_fedavg \
    aggregator_config.unweighted_fedavg.eta=0.1 \
    atk_config=tiny_multishot \
    atk_config.model_poison_method=base \
    atk_config.data_poison_method=pattern \
    atk_config.poison_start_round=2001 \
    atk_config.poison_end_round=2100 \
    atk_config.adversary_selection=fixed \
    atk_config.selection_scheme=single-adversary \
    atk_config.scale_poison=True \
    atk_config.scale_factor=100 \
    checkpoint=checkpoints/TINYIMAGENET_unweighted_fedavg_0.5/vgg11_round_2000_dir_0.5.pth \
    num_rounds=200 \
    save_logging=csv \
    dir_tag=multisht_dba_compare \
    cuda_visible_devices=\"2,3,4,1\" &&
python main.py -cn tiny \
    aggregator=unweighted_fedavg \
    aggregator_config.unweighted_fedavg.eta=0.1 \
    atk_config=tiny_multishot \
    atk_config.model_poison_method=base \
    atk_config.data_poison_method=distributed \
    atk_config.poison_start_round=2001 \
    atk_config.poison_end_round=2100 \
    atk_config.adversary_selection=fixed \
    atk_config.selection_scheme=single-adversary \
    atk_config.scale_poison=True \
    atk_config.scale_factor=100 \
    checkpoint=checkpoints/TINYIMAGENET_unweighted_fedavg_0.5/vgg11_round_2000_dir_0.5.pth \
    num_rounds=200 \
    num_gpus=0.5 \
    num_cpus=1 \
    save_logging=csv \
    dir_tag=multisht_dba_compare \
    cuda_visible_devices=\"2,3,4,1\"

# Continuous
python main.py -cn tiny \
    aggregator=unweighted_fedavg \
    aggregator_config.unweighted_fedavg.eta=0.1 \
    atk_config=tiny_multishot \
    atk_config.model_poison_method=base \
    atk_config.data_poison_method=pattern \
    atk_config.poison_start_round=2001 \
    atk_config.poison_end_round=2100 \
    atk_config.adversary_selection=fixed \
    atk_config.selection_scheme=single-adversary \
    checkpoint=checkpoints/TINYIMAGENET_unweighted_fedavg_0.5/vgg11_round_2000_dir_0.5.pth \
    num_rounds=200 \
    num_gpus=0.5 \
    num_cpus=1 \
    save_logging=csv \
    dir_tag=multisht_dba_compare \
    cuda_visible_devices=\"3,2,1,4\" &&
python main.py -cn tiny \
    aggregator=unweighted_fedavg \
    aggregator_config.unweighted_fedavg.eta=0.1 \
    atk_config=tiny_multishot \
    atk_config.model_poison_method=base \
    atk_config.data_poison_method=distributed \
    atk_config.poison_start_round=2001 \
    atk_config.poison_end_round=2100 \
    atk_config.adversary_selection=fixed \
    atk_config.selection_scheme=single-adversary \
    checkpoint=checkpoints/TINYIMAGENET_unweighted_fedavg_0.5/vgg11_round_2000_dir_0.5.pth \
    num_rounds=200 \
    num_gpus=0.5 \
    num_cpus=1 \
    save_logging=csv \
    dir_tag=multisht_dba_compare \
    cuda_visible_devices=\"3,2,1,4\"

# Scaled
python main.py -cn tiny \
    aggregator=unweighted_fedavg \
    aggregator_config.unweighted_fedavg.eta=0.1 \
    atk_config=tiny_multishot \
    atk_config.model_poison_method=base \
    atk_config.data_poison_method=pattern \
    atk_config.poison_start_round=2001 \
    atk_config.poison_end_round=2100 \
    atk_config.scale_poison=True \
    atk_config.scale_factor=100 \
    checkpoint=checkpoints/TINYIMAGENET_unweighted_fedavg_0.5/vgg11_round_2000_dir_0.5.pth \
    num_rounds=200 \
    num_gpus=0.5 \
    num_cpus=1 \
    save_logging=csv \
    dir_tag=multisht_dba_compare \
    cuda_visible_devices=\"3,2,1,4\" &&
python main.py -cn tiny \
    aggregator=unweighted_fedavg \
    aggregator_config.unweighted_fedavg.eta=0.1 \
    atk_config=tiny_multishot \
    atk_config.model_poison_method=base \
    atk_config.data_poison_method=distributed \
    atk_config.poison_start_round=2001 \
    atk_config.poison_end_round=2100 \
    atk_config.scale_poison=True \
    atk_config.scale_factor=100 \
    checkpoint=checkpoints/TINYIMAGENET_unweighted_fedavg_0.5/vgg11_round_2000_dir_0.5.pth \
    num_rounds=200 \
    num_gpus=0.5 \
    num_cpus=1 \
    save_logging=csv \
    dir_tag=multisht_dba_compare \
    cuda_visible_devices=\"2,3,4,1\"