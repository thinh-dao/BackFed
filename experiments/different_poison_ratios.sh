python main.py -m -cn cifar10 \
    aggregator=alignins \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern,cerberus,distributed,edge_case \
    atk_config.poison_start_round=2000 \
    atk_config.poison_end_round=2200 \
    atk_config.poison_ratio=0.25,0.75,1.0 \
    cuda_visible_devices=\"0,1,2,3,4\" \
    num_rounds=100 \
    dir_tag=cifar10_poison_ratios

python main.py -m -cn cifar10 \
    aggregator=deepsight \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern,cerberus,distributed,edge_case \
    atk_config.poison_start_round=2000 \
    atk_config.poison_end_round=2200 \
    atk_config.poison_ratio=0.25,0.75,1.0 \
    cuda_visible_devices=\"4,3,2,1,0\" \
    num_rounds=100 \
    dir_tag=cifar10_poison_ratios

python main.py -m -cn cifar10 \
    aggregator=feddlad \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern,cerberus,distributed,edge_case \
    atk_config.poison_start_round=2000 \
    atk_config.poison_end_round=2200 \
    atk_config.poison_ratio=0.25,0.75,1.0 \
    cuda_visible_devices=\"2,1,0,3,4\" \
    num_rounds=100 \
    dir_tag=cifar10_poison_ratios

python main.py -m -cn cifar10 \
    aggregator=indicator \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern,cerberus,distributed,edge_case \
    atk_config.poison_start_round=2000 \
    atk_config.poison_end_round=2200 \
    atk_config.poison_ratio=0.25,0.75,1.0 \
    cuda_visible_devices=\"3,2,1,0,4\" \
    num_rounds=100 \
    dir_tag=cifar10_poison_ratios



python main.py -m -cn cifar10 \
    aggregator=unweighted_fedavg \
    checkpoint=checkpoints/CIFAR10_unweighted_fedavg_0.5/resnet18_round_1000_dir_0.9.pth \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern,distributed,cerberus,edge_case \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1100 \
    atk_config.poison_ratio=0.25,0.75,1.0 \
    cuda_visible_devices=\"0,1,2,3\" \
    num_rounds=100 \
    dir_tag=cifar10_poison_ratios

python main.py -m -cn cifar10 \
    aggregator=krum \
    checkpoint=checkpoints/CIFAR10_krum_0.5/resnet18_round_1000_dir_0.9.pth \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern,distributed,cerberus,edge_case \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1100 \
    atk_config.poison_ratio=0.25,0.75,1.0 \
    cuda_visible_devices=\"3,2,1,0\" \
    num_rounds=100 \
    dir_tag=cifar10_poison_ratios

python main.py -m -cn cifar10 \
    aggregator=fltrust \
    checkpoint=checkpoints/CIFAR10_fltrust_0.5/resnet18_round_1000_dir_0.9.pth \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern,distributed,cerberus,edge_case \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1100 \
    atk_config.poison_ratio=0.25,0.75,1.0 \
    cuda_visible_devices=\"4,0,1,2\" \
    num_rounds=100 \
    dir_tag=cifar10_poison_ratios

python main.py -m -cn cifar10 \
    aggregator=norm_clipping \
    checkpoint=checkpoints/CIFAR10_norm_clipping_0.5/resnet18_round_1000_dir_0.9.pth \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern,distributed,cerberus,edge_case \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1100 \
    atk_config.poison_ratio=0.25,0.75,1.0 \
    cuda_visible_devices=\"2,3,1,4\" \
    num_rounds=100 \
    dir_tag=cifar10_poison_ratios