python main.py -m \
    aggregator=unweighted_fedavg \
    atk_config.data_poison_method=pattern \
    atk_config.selection_scheme=all-adversary,random \
    atk_config.poison_start_round=2000 \
    atk_config.poison_end_round=2300 \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"0,1,3,4,7\"

python main.py -m \
    aggregator=unweighted_fedavg \
    atk_config.data_poison_method=distributed \
    atk_config.selection_scheme=all-adversary,random \
    atk_config.poison_start_round=2000 \
    atk_config.poison_end_round=2300 \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"0,1,3,4,7\"

python main.py -m   \
    aggregator=unweighted_fedavg \
    atk_config.data_poison_method=edge_case \
    atk_config.selection_scheme=all-adversary,random \
    atk_config.poison_start_round=2000 \
    atk_config.poison_end_round=2300 \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"2,3,0,5,6\"

python main.py -m \
    aggregator=unweighted_fedavg \
    atk_config.data_poison_method=a3fl \
    atk_config.selection_scheme=all-adversary,random \
    atk_config.poison_start_round=2000 \
    atk_config.poison_end_round=2300 \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"7,4,1,0,6\"

python main.py -m \
    aggregator=unweighted_fedavg \
    atk_config.data_poison_method=iba \
    atk_config.selection_scheme=all-adversary,random \
    atk_config.poison_start_round=2000 \
    atk_config.poison_end_round=2300 \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"2,3,0,5,6\"


