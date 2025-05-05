python main.py -m \
    aggregator=unweighted_fedavg \
    aggregator_config.unweighted_fedavg.eta=0.1,0.3,0.5,0.7,0.9 \
    atk_config.data_poison_method=pattern \
    atk_config.selection_scheme=all-adversary \
    atk_config.poison_start_round=2000 \
    atk_config.poison_end_round=2300 \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"1,2,5,6,7\" \
    dir_tag=server_lr