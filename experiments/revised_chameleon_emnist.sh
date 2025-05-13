
python main.py -m \
    aggregator=unweighted_fedavg \
    num_clients=3383 \
    num_clients_per_round=30 \
    dataset=emnist_byclass \
    test_batch_size=5000 \
    num_workers=8 \
    model=mnistnet \
    checkpoint=1000 \
    atk_config=emnist_multishot \
    atk_config.model_poison_method=chameleon \
    atk_config.data_poison_method=pattern,distributed,edge_case,a3fl,iba \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=chameleon_emnist_no_mutual \
    cuda_visible_devices=\"3,5,6,7,2\"

# revised with mutual
python main.py -m \
    aggregator=unweighted_fedavg \
    num_clients=3383 \
    num_clients_per_round=30 \
    dataset=emnist_byclass \
    test_batch_size=5000 \
    num_workers=8 \
    model=mnistnet \
    checkpoint=1000 \
    atk_config=emnist_multishot \
    atk_config.model_poison_method=chameleon \
    atk_config.data_poison_method=pattern,distributed,edge_case,a3fl,iba \
    atk_config.mutual_dataset=True \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=chameleon_emnist_mutual \
    cuda_visible_devices=\"3,5,6,7,2\"





python main.py -m \
    aggregator=unweighted_fedavg \
    num_clients=3383 \
    num_clients_per_round=30 \
    dataset=emnist_byclass \
    test_batch_size=5000 \
    num_workers=8 \
    model=mnistnet \
    checkpoint=1000 \
    atk_config=emnist_singleshot \
    atk_config.model_poison_method=chameleon \
    atk_config.data_poison_method=pattern,distributed,edge_case,a3fl,iba \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=chameleon_emnist_no_mutual \
    cuda_visible_devices=\"2,3\"



# revised with mutual
python main.py -m \
    aggregator=unweighted_fedavg \
    num_clients=3383 \
    num_clients_per_round=30 \
    dataset=emnist_byclass \
    test_batch_size=5000 \
    num_workers=8 \
    model=mnistnet \
    checkpoint=1000 \
    atk_config=emnist_singleshot \
    atk_config.model_poison_method=chameleon \
    atk_config.data_poison_method=pattern,distributed,edge_case,a3fl,iba \
    atk_config.mutual_dataset=True \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=chameleon_emnist_mutual \
    cuda_visible_devices=\"2,3\"

