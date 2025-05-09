python main.py -m \
    aggregator=unweighted_fedavg \
    atk_config=multishot \
    atk_config.model_poison_method=base,neurotoxin,chameleon \
    atk_config.data_poison_method=pattern \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=fed_avg_vs_attacks \
    cuda_visible_devices=\"1,2,3,4,5\"



python main.py -m \
    aggregator=unweighted_fedavg \
    atk_config=multishot \
    atk_config.model_poison_method=base,neurotoxin,chameleon \
    atk_config.data_poison_method=distributed \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=fed_avg_vs_attacks \
    cuda_visible_devices=\"0,1,2,3,4\"




python main.py -m \
    aggregator=unweighted_fedavg \
    atk_config=multishot \
    atk_config.model_poison_method=base,neurotoxin,chameleon \
    atk_config.data_poison_method=edge_case \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=fed_avg_vs_attacks \
    cuda_visible_devices=\"5,4,3,2,1\"




python main.py -m \
    aggregator=unweighted_fedavg \
    atk_config=multishot \
    atk_config.model_poison_method=base,neurotoxin,chameleon \
    atk_config.data_poison_method=a3fl \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=fed_avg_vs_attacks \
    cuda_visible_devices=\"5,4,3,2,1\"




python main.py -m \
    aggregator=unweighted_fedavg \
    atk_config=multishot \
    atk_config.model_poison_method=base,neurotoxin,chameleon \
    atk_config.data_poison_method=iba \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=fed_avg_vs_attacks \
    cuda_visible_devices=\"3,4,5,2,1\"



################ FEMNIST ################
# num_clients=3383 \
# num_clients_per_round=30 \
# dataset=emnist_byclass \
# test_batch_size=5000 \
# num_workers=8 \
# model=mnistnet \
# checkpoint=1000 \
# atk_config=emnist_multishot \

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
    atk_config.model_poison_method=base,neurotoxin,chameleon \
    atk_config.data_poison_method=pattern \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=fed_avg_vs_attacks_emnist \
    cuda_visible_devices=\"1,2,3,4,5\"



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
    atk_config.model_poison_method=base,neurotoxin,chameleon \
    atk_config.data_poison_method=distributed \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=fed_avg_vs_attacks_emnist \
    cuda_visible_devices=\"6,5,4,3,2\"




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
    atk_config.model_poison_method=base,neurotoxin,chameleon \
    atk_config.data_poison_method=edge_case \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=fed_avg_vs_attacks_emnist \
    cuda_visible_devices=\"6,5,4,3,2\"




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
    atk_config.model_poison_method=base,neurotoxin,chameleon \
    atk_config.data_poison_method=a3fl \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=fed_avg_vs_attacks_emnist \
    cuda_visible_devices=\"6,5,4,3,2\"


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
    atk_config.model_poison_method=base,neurotoxin,chameleon \
    atk_config.data_poison_method=iba \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=fed_avg_vs_attacks_emnist \
    cuda_visible_devices=\"7,6,5,4,3\"