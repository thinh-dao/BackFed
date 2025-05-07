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
    cuda_visible_devices=\"1,2,3,4,5\" && \
# Model replacement
python main.py \
    aggregator=unweighted_fedavg \
    atk_config=singleshot \
    atk_config.model_poison_method=base \
    atk_config.data_poison_method=pattern \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=300 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=model_replacement \
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
    cuda_visible_devices=\"0,1,2,3,4\" && \
# Model replacement
python main.py \
    aggregator=unweighted_fedavg \
    atk_config=singleshot \
    atk_config.model_poison_method=base \
    atk_config.data_poison_method=distributed \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=300 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=model_replacement \
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
    cuda_visible_devices=\"5,4,3,2,1\" && \
# Model replacement
python main.py \
    aggregator=unweighted_fedavg \
    atk_config=singleshot \
    atk_config.model_poison_method=base \
    atk_config.data_poison_method=edge_case \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=300 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=model_replacement \
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
    cuda_visible_devices=\"5,4,3,2,1\" && \
# Model replacement
python main.py \
    aggregator=unweighted_fedavg \
    atk_config=singleshot \
    atk_config.model_poison_method=base \
    atk_config.data_poison_method=a3fl \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=300 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=model_replacement \
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
    cuda_visible_devices=\"3,4,5,2,1\" && \
# Model replacement
python main.py \
    aggregator=unweighted_fedavg \
    atk_config=singleshot \
    atk_config.model_poison_method=base \
    atk_config.data_poison_method=iba \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=300 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=model_replacement \
    cuda_visible_devices=\"3,4,5,2,1\"