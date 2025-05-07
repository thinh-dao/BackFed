python main.py -m \
    aggregator=unweighted_fedavg \
    atk_config=multishot \
    atk_config.mutual_dataset=True \
    atk_config.model_poison_method=base,neurotoxin,chameleon \
    atk_config.data_poison_method=pattern \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=mutual_dataset \
    cuda_visible_devices=\"6,4,3,2,1\"




python main.py -m \
    aggregator=unweighted_fedavg \
    atk_config=multishot \
    atk_config.mutual_dataset=True \
    atk_config.model_poison_method=base,neurotoxin,chameleon \
    atk_config.data_poison_method=distributed \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.4 \
    num_cpus=1 \
    dir_tag=mutual_dataset \
    cuda_visible_devices=\"6,4,3,2\" 




python main.py -m \
    aggregator=unweighted_fedavg \
    atk_config=multishot \
    atk_config.mutual_dataset=True \
    atk_config.model_poison_method=base,neurotoxin,chameleon \
    atk_config.data_poison_method=edge_case \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.4 \
    num_cpus=1 \
    dir_tag=mutual_dataset \
    cuda_visible_devices=\"6,4,3,2\"




python main.py -m \
    aggregator=unweighted_fedavg \
    atk_config=multishot \
    atk_config.mutual_dataset=True \
    atk_config.model_poison_method=base,neurotoxin,chameleon \
    atk_config.data_poison_method=a3fl \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.4 \
    num_cpus=1 \
    dir_tag=mutual_dataset \
    cuda_visible_devices=\"6,4,3,2\"



python main.py -m \
    aggregator=unweighted_fedavg \
    atk_config=multishot \
    atk_config.mutual_dataset=True \
    atk_config.model_poison_method=base,neurotoxin,chameleon \
    atk_config.data_poison_method=iba \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.4 \
    num_cpus=1 \
    dir_tag=mutual_dataset \
    cuda_visible_devices=\"6,4,3,2\"
