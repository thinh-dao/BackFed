python main.py \
    aggregator=unweighted_fedavg \
    atk_config=multishot \
    atk_config.model_poison_method=base \
    atk_config.data_poison_method=pattern \
    atk_config.mutual_dataset=True \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=mutual_dataset \
    cuda_visible_devices=\"0,3,1,5,7\" && \
python main.py \
    aggregator=unweighted_fedavg \
    atk_config=multishot \
    atk_config.model_poison_method=neurotoxin \
    atk_config.data_poison_method=pattern \
    atk_config.mutual_dataset=True \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=mutual_dataset \
    cuda_visible_devices=\"0,3,1,5,7\" && \
python main.py \
    aggregator=unweighted_fedavg \
    atk_config=multishot \
    atk_config.model_poison_method=chameleon \
    atk_config.data_poison_method=pattern \
    atk_config.mutual_dataset=True \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=mutual_dataset \
    cuda_visible_devices=\"0,3,1,5,7\" && \



