python main.py -m -cn cifar10 \
    aggregator=unweighted_fedavg \
    atk_config.model_poison_method=base \
    atk_config.data_poison_config.pattern.location=top_left \
    atk_config.data_poison_method=distributed,pattern \
    atk_config.adversary_selection=fixed \
    atk_config.selection_scheme=single-adversary \
    atk_config.poison_end_round=2200 \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=dba_multishot \
    cuda_visible_devices=\"3,2,1,0\"


python main.py -cn cifar10 \
    aggregator=unweighted_fedavg \
    atk_config.model_poison_method=base \
    atk_config.data_poison_method=distributed \
    atk_config.poison_end_round=2200 \
    atk_config.scale_poison=True \
    atk_config.scale_factor=5 \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=dba_multishot \
    cuda_visible_devices=\"3,2,1,0\"