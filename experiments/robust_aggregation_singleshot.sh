python main.py -m \
    aggregator=foolsgold \
    checkpoint=2300 \
    atk_config=singleshot \
    atk_config.data_poison_method=pattern \
    atk_config.poison_start_round=2301 \
    atk_config.poison_end_round=2600 \
    num_rounds=300 \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"6,5,4,3,2\" \
    dir_tag=robust_aggregation_singleshot

python main.py -m \
    aggregator=robustlr \
    checkpoint=2300 \
    atk_config=singleshot \
    atk_config.data_poison_method=pattern \
    atk_config.poison_start_round=2301 \
    atk_config.poison_end_round=2600 \
    num_rounds=300 \
    save_logging=csv \
    num_gpus=0.3 \
    num_cpus=1 \
    cuda_visible_devices=\"6,5,4\" \
    dir_tag=robust_aggregation_singleshot

python main.py -m \
    aggregator=norm_clipping \
    checkpoint=2300 \
    atk_config=singleshot \
    atk_config.data_poison_method=pattern \
    atk_config.poison_start_round=2301 \
    atk_config.poison_end_round=2600 \
    num_rounds=300 \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"6,5,4,3,2\" \
    dir_tag=robust_aggregation_singleshot

python main.py -m \
    aggregator=norm_clipping \
    checkpoint=2300 \
    atk_config=singleshot \
    atk_config.data_poison_method=pattern \
    atk_config.poison_start_round=2301 \
    atk_config.poison_end_round=2600 \
    num_rounds=300 \
    save_logging=csv \
    num_gpus=0.3 \
    num_cpus=1 \
    cuda_visible_devices=\"6,5,4\" \
    dir_tag=robust_aggregation_singleshot




python main.py \
    aggregator=foolsgold \
    checkpoint=2300 \
    atk_config=multishot \
    atk_config.data_poison_method=pattern \
    atk_config.poison_start_round=2301 \
    atk_config.poison_end_round=2600 \
    atk_config.selection_scheme=multi-adversary \
    num_rounds=300 \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"6,5,4,3,2\" \
    atk_config.mutual_dataset=True \
    dir_tag=robust_aggregation_singleshot