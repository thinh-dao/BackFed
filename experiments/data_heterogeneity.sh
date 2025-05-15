python main.py -m \
    aggregator=robustlr,norm_clipping,trimmed_mean,coordinate_median,geometric_median,krum \
    checkpoint=2300 \
    partitioner=uniform \
    atk_config=mult_shot \
    atk_config.data_poison_method=pattern \
    atk_config.poison_start_round=2301 \
    atk_config.poison_end_round=2600 \
    num_rounds=300 \
    save_logging=csv \
    num_gpus=1 \
    num_cpus=1 \
    cuda_visible_devices=\"0,1,2,4,5\" \
    dir_tag=robust_aggregation_multishot_uniform 

python main.py -m \
    aggregator=robustlr,norm_clipping,trimmed_mean,coordinate_median,geometric_median,krum \
    checkpoint=2300 \
    partitioner=uniform \
    atk_config.mutual_dataset=True \
    atk_config=mult_shot \
    atk_config.data_poison_method=pattern \
    atk_config.poison_start_round=2301 \
    atk_config.poison_end_round=2600 \
    num_rounds=300 \
    save_logging=csv \
    num_gpus=1 \
    num_cpus=1 \
    cuda_visible_devices=\"0,1,2,4,5\" \
    dir_tag=robust_aggregation_multishot_uniform_mutual 