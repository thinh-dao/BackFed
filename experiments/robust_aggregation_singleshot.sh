python main.py -m \
    aggregator=robustlr,norm_clipping,trimmed_mean,coordinate_median,geometric_median,krum \
    checkpoint=2300 \
    atk_config=singleshot \
    atk_config.data_poison_method=pattern \
    atk_config.poison_start_round=2301 \
    atk_config.poison_end_round=2600 \
    num_rounds=10 \
    save_logging=csv \
    num_gpus=1 \
    num_cpus=1 \
    cuda_visible_devices=\"0,1,2\" \
    dir_tag=robust_aggregation_singleshot && \

python main.py -m \
    aggregator=foolsgold,robustlr,norm_clipping,trimmed_mean,coordinate_median,geometric_median,krum \
    num_clients=3383 \
    num_clients_per_round=30 \
    dataset=emnist_byclass \
    test_batch_size=512 \
    num_workers=8 \
    model=mnistnet \
    atk_config=emnist_singleshot \
    atk_config.data_poison_method=pattern \
    atk_config.poison_start_round=1301 \
    atk_config.poison_end_round=1301 \
    checkpoint=1300 \
    num_rounds=100 \
    num_rounds=10 \
    save_logging=csv \
    num_gpus=1 \
    num_cpus=1 \
    cuda_visible_devices=\"0,1,2\" \
    dir_tag=robust_aggregation_singleshot_emnist


