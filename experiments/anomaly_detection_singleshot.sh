python main.py -m \
    aggregator=flame,deepsight,rflbat,ad_multi_krum,indicator,fldetector \
    atk_config=singleshot \
    atk_config.data_poison_method=pattern \
    atk_config.poison_start_round=2301 \
    atk_config.poison_end_round=2301 \
    checkpoint=2300 \
    num_rounds=1 \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"0,1,2,4,5\" \
    dir_tag=anomaly_detection_cifar10 && \
python main.py -m \
    aggregator=flame,deepsight,rflbat,ad_multi_krum,indicator,fldetector \
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
    num_rounds=1 \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"0,1,2,4,5\" \
    dir_tag=robust_aggregation_singleshot




