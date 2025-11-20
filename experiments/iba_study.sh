python main.py -m -cn cifar10 \
    aggregator=coordinate_median,geometric_median,trimmed_mean \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=a3fl \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=200 \
    checkpoint=1000 \
    dir_tag=cifar10_robust_aggregation_baseline 

python main.py -m -cn cifar10 \
    aggregator=krum,foolsgold,robustlr,norm_clipping,weakdp \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=a3fl \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=200 \
    checkpoint=1000 \
    dir_tag=cifar10_robust_aggregation_baseline 

python main.py -m -cn cifar10 \
    aggregator=fltrust,flare,bulyan \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=a3fl \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=200 \
    checkpoint=1000 \
    dir_tag=cifar10_robust_aggregation_baseline 


python main.py -m -cn cifar10 \
    aggregator=ad_multi_krum,alignins,deepsight \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.poison_ratio=0.5 \
    cuda_visible_devices=\"0,1,2,3,4\" \
    num_rounds=200 \
    dir_tag=cifar10_anomaly_detection_baseline 


python main.py -m -cn cifar10 \
    aggregator=flame,rflbat,multi_metrics,feddlad \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.poison_ratio=0.5 \
    cuda_visible_devices=\"0,1,2,3,4\" \
    num_rounds=200 \
    dir_tag=cifar10_anomaly_detection_baseline 