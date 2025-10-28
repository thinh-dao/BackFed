python main.py -m -cn cifar10 \
    aggregator=unweighted_fedavg,coordinate_median,geometric_median,trimmed_mean,krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust \
    atk_config=cifar10_multishot \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    atk_config.data_poison_method=pattern \
    cuda_visible_devices=\"0,1,2,3\" \
    num_rounds=200 \
    dir_tag=cifar10_robust_aggregation

python main.py -m -cn femnist \
    aggregator=unweighted_fedavg,coordinate_median,geometric_median,trimmed_mean,krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust \
    atk_config=femnist_multishot \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    atk_config.data_poison_method=pattern \
    cuda_visible_devices=\"0,1,2,3\" \
    num_rounds=200 \
    dir_tag=femnist_robust_aggregation