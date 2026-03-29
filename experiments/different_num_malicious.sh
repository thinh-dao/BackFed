########################################### Different number of malicious clients

# Robust Aggregation with 20% malicious clients
python main.py -m -cn cifar10 \
    aggregator=unweighted_fedavg \
    checkpoint=checkpoints/CIFAR10_unweighted_fedavg_0.5/resnet18_round_1000_dir_0.9.pth \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern,distributed,cerberus,edge_case \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    atk_config.fraction_adversaries=0.2 \
    cuda_visible_devices=\"0,1,2,3,4\" \
    num_rounds=200 \
    dir_tag=cifar10_robust_aggregation_num_malicious_20 &&
python main.py -m -cn cifar10 \
    aggregator=coordinate_median \
    checkpoint=checkpoints/CIFAR10_coordinate_median_0.5/resnet18_round_1000_dir_0.9.pth \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern,distributed,cerberus,edge_case \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    atk_config.fraction_adversaries=0.2 \
    cuda_visible_devices=\"1,0,2,3,4\" \
    num_rounds=200 \
    dir_tag=cifar10_robust_aggregation_num_malicious_20

python main.py -m -cn cifar10 \
    aggregator=krum \
    checkpoint=checkpoints/CIFAR10_krum_0.5/resnet18_round_1000_dir_0.9.pth \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern,distributed,cerberus,edge_case \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    atk_config.fraction_adversaries=0.2 \
    cuda_visible_devices=\"2,1,0,3,4\" \
    num_rounds=200 \
    dir_tag=cifar10_robust_aggregation_num_malicious_20 &&
python main.py -m -cn cifar10 \
    aggregator=weakdp \
    checkpoint=checkpoints/CIFAR10_weakdp_0.5/resnet18_round_1000_dir_0.9.pth \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern,distributed,cerberus,edge_case \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    atk_config.fraction_adversaries=0.2 \
    cuda_visible_devices=\"3,2,1,0,4\" \
    num_rounds=200 \
    dir_tag=cifar10_robust_aggregation_num_malicious_20

python main.py -m -cn cifar10 \
    aggregator=fltrust \
    checkpoint=checkpoints/CIFAR10_fltrust_0.5/resnet18_round_1000_dir_0.9.pth \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern,distributed,cerberus,edge_case \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    atk_config.fraction_adversaries=0.2 \
    cuda_visible_devices=\"4,3,2,1,0\" \
    num_rounds=200 \
    dir_tag=cifar10_robust_aggregation_num_malicious_20



######### FEMNIST ##########
python main.py -m -cn femnist \
    aggregator=unweighted_fedavg \
    checkpoint=checkpoints/FEMNIST_unweighted_fedavg_0.5/mnistnet_round_1000_dir_0.9.pth \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=pattern,distributed,cerberus,edge_case \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    atk_config.fraction_adversaries=0.2 \
    cuda_visible_devices=\"0,1,2,3,4\" \
    num_rounds=200 \
    dir_tag=femnist_robust_aggregation_num_malicious_20 &&
python main.py -m -cn femnist \
    aggregator=coordinate_median \
    checkpoint=checkpoints/FEMNIST_coordinate_median_0.5/mnistnet_round_1000_dir_0.9.pth \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=pattern,distributed,cerberus,edge_case \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    atk_config.fraction_adversaries=0.2 \
    cuda_visible_devices=\"1,0,2,3,4\" \
    num_rounds=200 \
    dir_tag=femnist_robust_aggregation_num_malicious_20

python main.py -m -cn femnist \
    aggregator=krum \
    checkpoint=checkpoints/FEMNIST_krum_0.5/mnistnet_round_1000_dir_0.9.pth \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=pattern,distributed,cerberus,edge_case \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    atk_config.fraction_adversaries=0.2 \
    cuda_visible_devices=\"2,1,0,3,4\" \
    num_rounds=200 \
    dir_tag=femnist_robust_aggregation_num_malicious_20 &&
python main.py -m -cn femnist \
    aggregator=weakdp \
    checkpoint=checkpoints/FEMNIST_weakdp_0.5/mnistnet_round_1000_dir_0.9.pth \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=pattern,distributed,cerberus,edge_case \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    atk_config.fraction_adversaries=0.2 \
    cuda_visible_devices=\"3,2,1,0,4\" \
    num_rounds=200 \
    dir_tag=femnist_robust_aggregation_num_malicious_20

python main.py -m -cn femnist \
    aggregator=fltrust \
    checkpoint=checkpoints/FEMNIST_fltrust_0.5/mnistnet_round_1000_dir_0.9.pth \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=pattern,distributed,cerberus,edge_case \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    atk_config.fraction_adversaries=0.2 \
    cuda_visible_devices=\"4,3,2,1,0\" \
    num_rounds=200 \
    dir_tag=femnist_robust_aggregation_num_malicious_20


# Anomaly Detection with 20% malicious clients
# CIFAR10
python main.py -m -cn cifar10 \
    aggregator=alignins \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern,distributed,cerberus,edge_case \
    atk_config.poison_start_round=2000 \
    atk_config.poison_end_round=2200 \
    atk_config.fraction_adversaries=0.2 \
    cuda_visible_devices=\"0,1,2,3,4\" \
    num_rounds=200 \
    dir_tag=cifar10_anomaly_detection_num_malicious_20 &&
python main.py -m -cn cifar10 \
    aggregator=indicator \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern,distributed,cerberus,edge_case \
    atk_config.poison_start_round=2000 \
    atk_config.poison_end_round=2200 \
    atk_config.fraction_adversaries=0.2 \
    cuda_visible_devices=\"4,3,2,1,0\" \
    num_rounds=200 \
    dir_tag=cifar10_anomaly_detection_num_malicious_20

python main.py -m -cn cifar10 \
    aggregator=ad_multi_krum \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern,distributed,cerberus,edge_case \
    atk_config.poison_start_round=2000 \
    atk_config.poison_end_round=2200 \
    atk_config.fraction_adversaries=0.2 \
    cuda_visible_devices=\"2,1,0,3,4\" \
    num_rounds=200 \
    dir_tag=cifar10_anomaly_detection_num_malicious_20 &&
python main.py -m -cn cifar10 \
    aggregator=feddlad \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern,distributed,cerberus,edge_case \
    atk_config.poison_start_round=2000 \
    atk_config.poison_end_round=2200 \
    atk_config.fraction_adversaries=0.2 \
    cuda_visible_devices=\"3,2,1,0,4\" \
    num_rounds=200 \
    dir_tag=cifar10_anomaly_detection_num_malicious_20

python main.py -m -cn cifar10 \
    aggregator=deepsight \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern,distributed,cerberus,edge_case \
    atk_config.poison_start_round=2000 \
    atk_config.poison_end_round=2200 \
    atk_config.fraction_adversaries=0.2 \
    cuda_visible_devices=\"1,0,2,3,4\" \
    num_rounds=200 \
    dir_tag=cifar10_anomaly_detection_num_malicious_20
    

# FEMNIST
python main.py -m -cn femnist \
    aggregator=alignins \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=pattern,distributed,cerberus,edge_case \
    atk_config.poison_start_round=2000 \
    atk_config.poison_end_round=2200 \
    atk_config.fraction_adversaries=0.2 \
    cuda_visible_devices=\"0,1,2,3,4\" \
    num_rounds=200 \
    dir_tag=femnist_anomaly_detection_num_malicious_20 &&
python main.py -m -cn femnist \
    aggregator=indicator \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=pattern,distributed,cerberus,edge_case \
    atk_config.poison_start_round=2000 \
    atk_config.poison_end_round=2200 \
    atk_config.fraction_adversaries=0.2 \
    cuda_visible_devices=\"4,3,2,1,0\" \
    num_rounds=200 \
    dir_tag=femnist_anomaly_detection_num_malicious_20

python main.py -m -cn femnist \
    aggregator=ad_multi_krum \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=pattern,distributed,cerberus,edge_case \
    atk_config.poison_start_round=2000 \
    atk_config.poison_end_round=2200 \
    atk_config.fraction_adversaries=0.2 \
    cuda_visible_devices=\"2,1,0,3,4\" \
    num_rounds=200 \
    dir_tag=femnist_anomaly_detection_num_malicious_20 &&
python main.py -m -cn femnist \
    aggregator=feddlad \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=pattern,distributed,cerberus,edge_case \
    atk_config.poison_start_round=2000 \
    atk_config.poison_end_round=2200 \
    atk_config.fraction_adversaries=0.2 \
    cuda_visible_devices=\"3,2,1,0,4\" \
    num_rounds=200 \
    dir_tag=femnist_anomaly_detection_num_malicious_20

python main.py -m -cn femnist \
    aggregator=deepsight \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=pattern,distributed,cerberus,edge_case \
    atk_config.poison_start_round=2000 \
    atk_config.poison_end_round=2200 \
    atk_config.fraction_adversaries=0.2 \
    cuda_visible_devices=\"1,0,2,3,4\" \
    num_rounds=200 \
    dir_tag=femnist_anomaly_detection_num_malicious_20