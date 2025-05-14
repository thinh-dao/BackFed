
########################## FEMNIST ########################## 
python main.py -m \
    aggregator=flame,deepsight,rflbat,ad_multi_krum,indicator,fldetector \
    no_attack=True \
    num_clients=3383 \
    num_clients_per_round=30 \
    dataset=emnist_byclass \
    test_batch_size=512 \
    num_workers=8 \
    model=mnistnet \
    checkpoint=checkpoints/EMNIST_BYCLASS_unweighted_fedavg/mnistnet_round_1000_dir_0.5.pth \
    save_checkpoint=True \
    num_rounds=301 \
    "save_model_rounds=[1300]" \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"7,6,5,3,2\" \
    dir_tag=pretrain_anomaly_detection_emnist && \
python main.py -m \
    aggregator=flame,deepsight,rflbat,ad_multi_krum,indicator,fldetector \
    num_clients=3383 \
    num_clients_per_round=30 \
    dataset=emnist_byclass \
    test_batch_size=512 \
    num_workers=8 \
    model=mnistnet \
    atk_config=emnist_multishot \
    atk_config.data_poison_method=pattern,edge_case,a3fl,iba,distributed \
    atk_config.poison_start_round=1301 \
    atk_config.poison_end_round=1400 \
    checkpoint=1300 \
    num_rounds=100 \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"7,6,5,3,2\" \
    dir_tag=anomaly_detection_emnist




##### Separate #####
python main.py -m \
    aggregator=flame \
    num_clients=3383 \
    num_clients_per_round=30 \
    dataset=emnist_byclass \
    test_batch_size=512 \
    num_workers=8 \
    model=mnistnet \
    atk_config=emnist_multishot \
    atk_config.data_poison_method=pattern,edge_case,a3fl,iba,distributed \
    atk_config.poison_start_round=1301 \
    atk_config.poison_end_round=1400 \
    checkpoint=1300 \
    num_rounds=100 \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"7,6,5,3,2\" \
    dir_tag=anomaly_detection_emnist

python main.py -m \
    aggregator=deepsight \
    num_clients=3383 \
    num_clients_per_round=30 \
    dataset=emnist_byclass \
    test_batch_size=512 \
    num_workers=8 \
    model=mnistnet \
    atk_config=emnist_multishot \
    atk_config.data_poison_method=pattern,edge_case,a3fl,iba,distributed \
    atk_config.poison_start_round=1301 \
    atk_config.poison_end_round=1400 \
    checkpoint=1300 \
    num_rounds=100 \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"7,6,5,3,2\" \
    dir_tag=anomaly_detection_emnist

python main.py -m \
    aggregator=rflbat \
    num_clients=3383 \
    num_clients_per_round=30 \
    dataset=emnist_byclass \
    test_batch_size=512 \
    num_workers=8 \
    model=mnistnet \
    atk_config=emnist_multishot \
    atk_config.data_poison_method=pattern,edge_case,a3fl,iba,distributed \
    atk_config.poison_start_round=1301 \
    atk_config.poison_end_round=1400 \
    checkpoint=1300 \
    num_rounds=100 \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"7,6,5,3,2\" \
    dir_tag=anomaly_detection_emnist

python main.py -m \
    aggregator=ad_multi_krum \
    num_clients=3383 \
    num_clients_per_round=30 \
    dataset=emnist_byclass \
    test_batch_size=512 \
    num_workers=8 \
    model=mnistnet \
    save_checkpoint=True \
    "save_model_rounds=[1300]" \
    atk_config.data_poison_method=pattern,edge_case,a3fl,iba,distributed \
    atk_config.poison_start_round=1301 \
    atk_config.poison_end_round=1600 \
    checkpoint=1300 \
    num_rounds=300 \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"7,6,5,3,2\" \
    dir_tag=anomaly_detection_emnist


python main.py -m \
    aggregator=indicator \
    num_clients=3383 \
    num_clients_per_round=30 \
    dataset=emnist_byclass \
    test_batch_size=512 \
    num_workers=8 \
    model=mnistnet \
    atk_config.data_poison_method=pattern,edge_case,a3fl,iba,distributed \
    atk_config.poison_start_round=1001 \
    atk_config.poison_end_round=1300 \
    checkpoint=checkpoints/EMNIST_BYCLASS_unweighted_fedavg/mnistnet_round_1000_dir_0.5.pth \
    num_rounds=300 \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"7,6,5,3,2\" \
    dir_tag=anomaly_detection_emnist

python main.py -m \
    aggregator=fldetector \
    no_attack=True \
    num_clients=3383 \
    num_clients_per_round=30 \
    dataset=emnist_byclass \
    test_batch_size=512 \
    num_workers=8 \
    model=mnistnet \
    checkpoint=checkpoints/EMNIST_BYCLASS_unweighted_fedavg/mnistnet_round_1000_dir_0.5.pth \
    save_checkpoint=True \
    num_rounds=301 \
    "save_model_rounds=[1300]" \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"7,6,5,3,2\" \
    dir_tag=pretrain_anomaly_detection_emnist && \
python main.py -m \
    aggregator=indicator \
    num_clients=3383 \
    num_clients_per_round=30 \
    dataset=emnist_byclass \
    test_batch_size=512 \
    num_workers=8 \
    model=mnistnet \
    atk_config=emnist_multishot \
    atk_config.data_poison_method=pattern,edge_case,a3fl,iba,distributed \
    atk_config.poison_start_round=1301 \
    atk_config.poison_end_round=1400 \
    checkpoint=1300 \
    num_rounds=100 \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"7,6,5,3,2\" \
    dir_tag=anomaly_detection_emnist


python main.py -m \
    aggregator=indicator \
    no_attack=True \
    num_clients=3383 \
    num_clients_per_round=30 \
    dataset=emnist_byclass \
    test_batch_size=512 \
    num_workers=8 \
    model=mnistnet \
    checkpoint=checkpoints/EMNIST_BYCLASS_unweighted_fedavg/mnistnet_round_1000_dir_0.5.pth \
    save_checkpoint=True \
    num_rounds=301 \
    "save_model_rounds=[1300]" \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"7,6,5,3,2\" \
    dir_tag=pretrain_anomaly_detection_emnist && \
python main.py -m \
    aggregator=indicator \
    num_clients=3383 \
    num_clients_per_round=30 \
    dataset=emnist_byclass \
    test_batch_size=512 \
    num_workers=8 \
    model=mnistnet \
    atk_config=emnist_multishot \
    atk_config.data_poison_method=pattern,edge_case,a3fl,iba,distributed \
    atk_config.poison_start_round=1301 \
    atk_config.poison_end_round=1400 \
    checkpoint=1300 \
    num_rounds=100 \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"7,6,5,3,2\" \
    dir_tag=anomaly_detection_emnist