python main.py \
    aggregator=flame \
    no_attack=True \
    num_rounds=301 \
    save_checkpoint=True \
    "save_model_rounds=[2100,2200,2300]" \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"1,2,4,5,6\" \
    dir_tag=pretrain_anomaly_detection && \
python main.py -m \
    aggregator=flame \
    atk_config=multishot \
    atk_config.data_poison_method=pattern,edge_case,a3fl,iba,distributed \
    atk_config.poison_start_round=2301 \
    atk_config.poison_end_round=2600 \
    checkpoint=2300 \
    num_rounds=300 \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"2,1,0,5,4\" \
    dir_tag=anomaly_detection_cifar10


python main.py \
    aggregator=deepsight \
    no_attack=True \
    num_rounds=301 \
    save_checkpoint=True \
    "save_model_rounds=[2100,2200,2300]" \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"7,6,5,4,3\" \
    dir_tag=pretrain_anomaly_detection && \
python main.py -m \
    aggregator=deepsight \
    atk_config=multishot \
    atk_config.data_poison_method=pattern,edge_case,a3fl,iba,distributed \
    atk_config.poison_start_round=2301 \
    atk_config.poison_end_round=2600 \
    checkpoint=2300 \
    num_rounds=300 \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"5,6,7,4,3\" \
    dir_tag=anomaly_detection_cifar10

python main.py \
    aggregator=rflbat \
    no_attack=True \
    num_rounds=301 \
    save_checkpoint=True \
    "save_model_rounds=[2100,2200,2300]" \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"3,2,1,0,7\" \
    dir_tag=pretrain_anomaly_detection && \
python main.py -m \
    aggregator=rflbat \
    atk_config=multishot \
    atk_config.data_poison_method=pattern,edge_case,a3fl,iba,distributed \
    atk_config.poison_start_round=2301 \
    atk_config.poison_end_round=2600 \
    checkpoint=2300 \
    num_rounds=300 \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"5,4,3,2,1\" \
    dir_tag=anomaly_detection_cifar10


python main.py \
    aggregator=ad_multi_krum \
    no_attack=True \
    num_rounds=301 \
    save_checkpoint=True \
    "save_model_rounds=[2100,2200,2300]" \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"1,2,4,5,6\" \
    dir_tag=pretrain_anomaly_detection && \
python main.py -m \
    aggregator=ad_multi_krum \
    atk_config=multishot \
    atk_config.data_poison_method=pattern,edge_case,a3fl,iba,distributed \
    atk_config.poison_start_round=2301 \
    atk_config.poison_end_round=2600 \
    checkpoint=2300 \
    num_rounds=300 \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"5,6,7,4,3\" \
    dir_tag=anomaly_detection_cifar10



python main.py \
    aggregator=indicator \
    no_attack=True \
    num_rounds=301 \
    save_checkpoint=True \
    "save_model_rounds=[2100,2200,2300]" \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"1,2,4,5,6\" \
    dir_tag=pretrain_anomaly_detection && \
python main.py -m \
    aggregator=indicator \
    atk_config=multishot \
    atk_config.data_poison_method=pattern,edge_case,a3fl,iba,distributed \
    atk_config.poison_start_round=2301 \
    atk_config.poison_end_round=2600 \
    checkpoint=2300 \
    num_rounds=300 \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"2,1,0,5,4\" \
    dir_tag=anomaly_detection_cifar10


python main.py \
    aggregator=fldetector \
    no_attack=True \
    num_rounds=301 \
    save_checkpoint=True \
    "save_model_rounds=[2100,2200,2300]" \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"1,3,4,5,6\" \
    dir_tag=pretrain_anomaly_detection && \
python main.py -m \
    aggregator=fldetector \
    atk_config=multishot \
    atk_config.data_poison_method=pattern,edge_case,a3fl,iba,distributed \
    atk_config.poison_start_round=2301 \
    atk_config.poison_end_round=2600 \
    checkpoint=2300 \
    num_rounds=300 \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"7,6,5,4,3\" \
    dir_tag=anomaly_detection_cifar10
