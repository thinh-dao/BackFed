######## Anomaly Detection defense against multishot attack ########

############## CIFAR10 ################
# One-line argument using Hydra --multirun 
# For efficiency, you may run attacks in different processes

python main.py -m -cn cifar10 \
    aggregator=flame,deepsight,rflbat,ad_multi_krum,indicator,fldetector \
    no_attack=True \
    checkpoint=checkpoints/CIFAR10_unweighted_fedavg/ResNet18_round_2000_dir_0.5.pth \
    save_checkpoint=True \
    num_rounds=301 \
    "save_checkpoint_rounds=[2300]" \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"0,1,2,3,4\" \
    dir_tag=cifar10_pretrain_anomaly_detection && \
python main.py -m -cn cifar10\
    aggregator=flame,deepsight,rflbat,ad_multi_krum,indicator,fldetector \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern,edge_case,a3fl,iba,distributed \
    atk_config.poison_start_round=2301 \
    atk_config.poison_end_round=2400 \
    checkpoint=2300 \
    num_rounds=100 \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"0,1,2,3,4\" \
    dir_tag=cifar10_anomaly_detection


############## EMNIST ################
python main.py -m -cn emnist \
    aggregator=flame,deepsight,rflbat,ad_multi_krum,indicator,fldetector \
    no_attack=True \
    checkpoint=checkpoints/EMNIST_BYCLASS_unweighted_fedavg/mnistnet_round_1000_dir_0.5.pth \
    save_checkpoint=True \
    num_rounds=301 \
    "save_checkpoint_rounds=[1300]" \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"0,1,2,3,4\" \
    dir_tag=emnist_pretrain_anomaly_detection && \
python main.py -m -cn emnist\
    aggregator=flame,deepsight,rflbat,ad_multi_krum,indicator,fldetector \
    atk_config=emnist_multishot \
    atk_config.data_poison_method=pattern,edge_case,a3fl,iba,distributed \
    atk_config.poison_start_round=1301 \
    atk_config.poison_end_round=1400 \
    checkpoint=1300 \
    num_rounds=100 \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"0,1,2,3,4\" \
    dir_tag=emnist_anomaly_detection
