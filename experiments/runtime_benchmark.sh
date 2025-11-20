python main.py -m -cn cifar10 dir_tag=cifar_runtime_sequential training_mode=sequential no_attack=True num_rounds=5 num_clients_per_round=20,40,60,80,100 
python main.py -m -cn cifar10 dir_tag=cifar_runtime_parallel training_mode=parallel no_attack=True num_rounds=5 num_clients_per_round=20,40,60,80,100 cuda_visible_devices=\"1,2,3,4,5,6\" num_gpus=0.5 num_cpus=1

python main.py -m -cn reddit dir_tag=reddit_runtime_sequential training_mode=sequential no_attack=True num_rounds=5 num_clients_per_round=100,200,300,400,500
python main.py -m -cn reddit dir_tag=reddit_runtime_parallel training_mode=parallel no_attack=True num_rounds=5 num_clients_per_round=100,200,300,400,500 cuda_visible_devices=\"1,2,3,4,5,6\" num_gpus=0.5 num_cpus=1





python main.py -cn cifar10 \
    aggregator=unweighted_fedavg \
    atk_config.data_poison_method=pattern \
    checkpoint=2000 \
    num_rounds=200 \
    num_clients_per_round=20 \
    selection_threshold=0.25 \
    cuda_visible_devices=\"0,1,2,3,4\" \
    dir_tag=cifar10_runtime_selection

python main.py -cn cifar10 \
    aggregator=unweighted_fedavg \
    atk_config.data_poison_method=a3fl \
    checkpoint=2000 \
    num_rounds=200 \
    num_clients_per_round=20 \
    selection_threshold=0.25 \
    cuda_visible_devices=\"3,2,1,4,0\" \
    dir_tag=cifar10_runtime_selection

python main.py -cn cifar10 \
    aggregator=unweighted_fedavg \
    atk_config.model_poison_method=neurotoxin \
    checkpoint=2000 \
    num_rounds=200 \
    num_clients_per_round=20 \
    selection_threshold=0.25 \
    cuda_visible_devices=\"4,3,2,1,0\" \
    dir_tag=cifar10_runtime_selection


####### Timeout

python main.py -cn cifar10 \
    aggregator=unweighted_fedavg \
    atk_config.data_poison_method=pattern \
    checkpoint=2000 \
    num_rounds=200 \
    num_clients_per_round=10 \
    client_timeout=1.2 \
    cuda_visible_devices=\"0,1,2,3,4\" \
    dir_tag=cifar10_runtime_timeout

python main.py -cn cifar10 \
    aggregator=unweighted_fedavg \
    atk_config.data_poison_method=a3fl \
    checkpoint=2000 \
    num_rounds=200 \
    num_clients_per_round=10 \
    client_timeout=1.2 \
    cuda_visible_devices=\"0,1,2,3,4\" \
    dir_tag=cifar10_runtime_timeout

python main.py -cn cifar10 \
    aggregator=unweighted_fedavg \
    atk_config.model_poison_method=neurotoxin \
    checkpoint=2000 \
    num_rounds=200 \
    num_clients_per_round=10 \
    client_timeout=1.2 \
    cuda_visible_devices=\"0,1,2,3,4\" \
    dir_tag=cifar10_runtime_timeout
