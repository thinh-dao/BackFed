######## Baseline: FedAvg against attacks ########

############## CIFAR10 Multishot ################
# One-line argument using Hydra --multirun 
# For efficiency, you may run attacks in different processes

python main.py -m -cn cifar10\
    aggregator=unweighted_fedavg \
    atk_config=cifar10_multishot \
    atk_config.model_poison_method=base,neurotoxin,anticipate,chameleon \
    atk_config.data_poison_method=pattern,distributed,edge_case,a3fl,iba \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    dir_tag=cifar10_fed_avg_vs_attacks \
    cuda_visible_devices=\"0,1,2,3,4\"

python main.py -m -cn cifar10 \
    aggregator=unweighted_fedavg \
    atk_config=cifar10_multishot \
    atk_config.model_poison_method=base \
    atk_config.data_poison_method=pattern,distributed,edge_case,a3fl,iba \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    dir_tag=cifar10_data_poisoning \
    cuda_visible_devices=\"4,3,2,1,0\"

python main.py -m -cn femnist \
    aggregator=unweighted_fedavg \
    atk_config=femnist_multishot \
    atk_config.model_poison_method=base \
    atk_config.data_poison_method=pattern,distributed,edge_case,a3fl,iba \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    dir_tag=femnist_data_poisoning \
    cuda_visible_devices=\"4,3,2,1,0\"

python main.py -m -cn tiny \
    aggregator=unweighted_fedavg \
    atk_config=tiny_multishot \
    atk_config.model_poison_method=base \
    atk_config.data_poison_method=pattern,distributed,iba,a3fl \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    dir_tag=tiny_data_poisoning \
    cuda_visible_devices=\"3,2,1,0\"

##### Durability Enhanced
python main.py -m -cn cifar10 \
    aggregator=unweighted_fedavg \
    atk_config=cifar10_multishot \
    atk_config.model_poison_method=base,neurotoxin,chameleon,anticipate \
    atk_config.data_poison_method=pattern\
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    dir_tag=cifar10_data_poisoning \
    cuda_visible_devices=\"4,3,2,1,0\"

python main.py -m -cn femnist \
    aggregator=unweighted_fedavg \
    atk_config=femnist_multishot \
    atk_config.model_poison_method=base,neurotoxin,chameleon,anticipate \
    atk_config.data_poison_method=pattern\
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    dir_tag=femnist_data_poisoning \
    cuda_visible_devices=\"4,3,2,1,0\"

python main.py -m -cn tiny \
    aggregator=unweighted_fedavg \
    atk_config=tiny_multishot \
    atk_config.model_poison_method=base,neurotoxin,chameleon,anticipate \
    atk_config.data_poison_method=pattern\
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    dir_tag=tiny_data_poisoning \
    cuda_visible_devices=\"4,3,2,1,0\"


#### DBA only
python main.py -cn cifar10 \
    aggregator=unweighted_fedavg \
    atk_config=cifar10_multishot \
    atk_config.model_poison_method=base \
    atk_config.data_poison_method=distributed \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    dir_tag=cifar10_data_poisoning \
    cuda_visible_devices=\"4,3,2,1,0\" &&
python main.py -m -cn femnist \
    aggregator=unweighted_fedavg \
    atk_config=femnist_multishot \
    atk_config.model_poison_method=base \
    atk_config.data_poison_method=distributed \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    dir_tag=femnist_data_poisoning \
    cuda_visible_devices=\"4,3,2,1,0\" &&
python main.py -m -cn tiny \
    aggregator=unweighted_fedavg \
    atk_config=tiny_multishot \
    atk_config.model_poison_method=base \
    atk_config.data_poison_method=distributed \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    dir_tag=tiny_data_poisoning \
    cuda_visible_devices=\"3,2,1,0\"


python main.py -m -cn tiny \
    aggregator=unweighted_fedavg \
    atk_config=tiny_multishot \
    atk_config.model_poison_method=base \
    atk_config.data_poison_method=distributed \
    atk_config.adversary_selection=fixed \
    atk_config.selection_scheme=single-adversary \
    atk_config.scale_poison=True \
    atk_config.scale_factor=100 \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    dir_tag=tiny_data_poisoning \
    cuda_visible_devices=\"4,3,2,1,0\"