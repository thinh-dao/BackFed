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
    num_rounds=1200 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=cifar10_fed_avg_vs_attacks \
    cuda_visible_devices=\"0,1,2,3,4\"

python main.py -m -cn cifar10 \
    aggregator=unweighted_fedavg \
    atk_config=cifar10_multishot \
    atk_config.model_poison_method=base \
    atk_config.data_poison_method=pattern,distributed,edge_case,a3fl,iba \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=1000 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=cifar10_data_poisoning \
    cuda_visible_devices=\"4,3,2,1,0\"


python main.py -m -cn cifar10 \
    aggregator=unweighted_fedavg \
    atk_config=cifar10_multishot \
    atk_config.model_poison_method=base \
    atk_config.data_poison_method=distributed,a3fl \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=cifar10_data_poisoning \
    cuda_visible_devices=\"4,3,2,1,0\"


python main.py -m -cn femnist \
    aggregator=unweighted_fedavg \
    atk_config=femnist_multishot \
    atk_config.model_poison_method=base \
    atk_config.data_poison_method=distributed,a3fl \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=femnist_data_poisoning \
    cuda_visible_devices=\"3,2,1,0\"



python main.py -m -cn tiny \
    aggregator=unweighted_fedavg \
    atk_config=tiny_multishot \
    atk_config.model_poison_method=base \
    atk_config.data_poison_method=pattern,distributed,iba,a3fl \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=tiny_data_poisoning \
    cuda_visible_devices=\"3,2,1,0\"


python main.py -m -cn tiny \
    aggregator=unweighted_fedavg \
    atk_config=tiny_multishot \
    atk_config.model_poison_method=chameleon \
    atk_config.data_poison_method=pattern \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=tiny_durability_enhanced \
    cuda_visible_devices=\"0,2,3,4,1\"

python main.py -m -cn tiny \
    aggregator=unweighted_fedavg \
    atk_config=tiny_multishot \
    atk_config.model_poison_method=anticipate \
    atk_config.data_poison_method=pattern \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=tiny_durability_enhanced \
    cuda_visible_devices=\"3,2,1,0,4\"



python main.py -m -cn tiny \
    aggregator=unweighted_fedavg \
    atk_config=tiny_multishot \
    atk_config.model_poison_method=base,neurotoxin,anticipate,chameleon \
    atk_config.data_poison_method=pattern \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=tiny_durability_enhanced \
    cuda_visible_devices=\"5,4,3,2,1\" &&
python main.py -m -cn tiny \
    aggregator=unweighted_fedavg \
    atk_config=tiny_multishot \
    atk_config.model_poison_method=base \
    atk_config.data_poison_method=distributed,a3fl,iba \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=tiny_data_poisoning \
    cuda_visible_devices=\"6,4,3,2,1\"


############## EDGE_CASE FIX ###############

python main.py -cn cifar10 \
    aggregator=unweighted_fedavg \
    atk_config.model_poison_method=base \
    atk_config.data_poison_method=edge_case \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=edgecase_datapoison \
    cuda_visible_devices=\"3,2,1,0\" && \
python main.py -cn femnist \
    aggregator=unweighted_fedavg \
    atk_config.model_poison_method=base \
    atk_config.data_poison_method=edge_case \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=edgecase_datapoison \
    cuda_visible_devices=\"3,2,1,0\"


################ EMNIST Multishot ################

python main.py -m -cn emnist \
    aggregator=unweighted_fedavg \
    model=mnistnet \
    checkpoint=1000 \
    atk_config=emnist_multishot \
    atk_config.model_poison_method=base,neurotoxin,anticipate,chameleon \
    atk_config.data_poison_method=pattern,distributed,edge_case,a3fl,iba \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=emnist_fed_avg_vs_attacks \
    cuda_visible_devices=\"0,1,2,3,4\"


############## CIFAR10 Singleshot ################
python main.py -m -cn cifar10\
    aggregator=unweighted_fedavg \
    atk_config=cifar10_multishot \
    atk_config.model_poison_method=base,neurotoxin,anticipate,chameleon \
    atk_config.data_poison_method=pattern,distributed,edge_case,a3fl,iba \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=cifar10_fed_avg_vs_attacks \
    cuda_visible_devices=\"0,1,2,3,4\"


python main.py -m -cn cifar10.yaml \
    aggregator=unweighted_fedavg \
    atk_config=cifar10_multishot \
    atk_config.model_poison_method=base,neurotoxin,anticipate,chameleon \
    atk_config.data_poison_method=pattern \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=1200 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=cifar10_durability_enhanced \
    cuda_visible_devices=\"0,1,2,3,4\"



python main.py -m -cn cifar10.yaml \
    aggregator=unweighted_fedavg \
    atk_config=cifar10_multishot \
    atk_config.model_poison_method=neurotoxin,chameleon \
    atk_config.data_poison_method=pattern \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=1200 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=cifar10_durability_enhanced \
    cuda_visible_devices=\"0,1,2,3,4\" \


python main.py -m -cn femnist \
    aggregator=unweighted_fedavg \
    atk_config=femnist_multishot \
    atk_config.model_poison_method=anticipate,neurotoxin,chameleon,base \
    atk_config.data_poison_method=pattern \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=1200 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=femnist_durability_enhanced \
    cuda_visible_devices=\"0,1,2,3,4\" \
&&

python main.py -m -cn femnist \
    aggregator=unweighted_fedavg \
    atk_config=femnist_multishot \
    atk_config.model_poison_method=anticipate,neurotoxin,chameleon,base \
    atk_config.data_poison_method=pattern \
    checkpoint=1000 \
    save_logging=csv \
    num_rounds=1200 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=femnist_durability_enhanced \
    cuda_visible_devices=\"0,1,2,3,4\" \


python main.py -cn cifar10.yaml \
    aggregator=unweighted_fedavg \
    atk_config=cifar10_multishot \
    atk_config.model_poison_method=base \
    atk_config.data_poison_method=pattern \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=1200 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=test_only \
    cuda_visible_devices=\"0,1,2,3,4\"


python main.py -m -cn cifar10\
    aggregator=unweighted_fedavg \
    atk_config=cifar10_multishot \
    atk_config.model_poison_method=base \
    atk_config.data_poison_method=pattern,distributed,edge_case,a3fl,iba \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=1000 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=cifar10_unweighted_fedavg_attacks \
    cuda_visible_devices=\"0,1,2,3,4\"


python main.py -m -cn femnist \
    aggregator=unweighted_fedavg \
    atk_config=femnist_multishot \
    atk_config.model_poison_method=base,neurotoxin,chameleon,anticipate \
    atk_config.data_poison_method=pattern \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=1000 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=femnist_durability_enhanced \
    cuda_visible_devices=\"0,1,2,3,4\" \
    training_mode=sequential \
    progress_bar=False


python main.py -m -cn femnist \
    aggregator=unweighted_fedavg \
    atk_config=femnist_multishot \
    atk_config.model_poison_method=base \
    atk_config.data_poison_method=edge_case \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=1000 \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"0,1,2,3,4\" \
    training_mode=sequential \
    progress_bar=False \
    save_logging=None 



python main.py -m -cn femnist \
    aggregator=unweighted_fedavg \
    atk_config=femnist_multishot \
    atk_config.model_poison_method=anticipate \
    atk_config.data_poison_method=pattern \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=femnist_durability_enhanced_anticipate \
    cuda_visible_devices=\"0,1,2,3,4\"