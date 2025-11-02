
#### Baseline
python main.py -m -cn cifar10 \
    aggregator=multi_krum \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.poison_ratio=0.5 \
    cuda_visible_devices=\"0,1,2,3\" \
    num_rounds=100 \
    dir_tag=cifar10_mkrum_baseline &&
python main.py -m -cn cifar10 \
    aggregator=multi_krum \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=100 \
    dir_tag=cifar10_mkrum_baseline &&
python main.py -m -cn cifar10 \
    aggregator=multi_krum \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=a3fl \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=100 \
    dir_tag=cifar10_mkrum_baseline &&
python main.py -m -cn cifar10 \
    aggregator=multi_krum \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=cerberus \
    atk_config.model_poison_method=cerberus \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=100 \
    dir_tag=cifar10_mkrum_baseline


#### PGD attack
python main.py -m -cn cifar10 \
    aggregator=multi_krum \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=3 \
    atk_config.poison_ratio=0.5 \
    cuda_visible_devices=\"0,1,2,3\" \
    num_rounds=100 \
    dir_tag=cifar10_mkrum_pgd &&
python main.py -m -cn cifar10 \
    aggregator=multi_krum \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=3 \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=100 \
    dir_tag=cifar10_mkrum_pgd &&
python main.py -m -cn cifar10 \
    aggregator=multi_krum \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=a3fl \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=3 \
    cuda_visible_devices=\"5,4,3,2,1,0\" \
    num_rounds=100 \
    dir_tag=cifar10_mkrum_pgd &&
python main.py -m -cn cifar10 \
    aggregator=multi_krum \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=cerberus \
    atk_config.model_poison_method=cerberus \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=3 \
    cuda_visible_devices=\"5,4,3,2,1,0\" \
    num_rounds=100 \
    dir_tag=cifar10_mkrum_pgd


#### Model Replacement
python main.py -m -cn cifar10 \
    aggregator=multi_krum \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    cuda_visible_devices=\"3,2,1,5,0\" \
    num_rounds=100 \
    dir_tag=cifar10_mkrum_modelreplace && 
python main.py -m -cn cifar10 \
    aggregator=multi_krum \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    cuda_visible_devices=\"3,2,1,5,0\" \
    num_rounds=100 \
    dir_tag=cifar10_mkrum_modelreplace &&
python main.py -m -cn cifar10 \
    aggregator=multi_krum \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=distributed \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    cuda_visible_devices=\"3,2,1,5,0\" \
    num_rounds=100 \
    dir_tag=cifar10_mkrum_modelreplace





###################
# FEMNIST
python main.py -m -cn femnist \
    aggregator=multi_krum \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.poison_ratio=0.5 \
    cuda_visible_devices=\"5,3,2,1,0\" \
    num_rounds=100 \
    dir_tag=femnist_mkrum_baseline &&
python main.py -m -cn femnist \
    aggregator=multi_krum \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=pattern \
    cuda_visible_devices=\"5,4,3,2,1\" \
    num_rounds=100 \
    dir_tag=femnist_mkrum_baseline &&
python main.py -m -cn femnist \
    aggregator=multi_krum \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=a3fl \
    cuda_visible_devices=\"2,3,4,1,0\" \
    num_rounds=100 \
    dir_tag=femnist_mkrum_baseline &&
python main.py -m -cn femnist \
    aggregator=multi_krum \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=cerberus \
    atk_config.model_poison_method=cerberus \
    cuda_visible_devices=\"3,2,1,4,0\" \
    num_rounds=100 \
    dir_tag=femnist_mkrum_baseline


#### PGD attack
python main.py -m -cn femnist \
    aggregator=multi_krum \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=3 \
    atk_config.poison_ratio=0.5 \
    cuda_visible_devices=\"0,1,2,3,4\" \
    num_rounds=100 \
    dir_tag=femnist_mkrum_pgd &&
python main.py -m -cn femnist \
    aggregator=multi_krum \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=pattern \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=3 \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=100 \
    dir_tag=femnist_mkrum_pgd &&
python main.py -m -cn femnist \
    aggregator=multi_krum \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=a3fl \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=3 \
    cuda_visible_devices=\"5,4,3,2,1,0\" \
    num_rounds=100 \
    dir_tag=femnist_mkrum_pgd &&
python main.py -m -cn femnist \
    aggregator=multi_krum \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=cerberus \
    atk_config.model_poison_method=cerberus \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=3 \
    cuda_visible_devices=\"5,4,3,2,1,0\" \
    num_rounds=100 \
    dir_tag=femnist_mkrum_pgd

#### Model Replacement
python main.py -m -cn femnist \
    aggregator=multi_krum \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=pattern \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=100 \
    dir_tag=femnist_mkrum_modelreplace && 
python main.py -m -cn femnist \
    aggregator=multi_krum \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=100 \
    dir_tag=femnist_mkrum_modelreplace &&
python main.py -m -cn femnist \
    aggregator=multi_krum \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=distributed \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=100 \
    dir_tag=femnist_mkrum_modelreplace