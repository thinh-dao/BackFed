python main.py -m -cn cifar10\
    aggregator=unweighted_fedavg \
    atk_config=cifar10_multishot \
    atk_config.model_poison_method=base \
    atk_config.data_poison_method=iba \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=200 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=test_adversarial_attack \
    cuda_visible_devices=\"7,6,5,4\"

python main.py -m -cn cifar10\
    aggregator=unweighted_fedavg \
    atk_config=cifar10_multishot \
    atk_config.model_poison_method=base \
    atk_config.data_poison_method=a3fl \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=200 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=test_adversarial_attack \
    cuda_visible_devices=\"4,5,6,7\"

python main.py -m -cn cifar10\
    aggregator=unweighted_fedavg \
    atk_config=cifar10_multishot \
    atk_config.model_poison_method=base \
    atk_config.data_poison_method=cerberus \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=200 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=test_adversarial_attack \
    cuda_visible_devices=\"5,6,7,4\"