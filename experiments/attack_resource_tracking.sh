python main.py -m -cn cifar10 \
    aggregator=unweighted_fedavg \
    atk_config=cifar10_multishot \
    atk_config.data_poison_config.a3fl.trigger_outter_epochs=200 \
    atk_config.data_poison_config.a3fl.dm_adv_K=1 \
    atk_config.data_poison_method=pattern,ceberus,a3fl,iba \
    atk_config.adversary_selection=fixed \
    atk_config.selection_scheme=single-adversary \
    cuda_visible_devices=\"6,7\" \
    num_rounds=10 \
    dir_tag=attack_resource_tracking \
    training_mode=sequential &&
python main.py -m -cn cifar10 \
    aggregator=multi_krum \
    atk_config=cifar10_multishot \
    atk_config.model_poison_method=anticipate,chameleon,neurotoxin \
    atk_config.data_poison_method=pattern \
    atk_config.adversary_selection=fixed \
    atk_config.selection_scheme=single-adversary \
    cuda_visible_devices=\"6,7\" \
    num_rounds=10 \
    dir_tag=attack_resource_tracking \
    training_mode=sequential

