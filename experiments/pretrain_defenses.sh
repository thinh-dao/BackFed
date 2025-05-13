######## Robust Aggregation ########
python main.py -m \
    aggregator=coordinate_median,geometric_median,trimmed_mean,krum,foolsgold,robustlr,norm_clipping \
    no_attack=True \
    num_rounds=301 \
    save_checkpoint=True \
    "save_model_rounds=[2300]" \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"0,1,2,4,5\" \
    dir_tag=pretrain_robust_aggregation





python main.py \
    aggregator=coordinate_median \
    no_attack=True \
    num_rounds=301 \
    save_checkpoint=True \
    "save_model_rounds=[2300]" \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"0,1,2,4,5\" \
    dir_tag=pretrain_robust_aggregation && \
python main.py -m \
    aggregator=coordinate_median \
    checkpoint=2300 \
    atk_config=multishot \
    atk_config.data_poison_method=pattern,edge_case,a3fl,iba,distributed \
    atk_config.poison_start_round=2301 \
    atk_config.poison_end_round=2600 \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"0,1,2,4,5\" \
    dir_tag=robust_aggregation_new



python main.py \
    aggregator=geometric_median \
    no_attack=True \
    num_rounds=301 \
    save_checkpoint=True \
    "save_model_rounds=[2300]" \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"3,4,5,2,1\" \
    dir_tag=pretrain_robust_aggregation && \
python main.py -m \
    aggregator=geometric_median \
    checkpoint=2300 \
    atk_config=multishot \
    atk_config.data_poison_method=pattern,edge_case,a3fl,iba,distributed \
    atk_config.poison_start_round=2301 \
    atk_config.poison_end_round=2600 \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"2,1,0,4,5\" \
    dir_tag=robust_aggregation_new



python main.py \
    aggregator=trimmed_mean \
    no_attack=True \
    num_rounds=301 \
    save_checkpoint=True \
    "save_model_rounds=[2300]" \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"4,3,2,1,0\" \
    dir_tag=pretrain_robust_aggregation && \
python main.py -m \
    aggregator=trimmed_mean \
    checkpoint=2300 \
    atk_config=multishot \
    atk_config.data_poison_method=pattern,edge_case,a3fl,iba,distributed \
    atk_config.poison_start_round=2301 \
    atk_config.poison_end_round=2600 \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"5,3,1,2,0\" \
    dir_tag=robust_aggregation_new




python main.py \
    aggregator=krum \
    no_attack=True \
    num_rounds=301 \
    save_checkpoint=True \
    "save_model_rounds=[2300]" \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"0,1,2,4,5\" \
    dir_tag=pretrain_robust_aggregation && \
python main.py -m \
    aggregator=krum \
    checkpoint=2300 \
    atk_config=multishot \
    atk_config.data_poison_method=pattern,edge_case,a3fl,iba,distributed \
    atk_config.poison_start_round=2301 \
    atk_config.poison_end_round=2600 \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"3,2,1,0,4\" \
    dir_tag=robust_aggregation_new




python main.py \
    aggregator=foolsgold \
    no_attack=True \
    num_rounds=301 \
    save_checkpoint=True \
    "save_model_rounds=[2300]" \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"1,2,5,4,6\" \
    dir_tag=pretrain_robust_aggregation && \
python main.py -m \
    aggregator=foolsgold \
    checkpoint=2300 \
    atk_config=multishot \
    atk_config.data_poison_method=pattern,edge_case,a3fl,iba,distributed \
    atk_config.poison_start_round=2301 \
    atk_config.poison_end_round=2600 \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"2,1,6,5,4\" \
    dir_tag=robust_aggregation_new


python main.py \
    aggregator=robustlr \
    no_attack=True \
    num_rounds=301 \
    save_checkpoint=True \
    "save_model_rounds=[2300]" \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"5,4,3,2,1\" \
    dir_tag=pretrain_robust_aggregation && \
python main.py -m \
    aggregator=robustlr \
    checkpoint=2300 \
    atk_config=multishot \
    atk_config.data_poison_method=pattern,edge_case,a3fl,iba,distributed \
    atk_config.poison_start_round=2301 \
    atk_config.poison_end_round=2600 \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"1,0,2,4,5\" \
    dir_tag=robust_aggregation_new



python main.py \
    aggregator=norm_clipping \
    no_attack=True \
    num_rounds=301 \
    save_checkpoint=True \
    "save_model_rounds=[2300]" \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"2,1,0,4,5\" \
    dir_tag=pretrain_robust_aggregation && \
python main.py -m \
    aggregator=norm_clipping \
    checkpoint=2300 \
    atk_config=multishot \
    atk_config.data_poison_method=pattern,edge_case,a3fl,iba,distributed \
    atk_config.poison_start_round=2301 \
    atk_config.poison_end_round=2600 \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"3,2,1,4,5\" \
    dir_tag=robust_aggregation_new



############### FEMNIST ##################


python main.py \
    aggregator=coordinate_median \
    no_attack=True \
    num_clients=3383 \
    num_clients_per_round=30 \
    dataset=emnist_byclass \
    test_batch_size=1000 \
    num_workers=8 \
    model=mnistnet \
    checkpoint=checkpoints/EMNIST_BYCLASS_unweighted_fedavg/mnistnet_round_1000_dir_0.5.pth \
    save_checkpoint=True \
    num_rounds=301 \
    "save_model_rounds=[1300]" \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"2,3,4,5,6\" \
    dir_tag=pretrain_robust_aggregation_emnist && \
python main.py -m \
    aggregator=coordinate_median \
    num_clients=3383 \
    num_clients_per_round=30 \
    dataset=emnist_byclass \
    test_batch_size=1000 \
    num_workers=8 \
    model=mnistnet \
    checkpoint=1300 \
    atk_config=emnist_multishot \
    atk_config.data_poison_method=pattern,edge_case,a3fl,iba,distributed \
    atk_config.poison_start_round=1301 \
    atk_config.poison_end_round=1600 \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"2,3,4,5,6\" \
    dir_tag=robust_aggregation_emnist



python main.py \
    aggregator=geometric_median \
    no_attack=True \
    num_clients=3383 \
    num_clients_per_round=30 \
    dataset=emnist_byclass \
    test_batch_size=1000 \
    num_workers=8 \
    model=mnistnet \
    checkpoint=checkpoints/EMNIST_BYCLASS_unweighted_fedavg/mnistnet_round_1000_dir_0.5.pth \
    save_checkpoint=True \
    num_rounds=301 \
    "save_model_rounds=[1300]" \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"0,1,2,3,4\" \
    dir_tag=pretrain_robust_aggregation_emnist && \
python main.py -m \
    aggregator=geometric_median \
    num_clients=3383 \
    num_clients_per_round=30 \
    dataset=emnist_byclass \
    test_batch_size=1000 \
    num_workers=8 \
    model=mnistnet \
    checkpoint=1300 \
    atk_config=emnist_multishot \
    atk_config.data_poison_method=pattern,edge_case,a3fl,iba,distributed \
    atk_config.poison_start_round=1301 \
    atk_config.poison_end_round=1600 \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"0,1,2,3,4\" \
    dir_tag=robust_aggregation_emnist



python main.py \
    aggregator=trimmed_mean \
    no_attack=True \
    num_clients=3383 \
    num_clients_per_round=30 \
    dataset=emnist_byclass \
    test_batch_size=1000 \
    num_workers=8 \
    model=mnistnet \
    checkpoint=checkpoints/EMNIST_BYCLASS_unweighted_fedavg/mnistnet_round_1000_dir_0.5.pth \
    save_checkpoint=True \
    num_rounds=301 \
    "save_model_rounds=[1300]" \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"4,3,2,1,0\" \
    dir_tag=pretrain_robust_aggregation_emnist && \
python main.py -m \
    aggregator=trimmed_mean \
    num_clients=3383 \
    num_clients_per_round=30 \
    dataset=emnist_byclass \
    test_batch_size=1000 \
    num_workers=8 \
    model=mnistnet \
    checkpoint=1300 \
    atk_config=emnist_multishot \
    atk_config.data_poison_method=pattern,edge_case,a3fl,iba,distributed \
    atk_config.poison_start_round=1301 \
    atk_config.poison_end_round=1600 \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"4,3,2,1,0\" \
    dir_tag=robust_aggregation_emnist




python main.py \
    aggregator=krum \
    no_attack=True \
    num_clients=3383 \
    num_clients_per_round=30 \
    dataset=emnist_byclass \
    test_batch_size=1000 \
    num_workers=8 \
    model=mnistnet \
    checkpoint=checkpoints/EMNIST_BYCLASS_unweighted_fedavg/mnistnet_round_1000_dir_0.5.pth \
    save_checkpoint=True \
    num_rounds=301 \
    "save_model_rounds=[1300]" \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"6,5,4,3,2\" \
    dir_tag=pretrain_robust_aggregation_emnist && \
python main.py -m \
    aggregator=krum \
    num_clients=3383 \
    num_clients_per_round=30 \
    dataset=emnist_byclass \
    test_batch_size=1000 \
    num_workers=8 \
    model=mnistnet \
    checkpoint=1300 \
    atk_config=emnist_multishot \
    atk_config.data_poison_method=pattern,edge_case,a3fl,iba,distributed \
    atk_config.poison_start_round=1301 \
    atk_config.poison_end_round=1600 \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"6,5,4,3,2\" \
    dir_tag=robust_aggregation_emnist




python main.py \
    aggregator=foolsgold \
    no_attack=True \
    num_clients=3383 \
    num_clients_per_round=30 \
    dataset=emnist_byclass \
    test_batch_size=1000 \
    num_workers=8 \
    model=mnistnet \
    checkpoint=checkpoints/EMNIST_BYCLASS_unweighted_fedavg/mnistnet_round_1000_dir_0.5.pth \
    save_checkpoint=True \
    num_rounds=301 \
    "save_model_rounds=[1300]" \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"0,1,2,3,4\" \
    dir_tag=pretrain_robust_aggregation_emnist && \
python main.py -m \
    aggregator=foolsgold \
    num_clients=3383 \
    num_clients_per_round=30 \
    dataset=emnist_byclass \
    test_batch_size=1000 \
    num_workers=8 \
    model=mnistnet \
    checkpoint=1300 \
    atk_config=emnist_multishot \
    atk_config.data_poison_method=pattern,edge_case,a3fl,iba,distributed \
    atk_config.poison_start_round=1301 \
    atk_config.poison_end_round=1600 \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"0,1,2,3,4\" \
    dir_tag=robust_aggregation_emnist



python main.py \
    aggregator=robustlr \
    no_attack=True \
    num_clients=3383 \
    num_clients_per_round=30 \
    dataset=emnist_byclass \
    test_batch_size=1000 \
    num_workers=8 \
    model=mnistnet \
    checkpoint=checkpoints/EMNIST_BYCLASS_unweighted_fedavg/mnistnet_round_1000_dir_0.5.pth \
    save_checkpoint=True \
    num_rounds=301 \
    "save_model_rounds=[1300]" \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"7,6,5,4,3\" \
    dir_tag=pretrain_robust_aggregation_emnist && \
python main.py -m \
    aggregator=robustlr \
    num_clients=3383 \
    num_clients_per_round=30 \
    dataset=emnist_byclass \
    test_batch_size=1000 \
    num_workers=8 \
    model=mnistnet \
    checkpoint=1300 \
    atk_config=emnist_multishot \
    atk_config.data_poison_method=pattern,edge_case,a3fl,iba,distributed \
    atk_config.poison_start_round=1301 \
    atk_config.poison_end_round=1600 \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"7,6,5,4,3\" \
    dir_tag=robust_aggregation_emnist



python main.py \
    aggregator=norm_clipping \
    no_attack=True \
    num_clients=3383 \
    num_clients_per_round=30 \
    dataset=emnist_byclass \
    test_batch_size=1000 \
    num_workers=8 \
    model=mnistnet \
    checkpoint=checkpoints/EMNIST_BYCLASS_unweighted_fedavg/mnistnet_round_1000_dir_0.5.pth \
    save_checkpoint=True \
    num_rounds=301 \
    "save_model_rounds=[1300]" \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"0,1,4,2,3\" \
    dir_tag=pretrain_robust_aggregation_emnist && \
python main.py -m \
    aggregator=norm_clipping \
    num_clients=3383 \
    num_clients_per_round=30 \
    dataset=emnist_byclass \
    test_batch_size=1000 \
    num_workers=8 \
    model=mnistnet \
    checkpoint=1300 \
    atk_config=emnist_multishot \
    atk_config.data_poison_method=pattern,edge_case,a3fl,iba,distributed \
    atk_config.poison_start_round=1301 \
    atk_config.poison_end_round=1600 \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"4,0,1,2,3\" \
    dir_tag=robust_aggregation_emnist

