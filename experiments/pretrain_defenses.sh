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
    cuda_visible_devices=\"1,0,2,4,5\" \
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
    cuda_visible_devices=\"5,4,3,2,1\" \
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