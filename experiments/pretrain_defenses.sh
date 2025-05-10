######## Robust Aggregation ########
python main.py -m \
    aggregator=coordinate_median,geometric_median,trimmed_mean,krum,foolsgold,robustlr,norm_clipping \
    no_attack=True \
    num_rounds=1 \
    save_model=True \
    "save_model_rounds=[2300]" \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"0,1,2,4,5\" \
    dir_tag=pretrain_robust_aggregation

