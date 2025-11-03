python main.py -m -cn cifar10 \
    aggregator=norm_clipping \
    aggregator_config.norm_clipping.clipping_norm=0.5,1,2 \
    no_attack=True \
    cuda_visible_devices=\"5,4,6,7\" \
    checkpoint=null \
    save_model=True \
    save_checkpoint=False \
    "save_checkpoint_rounds=[1000]" \
    num_rounds=1000 \
    dir_tag=study_normclipping


