python main.py -m -cn cifar10 \
    aggregator=norm_clipping \
    aggregator_config.norm_clipping.clipping_norm=0.5,1,1.5,2 \
    no_attack=True \
    cuda_visible_devices=\"5,4,6,7\" \
    checkpoint=null \
    save_model=True \
    save_checkpoint=False \
    "save_checkpoint_rounds=[1000]" \
    num_rounds=1000 \
    dir_tag=study_normclipping

python main.py -m -cn cifar10 \
    aggregator=weakdp \
    aggregator_config.weakdp.clipping_norm=1.0 \
    aggregator_config.weakdp.std_dev=0.025 \
    no_attack=True \
    cuda_visible_devices=\"4,6,5,7\" \
    checkpoint=null \
    save_model=True \
    save_checkpoint=False \
    "save_checkpoint_rounds=[1000]" \
    num_rounds=1000 \
    dir_tag=study_weakdp

python main.py -m -cn cifar10 \
    aggregator=weakdp \
    aggregator_config.weakdp.clipping_norm=1.0 \
    aggregator_config.weakdp.std_dev=0.025 \
    no_attack=True \
    cuda_visible_devices=\"4,6,5,7\" \
    checkpoint=null \
    save_model=True \
    save_checkpoint=False \
    "save_checkpoint_rounds=[1000]" \
    num_rounds=1000 \
    dir_tag=study_weakdp

python main.py -m -cn cifar10 \
    aggregator=weakdp \
    aggregator_config.weakdp.clipping_norm=1.5 \
    aggregator_config.weakdp.std_dev=0.025 \
    no_attack=True \
    cuda_visible_devices=\"4,6,5,7\" \
    checkpoint=null \
    save_model=True \
    save_checkpoint=False \
    "save_checkpoint_rounds=[1000]" \
    num_rounds=1000 \
    dir_tag=study_weakdp

python main.py -m -cn cifar10 \
    aggregator=weakdp \
    aggregator_config.weakdp.clipping_norm=2.0 \
    aggregator_config.weakdp.std_dev=0.025 \
    no_attack=True \
    cuda_visible_devices=\"4,6,5,7\" \
    checkpoint=null \
    save_model=True \
    save_checkpoint=False \
    "save_checkpoint_rounds=[1000]" \
    num_rounds=1000 \
    dir_tag=study_weakdp

python main.py -m -cn cifar10 \
    aggregator=weakdp \
    aggregator_config.weakdp.clipping_norm=3.0 \
    aggregator_config.weakdp.std_dev=0.025 \
    no_attack=True \
    cuda_visible_devices=\"4,6,5,7\" \
    checkpoint=null \
    save_model=True \
    save_checkpoint=False \
    "save_checkpoint_rounds=[1000]" \
    num_rounds=1000 \
    dir_tag=study_weakdp

python main.py -m -cn femnist \
    aggregator=norm_clipping \
    aggregator_config.norm_clipping.clipping_norm=0.5 \
    no_attack=True \
    cuda_visible_devices=\"5,4,3,2\" \
    checkpoint=null \
    save_model=True \
    save_checkpoint=False \
    "save_checkpoint_rounds=[1000]" \
    num_rounds=1000 \
    dir_tag=study_normclipping

python main.py -m -cn femnist \
    aggregator=weakdp \
    aggregator_config.weakdp.clipping_norm=0.5 \
    aggregator_config.weakdp.std_dev=0.025 \
    no_attack=True \
    cuda_visible_devices=\"4,6,5,7\" \
    checkpoint=null \
    save_model=True \
    save_checkpoint=False \
    "save_checkpoint_rounds=[1000]" \
    num_rounds=1000 \
    dir_tag=study_weakdp

python main.py -m -cn femnist \
    aggregator=weakdp \
    aggregator_config.weakdp.clipping_norm=0.25 \
    aggregator_config.weakdp.std_dev=0.025 \
    no_attack=True \
    cuda_visible_devices=\"4,6,5,7\" \
    checkpoint=null \
    save_model=True \
    save_checkpoint=False \
    "save_checkpoint_rounds=[1000]" \
    num_rounds=1000 \
    dir_tag=study_weakdp

python main.py -m -cn femnist \
    aggregator=weakdp \
    aggregator_config.weakdp.clipping_norm=3.0 \
    aggregator_config.weakdp.std_dev=0.025 \
    no_attack=True \
    cuda_visible_devices=\"4,6,5,7\" \
    checkpoint=null \
    save_model=True \
    save_checkpoint=False \
    "save_checkpoint_rounds=[1000]" \
    num_rounds=1000 \
    dir_tag=study_weakdp