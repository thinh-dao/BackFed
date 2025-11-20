python main.py -m -cn cifar10 \
    aggregator=krum,multi_krum,bulyan,flare,fltrust,foolsgold,geometric_median,coordinate_median,robustlr,trimmed_mean,norm_clipping,weakdp \
    no_attack=True \
    num_rounds=1000 \
    checkpoint=null \
    save_checkpoint=False \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"0,1,2,3,4\" \
    save_checkpoint=True \
    "save_checkpoint_rounds=[1000]" \
    test_every=5 \
    dir_tag=check_acc_robust_aggregation_cifar10

python main.py -m -cn femnist \
    aggregator=krum,multi_krum,bulyan,flare,fltrust,foolsgold,geometric_median,coordinate_median,robustlr,trimmed_mean,norm_clipping,weakdp \
    no_attack=True \
    num_rounds=1000 \
    checkpoint=null \
    save_checkpoint=False \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"0,1,2,3,4\" \
    save_checkpoint=True \
    "save_checkpoint_rounds=[1000]" \
    test_every=5 \
    dir_tag=check_acc_robust_aggregation_femnist

python main.py -m -cn tiny \
    aggregator=flare,fltrust,foolsgold,geometric_median,coordinate_median \
    no_attack=True \
    num_rounds=1000 \
    checkpoint=null \
    save_checkpoint=False \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"0,1,2,3,4\" \
    save_checkpoint=True \
    "save_checkpoint_rounds=[1000]" \
    test_every=10 \
    dir_tag=check_acc_robust_aggregation_tiny

python main.py -m -cn tiny \
    aggregator=robustlr,trimmed_mean,norm_clipping,weakdp \
    no_attack=True \
    num_rounds=1000 \
    checkpoint=null \
    save_checkpoint=False \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"0,1,2,3,4\" \
    save_checkpoint=True \
    "save_checkpoint_rounds=[1000]" \
    test_every=10 \
    dir_tag=check_acc_robust_aggregation_tiny
    
python main.py -cn cifar10 \
    aggregator=robustlr \
    aggregator_config.robust_lr.robustLR_threshold=4 \
    no_attack=True \
    num_rounds=1000 \
    checkpoint=null \
    save_checkpoint=False \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"0,1,2,3,4\" \
    save_checkpoint=True \
    "save_checkpoint_rounds=[1000]" \
    test_every=5 \
    dir_tag=check_acc_robust_aggregation_cifar10 



python main.py -m -cn cifar10 \
    aggregator=krum,multi_krum,bulyan,flare,fltrust,foolsgold,geometric_median,coordinate_median,robustlr,trimmed_mean,norm_clipping,weakdp \
    no_attack=True \
    num_rounds=6 \
    checkpoint=null \
    save_checkpoint=False \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"0,1,2,3,4\" \
    save_checkpoint=True \
    "save_checkpoint_rounds=[1000]" \
    test_every=5 \
    dir_tag=agg_time 
    
python main.py -m -cn femnist \
    aggregator=krum,multi_krum,bulyan,flare,fltrust,foolsgold,geometric_median,coordinate_median,robustlr,trimmed_mean,norm_clipping,weakdp \
    no_attack=True \
    num_rounds=6 \
    checkpoint=null \
    save_checkpoint=False \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"0,1,2,3,4\" \
    save_checkpoint=True \
    "save_checkpoint_rounds=[1000]" \
    test_every=5 \
    dir_tag=agg_time