######## Robust Aggregation defense against multishot attack ########

############## CIFAR10 ################
# One-line argument using Hydra --multirun 
# For efficiency, you may run attacks in different processes


python main.py -m -cn cifar10 \
    aggregator=coordinate_median,geometric_median,trimmed_mean,krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern,distributed,edge_case,iba \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=200 \
    dir_tag=cifar10_robust_aggregation


python main.py -m -cn femnist \
    aggregator=coordinate_median,geometric_median,trimmed_mean,krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=pattern,distributed,edge_case,iba \
    cuda_visible_devices=\"1,2,3\" \
    num_rounds=200 \
    dir_tag=femnist_robust_aggregation



#### edge-case only
python main.py -m -cn cifar10 \
    aggregator=coordinate_median,geometric_median,trimmed_mean,krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=edge_case \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=200 \
    dir_tag=cifar10_robust_aggregation_edgecase


python main.py -m -cn femnist \
    aggregator=coordinate_median,geometric_median,trimmed_mean,krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=edge_case \
    cuda_visible_devices=\"0,2,1,4,5\" \
    num_rounds=200 \
    dir_tag=femnist_robust_aggregation_edgecase



python main.py -cn cifar10 \
    aggregator=robustlr \
    aggregator_config.robust_lr.robustLR_threshold=4 \
    no_attack=True \
    num_rounds=2000 \
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





#### Fix NC, WeakDP and RobustLR parameters
python main.py -cn cifar10 \
    aggregator=robustlr \
    aggregator_config.robust_lr.robustLR_threshold=4 \
    no_attack=True \
    num_rounds=2000 \
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

python main.py -cn cifar10 \
    aggregator=weakdp \
    no_attack=True \
    num_rounds=2000 \
    checkpoint=null \
    save_checkpoint=False \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"0,1,2,3,4\" \
    save_checkpoint=True \
    "save_checkpoint_rounds=[1000]" \
    test_every=10 \
    dir_tag=check_acc_robust_aggregation_cifar10 

python main.py -cn femnist \
    aggregator=weakdp \
    no_attack=True \
    num_rounds=2000 \
    checkpoint=null \
    save_checkpoint=False \
    save_logging=csv \
    num_gpus=0.5 \
    num_cpus=1 \
    cuda_visible_devices=\"0,1,2,3,4\" \
    save_checkpoint=True \
    "save_checkpoint_rounds=[1000]" \
    test_every=10 \
    dir_tag=check_acc_robust_aggregation_femnist 

python main.py -cn cifar10 \
    aggregator=norm_clipping \
    no_attack=True \
    num_rounds=2000 \
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


#### Baseline
python main.py -m -cn cifar10 \
    aggregator=unweighted_fedavg,coordinate_median,geometric_median,trimmed_mean,krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.poison_ratio=0.5 \
    cuda_visible_devices=\"0,1,2,3\" \
    num_rounds=200 \
    dir_tag=cifar10_robust_aggregation_baseline

python main.py -m -cn cifar10 \
    aggregator=unweighted_fedavg,coordinate_median,geometric_median,trimmed_mean,krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=200 \
    dir_tag=cifar10_robust_aggregation_baseline

python main.py -m -cn cifar10 \
    aggregator=unweighted_fedavg,coordinate_median,geometric_median,trimmed_mean,krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=a3fl \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=200 \
    dir_tag=cifar10_robust_aggregation_baseline

python main.py -m -cn cifar10 \
    aggregator=unweighted_fedavg,coordinate_median,geometric_median,trimmed_mean,krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=cerberus \
    atk_config.model_poison_method=cerberus \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=200 \
    dir_tag=cifar10_robust_aggregation_baseline

# FIX WDP
python main.py -m -cn cifar10 \
    aggregator=weakdp \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.poison_ratio=0.5 \
    cuda_visible_devices=\"0,1,2,3\" \
    num_rounds=200 \
    dir_tag=cifar10_robust_aggregation_baseline

python main.py -m -cn cifar10 \
    aggregator=weakdp \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern,a3fl \
    cuda_visible_devices=\"2,1,4,3\" \
    num_rounds=200 \
    dir_tag=cifar10_robust_aggregation_baseline

python main.py -m -cn cifar10 \
    aggregator=weakdp \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=cerberus \
    atk_config.model_poison_method=cerberus \
    cuda_visible_devices=\"4,3,2,1\" \
    num_rounds=200 \
    dir_tag=cifar10_robust_aggregation_baseline

#### PGD attack
python main.py -m -cn cifar10 \
    aggregator=unweighted_fedavg,coordinate_median,geometric_median,trimmed_mean,krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=3 \
    atk_config.poison_ratio=0.5 \
    cuda_visible_devices=\"0,1,2,3\" \
    num_rounds=200 \
    dir_tag=cifar10_robust_aggregation_pgd

# FIX WDP
python main.py -m -cn cifar10 \
    aggregator=weakdp \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=3 \
    atk_config.poison_ratio=0.5 \
    cuda_visible_devices=\"0,1,2,3\" \
    num_rounds=200 \
    dir_tag=cifar10_robust_aggregation_pgd

python main.py -m -cn cifar10 \
    aggregator=unweighted_fedavg,coordinate_median,geometric_median,trimmed_mean,krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=3 \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=200 \
    dir_tag=cifar10_robust_aggregation_pgd

python main.py -m -cn cifar10 \
    aggregator=unweighted_fedavg,coordinate_median,geometric_median,trimmed_mean,krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=a3fl \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=3 \
    cuda_visible_devices=\"5,4,3,2,1,0\" \
    num_rounds=200 \
    dir_tag=cifar10_robust_aggregation_pgd

python main.py -m -cn cifar10 \
    aggregator=unweighted_fedavg,coordinate_median,geometric_median,trimmed_mean,krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=cerberus \
    atk_config.model_poison_method=cerberus \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=3 \
    cuda_visible_devices=\"5,4,3,2,1,0\" \
    num_rounds=200 \
    dir_tag=cifar10_robust_aggregation_pgd


#### Model Replacement
python main.py -m -cn cifar10 \
    aggregator=unweighted_fedavg,coordinate_median,geometric_median,trimmed_mean,krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=200 \
    dir_tag=cifar10_robust_aggregation_modelreplace && 
python main.py -m -cn cifar10 \
    aggregator=unweighted_fedavg,coordinate_median,geometric_median,trimmed_mean,krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=200 \
    dir_tag=cifar10_robust_aggregation_modelreplace &&
python main.py -m -cn cifar10 \
    aggregator=unweighted_fedavg,coordinate_median,geometric_median,trimmed_mean,krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=distributed \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=200 \
    dir_tag=cifar10_robust_aggregation_modelreplace





###################
# FEMNIST
python main.py -m -cn femnist \
    aggregator=unweighted_fedavg,coordinate_median,geometric_median,trimmed_mean,krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.poison_ratio=0.5 \
    cuda_visible_devices=\"0,1,2,3\" \
    num_rounds=200 \
    dir_tag=femnist_robust_aggregation_baseline

python main.py -m -cn femnist \
    aggregator=unweighted_fedavg,coordinate_median,geometric_median,trimmed_mean,krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=pattern \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=200 \
    dir_tag=femnist_robust_aggregation_baseline

python main.py -m -cn femnist \
    aggregator=unweighted_fedavg,coordinate_median,geometric_median,trimmed_mean,krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=a3fl \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=200 \
    dir_tag=femnist_robust_aggregation_baseline

python main.py -m -cn femnist \
    aggregator=unweighted_fedavg,coordinate_median,geometric_median,trimmed_mean,krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=cerberus \
    atk_config.model_poison_method=cerberus \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=200 \
    dir_tag=femnist_robust_aggregation_baseline


#### PGD attack
python main.py -m -cn femnist \
    aggregator=unweighted_fedavg,coordinate_median,geometric_median,trimmed_mean,krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=3 \
    atk_config.poison_ratio=0.5 \
    cuda_visible_devices=\"0,1,2,3\" \
    num_rounds=200 \
    dir_tag=femnist_robust_aggregation_pgd

python main.py -m -cn femnist \
    aggregator=unweighted_fedavg,coordinate_median,geometric_median,trimmed_mean,krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=pattern \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=3 \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=200 \
    dir_tag=femnist_robust_aggregation_pgd

python main.py -m -cn femnist \
    aggregator=unweighted_fedavg,coordinate_median,geometric_median,trimmed_mean,krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=a3fl \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=3 \
    cuda_visible_devices=\"5,4,3,2,1,0\" \
    num_rounds=200 \
    dir_tag=femnist_robust_aggregation_pgd

python main.py -m -cn femnist \
    aggregator=unweighted_fedavg,coordinate_median,geometric_median,trimmed_mean,krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=cerberus \
    atk_config.model_poison_method=cerberus \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=3 \
    cuda_visible_devices=\"5,4,3,2,1,0\" \
    num_rounds=200 \
    dir_tag=femnist_robust_aggregation_pgd


#### Model Replacement
python main.py -m -cn femnist \
    aggregator=unweighted_fedavg,coordinate_median,geometric_median,trimmed_mean,krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=pattern \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=200 \
    dir_tag=femnist_robust_aggregation_modelreplace && 
python main.py -m -cn femnist \
    aggregator=unweighted_fedavg,coordinate_median,geometric_median,trimmed_mean,krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=200 \
    dir_tag=femnist_robust_aggregation_modelreplace &&
python main.py -m -cn femnist \
    aggregator=unweighted_fedavg,coordinate_median,geometric_median,trimmed_mean,krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=distributed \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=200 \
    dir_tag=femnist_robust_aggregation_modelreplace