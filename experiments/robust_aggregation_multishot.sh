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
    aggregator=unweighted_fedavg,coordinate_median,geometric_median,trimmed_mean,krum,multi_krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=200 \
    checkpoint=1000 \
    dir_tag=cifar10_robust_aggregation_baseline &&
python main.py -m -cn cifar10 \
    aggregator=unweighted_fedavg,coordinate_median,geometric_median,trimmed_mean,krum,multi_krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.poison_ratio=0.5 \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=200 \
    checkpoint=1000 \
    dir_tag=cifar10_robust_aggregation_baseline &&
python main.py -m -cn cifar10 \
    aggregator=unweighted_fedavg,coordinate_median,geometric_median,trimmed_mean,krum,multi_krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=a3fl \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=200 \
    checkpoint=1000 \
    dir_tag=cifar10_robust_aggregation_baseline &&
python main.py -m -cn cifar10 \
    aggregator=unweighted_fedavg,coordinate_median,geometric_median,trimmed_mean,krum,multi_krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=cifar10_multishot \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    atk_config.data_poison_method=cerberus \
    atk_config.model_poison_method=cerberus \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=200 \
    checkpoint=1000 \
    dir_tag=cifar10_robust_aggregation_baseline

#### PGD attack
python main.py -m -cn cifar10 \
    aggregator=unweighted_fedavg,coordinate_median,geometric_median,trimmed_mean,krum,multi_krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=3.0 \
    atk_config.poison_ratio=0.625 \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"0,1,2,3\" \
    num_rounds=200 \
    checkpoint=1000 \
    dir_tag=cifar10_robust_aggregation_pgd &&
python main.py -m -cn cifar10 \
    aggregator=unweighted_fedavg,coordinate_median,geometric_median,trimmed_mean,krum,multi_krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=3.0 \
    atk_config.poison_ratio=0.625 \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"1,2,3,5\" \
    num_rounds=200 \
    checkpoint=1000 \
    dir_tag=cifar10_robust_aggregation_pgd &&
python main.py -m -cn cifar10 \
    aggregator=unweighted_fedavg,coordinate_median,geometric_median,trimmed_mean,krum,multi_krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=a3fl \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=3.0 \
    atk_config.poison_ratio=0.625 \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"5,4,3,2\" \
    num_rounds=200 \
    checkpoint=1000 \
    dir_tag=cifar10_robust_aggregation_pgd


#### Model Replacement
python main.py -m -cn cifar10 \
    aggregator=unweighted_fedavg,coordinate_median,geometric_median,trimmed_mean,krum,multi_krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    num_rounds=200 \
    checkpoint=1000 \
    cuda_visible_devices=\"2,3,4,0\" \
    dir_tag=cifar10_robust_aggregation_modelreplace && 
python main.py -m -cn cifar10 \
    aggregator=unweighted_fedavg,coordinate_median,geometric_median,trimmed_mean,krum,multi_krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    num_rounds=200 \
    checkpoint=1000 \
    cuda_visible_devices=\"3,2,1,0\" \
    dir_tag=cifar10_robust_aggregation_modelreplace &&
python main.py -m -cn cifar10 \
    aggregator=unweighted_fedavg,coordinate_median,geometric_median,trimmed_mean,krum,multi_krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=distributed \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    num_rounds=200 \
    checkpoint=1000 \
    cuda_visible_devices=\"1,2,3,4\" \
    dir_tag=cifar10_robust_aggregation_modelreplace





###################
# FEMNIST
#### Baseline
python main.py -m -cn femnist \
    aggregator=unweighted_fedavg,coordinate_median,geometric_median,trimmed_mean,krum,multi_krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=pattern \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"4,3,2,1\" \
    num_rounds=200 \
    checkpoint=1000 \
    dir_tag=femnist_robust_aggregation_baseline &&
python main.py -m -cn femnist \
    aggregator=unweighted_fedavg,coordinate_median,geometric_median,trimmed_mean,krum,multi_krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.poison_ratio=0.5 \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=200 \
    checkpoint=1000 \
    dir_tag=femnist_robust_aggregation_baseline &&
python main.py -m -cn femnist \
    aggregator=unweighted_fedavg,coordinate_median,geometric_median,trimmed_mean,krum,multi_krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=a3fl \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"1,2,3,5\" \
    num_rounds=200 \
    checkpoint=1000 \
    dir_tag=femnist_robust_aggregation_baseline &&
python main.py -m -cn femnist \
    aggregator=unweighted_fedavg,coordinate_median,geometric_median,trimmed_mean,krum,multi_krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=femnist_multishot \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    atk_config.data_poison_method=cerberus \
    atk_config.model_poison_method=cerberus \
    cuda_visible_devices=\"1,2,3,5\" \
    num_rounds=200 \
    checkpoint=1000 \
    dir_tag=femnist_robust_aggregation_baseline

#### PGD attack
python main.py -m -cn femnist \
    aggregator=unweighted_fedavg,coordinate_median,geometric_median,trimmed_mean,krum,multi_krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=0.75 \
    atk_config.poison_ratio=0.625 \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"0,1,2,3\" \
    num_rounds=200 \
    checkpoint=1000 \
    dir_tag=femnist_robust_aggregation_pgd &&
python main.py -m -cn femnist \
    aggregator=unweighted_fedavg,coordinate_median,geometric_median,trimmed_mean,krum,multi_krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=pattern \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=0.75 \
    atk_config.poison_ratio=0.625 \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"1,2,3,5\" \
    num_rounds=200 \
    checkpoint=1000 \
    dir_tag=femnist_robust_aggregation_pgd &&
python main.py -m -cn femnist \
    aggregator=unweighted_fedavg,coordinate_median,geometric_median,trimmed_mean,krum,multi_krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=a3fl \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=0.75 \
    atk_config.poison_ratio=0.625 \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"5,4,3,2\" \
    num_rounds=200 \
    checkpoint=1000 \
    dir_tag=femnist_robust_aggregation_pgd


#### Model Replacement
python main.py -m -cn femnist \
    aggregator=unweighted_fedavg,coordinate_median,geometric_median,trimmed_mean,krum,multi_krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=pattern \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    num_rounds=200 \
    checkpoint=1000 \
    cuda_visible_devices=\"2,3,4,0\" \
    dir_tag=femnist_robust_aggregation_modelreplace && 
python main.py -m -cn femnist \
    aggregator=unweighted_fedavg,coordinate_median,geometric_median,trimmed_mean,krum,multi_krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    num_rounds=200 \
    checkpoint=1000 \
    cuda_visible_devices=\"3,2,1,0\" \
    dir_tag=femnist_robust_aggregation_modelreplace &&
python main.py -m -cn femnist \
    aggregator=unweighted_fedavg,coordinate_median,geometric_median,trimmed_mean,krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=distributed \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    num_rounds=200 \
    checkpoint=1000 \
    cuda_visible_devices=\"1,2,3,4\" \
    dir_tag=femnist_robust_aggregation_modelreplace



################### Fix RLR ###################

# Baseline
python main.py -m -cn cifar10 \
    aggregator=robustlr \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=200 \
    checkpoint=1000 \
    dir_tag=cifar10_robust_aggregation_baseline &&
python main.py -m -cn cifar10 \
    aggregator=robustlr \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.poison_ratio=0.5 \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=200 \
    checkpoint=1000 \
    dir_tag=cifar10_robust_aggregation_baseline &&
python main.py -m -cn cifar10 \
    aggregator=robustlr \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=a3fl \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=200 \
    checkpoint=1000 \
    dir_tag=cifar10_robust_aggregation_baseline &&
python main.py -m -cn cifar10 \
    aggregator=robustlr \
    atk_config=cifar10_multishot \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    atk_config.data_poison_method=cerberus \
    atk_config.model_poison_method=cerberus \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=200 \
    checkpoint=1000 \
    dir_tag=cifar10_robust_aggregation_baseline

#### PGD attack
python main.py -m -cn cifar10 \
    aggregator=robustlr \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=3.0 \
    atk_config.poison_ratio=0.625 \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"0,1,2,3\" \
    num_rounds=200 \
    checkpoint=1000 \
    dir_tag=cifar10_robust_aggregation_pgd &&
python main.py -m -cn cifar10 \
    aggregator=robustlr \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=3.0 \
    atk_config.poison_ratio=0.625 \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"1,2,3,5\" \
    num_rounds=200 \
    checkpoint=1000 \
    dir_tag=cifar10_robust_aggregation_pgd &&
python main.py -m -cn cifar10 \
    aggregator=robustlr \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=a3fl \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=3.0 \
    atk_config.poison_ratio=0.625 \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"5,4,3,2\" \
    num_rounds=200 \
    checkpoint=1000 \
    dir_tag=cifar10_robust_aggregation_pgd


#### Model Replacement
python main.py -m -cn cifar10 \
    aggregator=robustlr \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    num_rounds=200 \
    checkpoint=1000 \
    cuda_visible_devices=\"2,3,4,0\" \
    dir_tag=cifar10_robust_aggregation_modelreplace && 
python main.py -m -cn cifar10 \
    aggregator=robustlr \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    num_rounds=200 \
    checkpoint=1000 \
    cuda_visible_devices=\"3,2,1,0\" \
    dir_tag=cifar10_robust_aggregation_modelreplace &&
python main.py -m -cn cifar10 \
    aggregator=robustlr \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=distributed \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    num_rounds=200 \
    checkpoint=1000 \
    cuda_visible_devices=\"1,2,3,4\" \
    dir_tag=cifar10_robust_aggregation_modelreplace





###################
# FEMNIST
#### Baseline
### Clean training RLR
python main.py -cn femnist \
    aggregator=robustlr \
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

python main.py -m -cn femnist \
    aggregator=robustlr \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=pattern \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"4,3,2,1\" \
    num_rounds=200 \
    checkpoint=1000 \
    dir_tag=femnist_robust_aggregation_baseline &&
python main.py -m -cn femnist \
    aggregator=robustlr \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.poison_ratio=0.5 \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=200 \
    checkpoint=1000 \
    dir_tag=femnist_robust_aggregation_baseline &&
python main.py -m -cn femnist \
    aggregator=robustlr \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=a3fl \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"1,2,3,5\" \
    num_rounds=200 \
    checkpoint=1000 \
    dir_tag=femnist_robust_aggregation_baseline &&
python main.py -m -cn femnist \
    aggregator=robustlr \
    atk_config=femnist_multishot \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    atk_config.data_poison_method=cerberus \
    atk_config.model_poison_method=cerberus \
    cuda_visible_devices=\"1,2,3,5\" \
    num_rounds=200 \
    checkpoint=1000 \
    dir_tag=femnist_robust_aggregation_baseline

#### PGD attack
python main.py -m -cn femnist \
    aggregator=robustlr \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=0.75 \
    atk_config.poison_ratio=0.625 \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"0,1,2,3\" \
    num_rounds=200 \
    checkpoint=1000 \
    dir_tag=femnist_robust_aggregation_pgd &&
python main.py -m -cn femnist \
    aggregator=robustlr \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=pattern \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=0.75 \
    atk_config.poison_ratio=0.625 \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"1,2,3,5\" \
    num_rounds=200 \
    checkpoint=1000 \
    dir_tag=femnist_robust_aggregation_pgd &&
python main.py -m -cn femnist \
    aggregator=robustlr \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=a3fl \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=0.75 \
    atk_config.poison_ratio=0.625 \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"5,4,3,2\" \
    num_rounds=200 \
    checkpoint=1000 \
    dir_tag=femnist_robust_aggregation_pgd


#### Model Replacement
python main.py -m -cn femnist \
    aggregator=robustlr \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=pattern \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    num_rounds=200 \
    checkpoint=1000 \
    cuda_visible_devices=\"2,3,4,0\" \
    dir_tag=femnist_robust_aggregation_modelreplace && 
python main.py -m -cn femnist \
    aggregator=robustlr \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    num_rounds=200 \
    checkpoint=1000 \
    cuda_visible_devices=\"3,2,1,0\" \
    dir_tag=femnist_robust_aggregation_modelreplace &&
python main.py -m -cn femnist \
    aggregator=robustlr \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=distributed \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    num_rounds=200 \
    checkpoint=1000 \
    cuda_visible_devices=\"1,2,3,4\" \
    dir_tag=femnist_robust_aggregation_modelreplace




##### ADD Mkrum
#CIFAR10
python main.py -m -cn cifar10 \
    aggregator=multi_krum \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    num_rounds=200 \
    checkpoint=1000 \
    cuda_visible_devices=\"2,3,4,0\" \
    dir_tag=cifar10_robust_aggregation_modelreplace && 
python main.py -m -cn cifar10 \
    aggregator=multi_krum \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    num_rounds=200 \
    checkpoint=1000 \
    cuda_visible_devices=\"3,2,1,0\" \
    dir_tag=cifar10_robust_aggregation_modelreplace &&
python main.py -m -cn cifar10 \
    aggregator=multi_krum \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=distributed \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    num_rounds=200 \
    checkpoint=1000 \
    cuda_visible_devices=\"1,2,3,4\" \
    dir_tag=cifar10_robust_aggregation_modelreplace


#FEMNIST
python main.py -m -cn femnist \
    aggregator=multi_krum \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=pattern \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    num_rounds=200 \
    checkpoint=1000 \
    cuda_visible_devices=\"2,3,4,0\" \
    dir_tag=femnist_robust_aggregation_modelreplace && 
python main.py -m -cn femnist \
    aggregator=multi_krum \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    num_rounds=200 \
    checkpoint=1000 \
    cuda_visible_devices=\"3,2,1,0\" \
    dir_tag=femnist_robust_aggregation_modelreplace &&
python main.py -m -cn femnist \
    aggregator=multi_krum \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=distributed \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    num_rounds=200 \
    checkpoint=1000 \
    cuda_visible_devices=\"1,2,3,4\" \
    dir_tag=femnist_robust_aggregation_modelreplace


# Add CRFL##################################### CRFLLLL



# Baseline
python main.py -m -cn cifar10 \
    aggregator=crfl \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=200 \
    checkpoint=1000 \
    test_every=200 \
    dir_tag=cifar10_robust_aggregation_baseline &&
python main.py -m -cn cifar10 \
    aggregator=crfl \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.poison_ratio=0.5 \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=200 \
    checkpoint=1000 \
    test_every=200 \
    dir_tag=cifar10_robust_aggregation_baseline &&
python main.py -m -cn cifar10 \
    aggregator=crfl \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=a3fl \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=200 \
    checkpoint=1000 \
    test_every=200 \
    dir_tag=cifar10_robust_aggregation_baseline &&
python main.py -m -cn cifar10 \
    aggregator=crfl \
    atk_config=cifar10_multishot \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    atk_config.data_poison_method=cerberus \
    atk_config.model_poison_method=cerberus \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=200 \
    checkpoint=1000 \
    test_every=200 \
    dir_tag=cifar10_robust_aggregation_baseline

#### PGD attack
python main.py -m -cn cifar10 \
    aggregator=crfl \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=3.0 \
    atk_config.poison_ratio=0.625 \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"0,1,2,3\" \
    num_rounds=200 \
    checkpoint=1000 \
    test_every=200 \
    dir_tag=cifar10_robust_aggregation_pgd &&
python main.py -m -cn cifar10 \
    aggregator=crfl \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=3.0 \
    atk_config.poison_ratio=0.625 \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"1,2,3,5\" \
    num_rounds=200 \
    checkpoint=1000 \
    test_every=200 \
    dir_tag=cifar10_robust_aggregation_pgd &&
python main.py -m -cn cifar10 \
    aggregator=crfl \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=a3fl \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=3.0 \
    atk_config.poison_ratio=0.625 \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"5,4,3,2\" \
    num_rounds=200 \
    checkpoint=1000 \
    test_every=200 \
    dir_tag=cifar10_robust_aggregation_pgd


#### Model Replacement
python main.py -m -cn cifar10 \
    aggregator=crfl \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    num_rounds=200 \
    checkpoint=1000 \
    test_every=200 \
    cuda_visible_devices=\"2,3,4,0\" \
    dir_tag=cifar10_robust_aggregation_modelreplace && 
python main.py -m -cn cifar10 \
    aggregator=crfl \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    num_rounds=200 \
    checkpoint=1000 \
    test_every=200 \
    cuda_visible_devices=\"3,2,1,0\" \
    dir_tag=cifar10_robust_aggregation_modelreplace &&
python main.py -m -cn cifar10 \
    aggregator=crfl \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=distributed \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    num_rounds=200 \
    checkpoint=1000 \
    test_every=200 \
    cuda_visible_devices=\"1,2,3,4\" \
    dir_tag=cifar10_robust_aggregation_modelreplace





###################
# FEMNIST
python main.py -m -cn femnist \
    aggregator=crfl \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=pattern \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"4,3,2,1\" \
    num_rounds=200 \
    checkpoint=1000 \
    test_every=200 \
    dir_tag=femnist_robust_aggregation_baseline &&
python main.py -m -cn femnist \
    aggregator=crfl \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.poison_ratio=0.5 \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=200 \
    checkpoint=1000 \
    test_every=200 \
    dir_tag=femnist_robust_aggregation_baseline &&
python main.py -m -cn femnist \
    aggregator=crfl \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=a3fl \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"1,2,3,5\" \
    num_rounds=200 \
    checkpoint=1000 \
    test_every=200 \
    dir_tag=femnist_robust_aggregation_baseline &&
python main.py -m -cn femnist \
    aggregator=crfl \
    atk_config=femnist_multishot \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    atk_config.data_poison_method=cerberus \
    atk_config.model_poison_method=cerberus \
    cuda_visible_devices=\"1,2,3,5\" \
    num_rounds=200 \
    checkpoint=1000 \
    test_every=200 \
    dir_tag=femnist_robust_aggregation_baseline

#### PGD attack
python main.py -m -cn femnist \
    aggregator=crfl \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=0.75 \
    atk_config.poison_ratio=0.625 \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"0,1,2,3\" \
    num_rounds=200 \
    checkpoint=1000 \
    test_every=200 \
    dir_tag=femnist_robust_aggregation_pgd &&
python main.py -m -cn femnist \
    aggregator=crfl \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=pattern \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=0.75 \
    atk_config.poison_ratio=0.625 \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"1,2,3,5\" \
    num_rounds=200 \
    checkpoint=1000 \
    test_every=200 \
    dir_tag=femnist_robust_aggregation_pgd &&
python main.py -m -cn femnist \
    aggregator=crfl \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=a3fl \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=0.75 \
    atk_config.poison_ratio=0.625 \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"5,4,3,2\" \
    num_rounds=200 \
    checkpoint=1000 \
    test_every=200 \
    dir_tag=femnist_robust_aggregation_pgd


#### Model Replacement
python main.py -m -cn femnist \
    aggregator=crfl \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=pattern \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    num_rounds=200 \
    checkpoint=1000 \
    test_every=200 \
    cuda_visible_devices=\"2,3,4,0\" \
    dir_tag=femnist_robust_aggregation_modelreplace && 
python main.py -m -cn femnist \
    aggregator=crfl \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    num_rounds=200 \
    checkpoint=1000 \
    test_every=200 \
    cuda_visible_devices=\"3,2,1,0\" \
    dir_tag=femnist_robust_aggregation_modelreplace &&
python main.py -m -cn femnist \
    aggregator=crfl \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=distributed \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    num_rounds=200 \
    checkpoint=1000 \
    test_every=200 \
    cuda_visible_devices=\"1,2,3,4\" \
    dir_tag=femnist_robust_aggregation_modelreplace


####### Tiny-ImageNet ########
#### Baseline
python main.py -m -cn tiny \
    aggregator=unweighted_fedavg,coordinate_median,geometric_median,trimmed_mean,krum,multi_krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=tiny_multishot \
    atk_config.data_poison_method=pattern \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"0,1,2,3,4\" \
    num_rounds=200 \
    checkpoint=1000 \
    dir_tag=tiny_robust_aggregation_baseline &&
python main.py -m -cn tiny \
    aggregator=unweighted_fedavg,coordinate_median,geometric_median,trimmed_mean,krum,multi_krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=tiny_multishot \
    atk_config.data_poison_method=a3fl \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"3,1,2,4,0\" \
    num_rounds=200 \
    checkpoint=1000 \
    dir_tag=tiny_robust_aggregation_baseline &&
python main.py -m -cn tiny \
    aggregator=unweighted_fedavg,coordinate_median,geometric_median,trimmed_mean,krum,multi_krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=tiny_multishot \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    atk_config.data_poison_method=cerberus \
    atk_config.model_poison_method=cerberus \
    cuda_visible_devices=\"4,3,2,1,0\" \
    num_rounds=200 \
    checkpoint=1000 \
    dir_tag=tiny_robust_aggregation_baseline

#### PGD attack
python main.py -m -cn tiny \
    aggregator=unweighted_fedavg,coordinate_median,geometric_median,trimmed_mean,krum,multi_krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=tiny_multishot \
    atk_config.data_poison_method=pattern \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=3.0 \
    atk_config.poison_ratio=0.625 \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"1,2,3,4,0\" \
    num_rounds=200 \
    checkpoint=1000 \
    dir_tag=tiny_robust_aggregation_pgd &&
python main.py -m -cn tiny \
    aggregator=unweighted_fedavg,coordinate_median,geometric_median,trimmed_mean,krum,multi_krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=tiny_multishot \
    atk_config.data_poison_method=a3fl \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=3.0 \
    atk_config.poison_ratio=0.625 \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"5,4,3,2,1\" \
    num_rounds=200 \
    checkpoint=1000 \
    dir_tag=tiny_robust_aggregation_pgd


#### Model Replacement
python main.py -m -cn tiny \
    aggregator=unweighted_fedavg,coordinate_median,geometric_median,trimmed_mean,krum,multi_krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=tiny_multishot \
    atk_config.data_poison_method=pattern \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    num_rounds=200 \
    checkpoint=1000 \
    cuda_visible_devices=\"2,3,4,0,1\" \
    dir_tag=tiny_robust_aggregation_modelreplace && 
python main.py -m -cn tiny \
    aggregator=unweighted_fedavg,coordinate_median,geometric_median,trimmed_mean,krum,multi_krum,foolsgold,robustlr,norm_clipping,weakdp,fltrust,flare,bulyan \
    atk_config=tiny_multishot \
    atk_config.data_poison_method=distributed \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    num_rounds=200 \
    checkpoint=1000 \
    cuda_visible_devices=\"0,1,2,4,5\" \
    dir_tag=tiny_robust_aggregation_modelreplace