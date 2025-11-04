######## Anomaly Detection defense against multishot attack ########

############## CIFAR10 ################
# One-line argument using Hydra --multirun 
# For efficiency, you may run attacks in different processes

#### Baseline
python main.py -m -cn cifar10 \
    aggregator=unweighted_fedavg,flame,rflbat,feddlad \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.poison_ratio=0.5 \
    cuda_visible_devices=\"0,1,2,3,4\" \
    num_rounds=200 \
    dir_tag=cifar10_anomaly_detection_baseline &&
python main.py -m -cn cifar10 \
    aggregator=unweighted_fedavg,deepsight,flame,rflbat,multi_metrics,feddlad \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern \
    cuda_visible_devices=\"4,2,1,3,0\" \
    num_rounds=200 \
    dir_tag=cifar10_anomaly_detection_baseline &&
python main.py -m -cn cifar10 \
    aggregator=unweighted_fedavg,ad_multi_krum,alignins,deepsight,flame,rflbat,multi_metrics,feddlad \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=a3fl \
    cuda_visible_devices=\"1,2,3,0,5\" \
    num_rounds=200 \
    dir_tag=cifar10_anomaly_detection_baseline &&
python main.py -m -cn cifar10 \
    aggregator=unweighted_fedavg,ad_multi_krum,alignins,deepsight,flame,rflbat,multi_metrics,feddlad \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=cerberus \
    atk_config.model_poison_method=cerberus \
    cuda_visible_devices=\"2,3,4,5\" \
    num_rounds=200 \
    dir_tag=cifar10_anomaly_detection_baseline

#### PGD attack
python main.py -m -cn cifar10 \
    aggregator=unweighted_fedavg,flame,rflbat,feddlad \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=3 \
    atk_config.poison_ratio=0.5 \
    cuda_visible_devices=\"0,1,2,3\" \
    num_rounds=200 \
    dir_tag=cifar10_anomaly_detection_pgd &&
python main.py -m -cn cifar10 \
    aggregator=unweighted_fedavg,alignins,deepsight,flame,rflbat,multi_metrics,feddlad \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=3 \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=200 \
    dir_tag=cifar10_anomaly_detection_pgd &&
python main.py -m -cn cifar10 \
    aggregator=unweighted_fedavg,ad_multi_krum,alignins,deepsight,flame,rflbat,multi_metrics,feddlad \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=a3fl \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=3 \
    cuda_visible_devices=\"5,4,3,2,1,0\" \
    num_rounds=200 \
    dir_tag=cifar10_anomaly_detection_pgd

#### Model Replacement
python main.py -m -cn cifar10 \
    aggregator=unweighted_fedavg,ad_multi_krum,alignins,deepsight,flame,rflbat,multi_metrics,feddlad \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=200 \
    dir_tag=cifar10_anomaly_detection_modelreplace && 
python main.py -m -cn cifar10 \
    aggregator=unweighted_fedavg,alignins,deepsight,flame,rflbat,multi_metrics,feddlad \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=200 \
    dir_tag=cifar10_anomaly_detection_modelreplace &&
python main.py -m -cn cifar10 \
    aggregator=ad_multi_krum,unweighted_fedavg,alignins,deepsight,flame,rflbat,multi_metrics,feddlad \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=distributed \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=200 \
    dir_tag=cifar10_anomaly_detection_modelreplace

###################
# FEMNIST
python main.py -m -cn femnist \
    aggregator=unweighted_fedavg,flame,rflbat,feddlad \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.poison_ratio=0.5 \
    cuda_visible_devices=\"0,1,2,3,4\" \
    num_rounds=200 \
    dir_tag=femnist_anomaly_detection_baseline &&
python main.py -m -cn femnist \
    aggregator=unweighted_fedavg,deepsight,flame,rflbat,multi_metrics,feddlad \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=pattern \
    cuda_visible_devices=\"4,2,1,3,0\" \
    num_rounds=200 \
    dir_tag=femnist_anomaly_detection_baseline &&
python main.py -m -cn femnist \
    aggregator=unweighted_fedavg,ad_multi_krum,alignins,deepsight,flame,rflbat,multi_metrics,feddlad \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=a3fl \
    cuda_visible_devices=\"1,2,3,0,5\" \
    num_rounds=200 \
    dir_tag=femnist_anomaly_detection_baseline &&
python main.py -m -cn femnist \
    aggregator=unweighted_fedavg,ad_multi_krum,alignins,deepsight,flame,rflbat,multi_metrics,feddlad \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=cerberus \
    atk_config.model_poison_method=cerberus \
    cuda_visible_devices=\"4,3,2,1,0\" \
    num_rounds=200 \
    dir_tag=femnist_anomaly_detection_baseline


#### PGD attack
python main.py -m -cn femnist \
    aggregator=unweighted_fedavg,flame,rflbat,feddlad \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=3 \
    atk_config.poison_ratio=0.5 \
    cuda_visible_devices=\"0,1,2,3\" \
    num_rounds=200 \
    dir_tag=femnist_anomaly_detection_pgd &&
python main.py -m -cn femnist \
    aggregator=unweighted_fedavg,alignins,deepsight,flame,rflbat,multi_metrics,feddlad \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=pattern \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=3 \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=200 \
    dir_tag=femnist_anomaly_detection_pgd &&
python main.py -m -cn femnist \
    aggregator=unweighted_fedavg,ad_multi_krum,alignins,deepsight,flame,rflbat,multi_metrics,feddlad \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=a3fl \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=3 \
    cuda_visible_devices=\"5,4,3,2,1,0\" \
    num_rounds=200 \
    dir_tag=femnist_anomaly_detection_pgd

#### Model Replacement
python main.py -m -cn femnist \
    aggregator=unweighted_fedavg,ad_multi_krum,alignins,deepsight,flame,rflbat,multi_metrics,feddlad \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=pattern \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=200 \
    dir_tag=femnist_anomaly_detection_modelreplace && 
python main.py -m -cn femnist \
    aggregator=unweighted_fedavg,alignins,deepsight,flame,rflbat,multi_metrics,feddlad \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=200 \
    dir_tag=femnist_anomaly_detection_modelreplace &&
python main.py -m -cn femnist \
    aggregator=ad_multi_krum,unweighted_fedavg,alignins,deepsight,flame,rflbat,multi_metrics,feddlad \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=distributed \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=200 \
    dir_tag=femnist_anomaly_detection_modelreplace



####################### FIXED AlignIns and DeepSight ###############################
python main.py -m -cn cifar10 \
    aggregator=alignins,deepsight \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.poison_ratio=0.5 \
    cuda_visible_devices=\"0,1,2,3,4\" \
    num_rounds=200 \
    dir_tag=cifar10_anomaly_detection_baseline &&
python main.py -m -cn cifar10 \
    aggregator=alignins,deepsight  \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern \
    cuda_visible_devices=\"4,2,1,3,0\" \
    num_rounds=200 \
    dir_tag=cifar10_anomaly_detection_baseline &&
python main.py -m -cn cifar10 \
    aggregator=alignins,deepsight \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=a3fl \
    cuda_visible_devices=\"1,2,3,0,5\" \
    num_rounds=200 \
    dir_tag=cifar10_anomaly_detection_baseline &&
python main.py -m -cn cifar10 \
    aggregator=alignins,deepsight  \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=cerberus \
    atk_config.model_poison_method=cerberus \
    cuda_visible_devices=\"4,3,2,1,0\" \
    num_rounds=200 \
    dir_tag=cifar10_anomaly_detection_baseline


#### PGD attack
python main.py -m -cn cifar10 \
    aggregator=alignins,deepsight  \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=3 \
    atk_config.poison_ratio=0.5 \
    cuda_visible_devices=\"0,1,2,3\" \
    num_rounds=200 \
    dir_tag=cifar10_anomaly_detection_pgd &&
python main.py -m -cn cifar10 \
    aggregator=alignins,deepsight  \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=3 \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=200 \
    dir_tag=cifar10_anomaly_detection_pgd &&
python main.py -m -cn cifar10 \
    aggregator=alignins,deepsight  \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=a3fl \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=3 \
    cuda_visible_devices=\"5,4,3,2,1,0\" \
    num_rounds=200 \
    dir_tag=cifar10_anomaly_detection_pgd

#### Model Replacement
python main.py -m -cn cifar10 \
    aggregator=alignins,deepsight  \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    cuda_visible_devices=\"7,6,3,5,0\" \
    num_rounds=200 \
    dir_tag=cifar10_anomaly_detection_modelreplace && 
python main.py -m -cn cifar10 \
    aggregator=alignins,deepsight  \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    cuda_visible_devices=\"7,6,3,5,0\" \
    num_rounds=200 \
    dir_tag=cifar10_anomaly_detection_modelreplace &&
python main.py -m -cn cifar10 \
    aggregator=alignins,deepsight  \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=distributed \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    cuda_visible_devices=\"7,6,3,5,0\" \
    num_rounds=200 \
    dir_tag=cifar10_anomaly_detection_modelreplace

###################
# FEMNIST
python main.py -m -cn femnist \
    aggregator=alignins,deepsight  \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.poison_ratio=0.5 \
    cuda_visible_devices=\"0,1,2,3,4\" \
    num_rounds=200 \
    dir_tag=femnist_anomaly_detection_baseline &&
python main.py -m -cn femnist \
    aggregator=alignins,deepsight  \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=pattern \
    cuda_visible_devices=\"4,2,1,3,0\" \
    num_rounds=200 \
    dir_tag=femnist_anomaly_detection_baseline &&
python main.py -m -cn femnist \
    aggregator=alignins,deepsight  \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=a3fl \
    cuda_visible_devices=\"1,2,3,0,5\" \
    num_rounds=200 \
    dir_tag=femnist_anomaly_detection_baseline &&
python main.py -m -cn femnist \
    aggregator=alignins,deepsight  \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=cerberus \
    atk_config.model_poison_method=cerberus \
    cuda_visible_devices=\"4,3,2,1,0\" \
    num_rounds=200 \
    dir_tag=femnist_anomaly_detection_baseline


#### PGD attack
python main.py -m -cn femnist \
    aggregator=alignins,deepsight  \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=3 \
    atk_config.poison_ratio=0.5 \
    cuda_visible_devices=\"0,1,2,3\" \
    num_rounds=200 \
    dir_tag=femnist_anomaly_detection_pgd &&
python main.py -m -cn femnist \
    aggregator=alignins,deepsight  \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=pattern \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=3 \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=200 \
    dir_tag=femnist_anomaly_detection_pgd &&
python main.py -m -cn femnist \
    aggregator=alignins,deepsight \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=a3fl \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=3 \
    cuda_visible_devices=\"5,4,3,2,1,0\" \
    num_rounds=200 \
    dir_tag=femnist_anomaly_detection_pgd

#### Model Replacement
python main.py -m -cn femnist \
    aggregator=alignins,deepsight  \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=pattern \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=200 \
    dir_tag=femnist_anomaly_detection_modelreplace && 
python main.py -m -cn femnist \
    aggregator=alignins,deepsight  \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=200 \
    dir_tag=femnist_anomaly_detection_modelreplace &&
python main.py -m -cn femnist \
    aggregator=alignins,deepsight  \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=distributed \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    cuda_visible_devices=\"1,2,3,5,0\" \
    num_rounds=200 \
    dir_tag=femnist_anomaly_detection_modelreplace



###### Fix Multi-Metrics Server only ##########
#### Baseline
python main.py -m -cn cifar10 \
    aggregator=multi_metrics \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.poison_ratio=0.5 \
    cuda_visible_devices=\"2,3,4,5\" \
    num_rounds=200 \
    dir_tag=cifar10_ad_multimetrics_baseline &&
python main.py -m -cn cifar10 \
    aggregator=multi_metrics \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern \
    cuda_visible_devices=\"3,2,4,5\" \
    num_rounds=200 \
    dir_tag=cifar10_ad_multimetrics_baseline &&
python main.py -m -cn cifar10 \
    aggregator=multi_metrics \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=a3fl \
    cuda_visible_devices=\"5,4,3,2\" \
    num_rounds=200 \
    dir_tag=cifar10_ad_multimetrics_baseline &&
python main.py -m -cn cifar10 \
    aggregator=multi_metrics \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=cerberus \
    atk_config.model_poison_method=cerberus \
    cuda_visible_devices=\"2,3,4,5\" \
    num_rounds=200 \
    dir_tag=cifar10_ad_multimetrics_baseline

#### PGD attack
python main.py -m -cn cifar10 \
    aggregator=multi_metrics \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=3 \
    atk_config.poison_ratio=0.5 \
    cuda_visible_devices=\"2,4,3,5\" \
    num_rounds=200 \
    dir_tag=cifar10_ad_multimetrics_pgd &&
python main.py -m -cn cifar10 \
    aggregator=multi_metrics \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=3 \
    cuda_visible_devices=\"4,3,2,5\" \
    num_rounds=200 \
    dir_tag=cifar10_ad_multimetrics_pgd &&
python main.py -m -cn cifar10 \
    aggregator=multi_metrics \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=a3fl \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=3 \
    cuda_visible_devices=\"5,3,4,2\" \
    num_rounds=200 \
    dir_tag=cifar10_ad_multimetrics_pgd

#### Model Replacement
python main.py -m -cn femnist \
    aggregator=multi_metrics \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=pattern \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    cuda_visible_devices=\"2,3,4,5\" \
    num_rounds=200 \
    dir_tag=femnist_ad_multimetrics_modelreplace && 
python main.py -m -cn femnist \
    aggregator=multi_metrics \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    cuda_visible_devices=\"4,2,3,5\" \
    num_rounds=200 \
    dir_tag=femnist_ad_multimetrics_modelreplace &&
python main.py -m -cn femnist \
    aggregator=multi_metrics \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=distributed \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    cuda_visible_devices=\"3,4,2,5\" \
    num_rounds=200 \
    dir_tag=femnist_ad_multimetrics_modelreplace

### FEMNIST
python main.py -m -cn femnist \
    aggregator=multi_metrics \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.poison_ratio=0.5 \
    cuda_visible_devices=\"2,3,4,5\" \
    num_rounds=200 \
    dir_tag=femnist_anomaly_detection_baseline &&
python main.py -m -cn femnist \
    aggregator=multi_metrics \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=pattern \
    cuda_visible_devices=\"3,2,4,5\" \
    num_rounds=200 \
    dir_tag=femnist_anomaly_detection_baseline &&
python main.py -m -cn femnist \
    aggregator=multi_metrics \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=a3fl \
    cuda_visible_devices=\"5,4,3,2\" \
    num_rounds=200 \
    dir_tag=femnist_anomaly_detection_baseline &&
python main.py -m -cn femnist \
    aggregator=multi_metrics \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=cerberus \
    atk_config.model_poison_method=cerberus \
    cuda_visible_devices=\"2,3,4,5\" \
    num_rounds=200 \
    dir_tag=femnist_ad_multimetrics_baseline

#### PGD attack
python main.py -m -cn femnist \
    aggregator=multi_metrics \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=3 \
    atk_config.poison_ratio=0.5 \
    cuda_visible_devices=\"2,4,3,5\" \
    num_rounds=200 \
    dir_tag=femnist_ad_multimetrics_pgd &&
python main.py -m -cn femnist \
    aggregator=multi_metrics \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=pattern \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=3 \
    cuda_visible_devices=\"4,3,2,5\" \
    num_rounds=200 \
    dir_tag=femnist_ad_multimetrics_pgd &&
python main.py -m -cn femnist \
    aggregator=multi_metrics \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=a3fl \
    atk_config.poisoned_is_projection=True \
    atk_config.poisoned_projection_eps=3 \
    cuda_visible_devices=\"5,3,4,2\" \
    num_rounds=200 \
    dir_tag=femnist_ad_multimetrics_pgd

#### Model Replacement
python main.py -m -cn femnist \
    aggregator=multi_metrics \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=pattern \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    cuda_visible_devices=\"2,3,4,5\" \
    num_rounds=200 \
    dir_tag=femnist_ad_multimetrics_modelreplace && 
python main.py -m -cn femnist \
    aggregator=multi_metrics \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    cuda_visible_devices=\"4,2,3,5\" \
    num_rounds=200 \
    dir_tag=femnist_ad_multimetrics_modelreplace &&
python main.py -m -cn femnist \
    aggregator=multi_metrics \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=distributed \
    atk_config.scale_poison=True \
    atk_config.scale_factor=10 \
    cuda_visible_devices=\"3,4,2,5\" \
    num_rounds=200 \
    dir_tag=femnist_ad_multimetrics_modelreplace