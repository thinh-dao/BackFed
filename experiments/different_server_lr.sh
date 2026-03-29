########################################### Different server learning rates
# eta=0.1
python main.py -m -cn cifar10 \
    aggregator=unweighted_fedavg \
    aggregator_config.unweighted_fedavg.eta=0.1 \
    checkpoint=checkpoints/CIFAR10_unweighted_fedavg_0.5/resnet18_round_1000_dir_0.9.pth \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern,distributed,cerberus,edge_case \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"5,1,2,3,4\" \
    num_rounds=200 \
    dir_tag=cifar10_robust_aggregation_slr_0.1 && \
python main.py -m -cn cifar10 \
    aggregator=coordinate_median \
    aggregator_config.coordinate_median.eta=0.n1 \
    checkpoint=checkpoints/CIFAR10_coordinate_median_0.5/resnet18_round_1000_dir_0.9.pth \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern,distributed,cerberus,edge_case \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"1,0,2,3,4\" \
    num_rounds=200 \
    dir_tag=cifar10_robust_aggregation_slr_0.1 && \
python main.py -m -cn cifar10 \
    aggregator=krum \
    aggregator_config.krum.eta=0.1 \
    checkpoint=checkpoints/CIFAR10_krum_0.5/resnet18_round_1000_dir_0.9.pth \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern,distributed,cerberus,edge_case \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"2,1,0,3,4\" \
    num_rounds=200 \
    dir_tag=cifar10_robust_aggregation_slr_0.1



python main.py -m -cn cifar10 \
    aggregator=weakdp \
    aggregator_config.weakdp.eta=0.1 \
    checkpoint=checkpoints/CIFAR10_weakdp_0.5/resnet18_round_1000_dir_0.9.pth \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern,distributed,cerberus,edge_case \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"3,2,1,0,4\" \
    num_rounds=200 \
    dir_tag=cifar10_robust_aggregation_slr_0.1 && \
python main.py -m -cn cifar10 \
    aggregator=fltrust \
    aggregator_config.fltrust.eta=0.1 \
    checkpoint=checkpoints/CIFAR10_fltrust_0.5/resnet18_round_1000_dir_0.9.pth \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern,distributed,cerberus,edge_case \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"4,3,2,1,0\" \
    num_rounds=200 \
    dir_tag=cifar10_robust_aggregation_slr_0.1 && \
python main.py -m -cn cifar10 \
    aggregator=robustlr \
    aggregator_config.robustlr.eta=0.1 \
    checkpoint=checkpoints/CIFAR10_robustlr_0.5/resnet18_round_1000_dir_0.9.pth \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern,distributed,cerberus,edge_case \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"4,3,2,1,0\" \
    num_rounds=200 \
    dir_tag=cifar10_robust_aggregation_slr_0.1


######### FEMNIST ##########
python main.py -m -cn femnist \
    aggregator=unweighted_fedavg \
    aggregator_config.unweighted_fedavg.eta=0.1 \
    checkpoint=checkpoints/FEMNIST_unweighted_fedavg_0.5/mnistnet_round_1000_dir_0.9.pth \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=pattern,distributed,cerberus,edge_case \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"0,1,2,3,4\" \
    num_rounds=200 \
    dir_tag=femnist_robust_aggregation_slr_0.1 && \
python main.py -m -cn femnist \
    aggregator=coordinate_median \
    aggregator_config.coordinate_median.eta=0.1 \
    checkpoint=checkpoints/FEMNIST_coordinate_median_0.5/mnistnet_round_1000_dir_0.9.pth \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=pattern,distributed,cerberus,edge_case \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"1,0,2,3,4\" \
    num_rounds=200 \
    dir_tag=femnist_robust_aggregation_slr_0.1 && \
python main.py -m -cn femnist \
    aggregator=krum \
    aggregator_config.krum.eta=0.1 \
    checkpoint=checkpoints/FEMNIST_krum_0.5/mnistnet_round_1000_dir_0.9.pth \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=pattern,distributed,cerberus,edge_case \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"2,1,0,3,4\" \
    num_rounds=200 \
    dir_tag=femnist_robust_aggregation_slr_0.1

    
python main.py -m -cn femnist \
    aggregator=weakdp \
    aggregator_config.weakdp.eta=0.1 \
    checkpoint=checkpoints/FEMNIST_weakdp_0.5/mnistnet_round_1000_dir_0.9.pth \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=pattern,distributed,cerberus,edge_case \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"3,2,1,0,4\" \
    num_rounds=200 \
    dir_tag=femnist_robust_aggregation_slr_0.1 && \ 
python main.py -m -cn femnist \
    aggregator=fltrust \
    aggregator_config.fltrust.eta=0.1 \
    checkpoint=checkpoints/FEMNIST_fltrust_0.5/mnistnet_round_1000_dir_0.9.pth \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=pattern,distributed,cerberus,edge_case \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"4,3,2,1,0\" \
    num_rounds=200 \
    dir_tag=femnist_robust_aggregation_slr_0.1 && \
python main.py -m -cn femnist \
    aggregator=robustlr \
    aggregator_config.robustlr.eta=0.1 \
    checkpoint=checkpoints/FEMNIST_robustlr_0.5/mnistnet_round_1000_dir_0.9.pth \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=pattern,distributed,cerberus,edge_case \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"2,3,1,0,4,5\" \
    num_rounds=200 \
    dir_tag=femnist_robust_aggregation_slr_0.1




# eta=1.0
python main.py -m -cn cifar10 \
    aggregator=unweighted_fedavg \
    aggregator_config.unweighted_fedavg.eta=1.0 \
    checkpoint=checkpoints/CIFAR10_unweighted_fedavg_0.5/resnet18_round_1000_dir_0.9.pth \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern,distributed,cerberus,edge_case \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"0,1,2,3,4\" \
    num_rounds=200 \
    dir_tag=cifar10_robust_aggregation_slr_1.0

python main.py -m -cn cifar10 \
    aggregator=coordinate_median \
    aggregator_config.coordinate_median.eta=1.0 \
    checkpoint=checkpoints/CIFAR10_coordinate_median_0.5/resnet18_round_1000_dir_0.9.pth \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern,distributed,cerberus,edge_case \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"1,0,2,3,4\" \
    num_rounds=200 \
    dir_tag=cifar10_robust_aggregation_slr_1.0

python main.py -m -cn cifar10 \
    aggregator=krum \
    aggregator_config.krum.eta=1.0 \
    checkpoint=checkpoints/CIFAR10_krum_0.5/resnet18_round_1000_dir_0.9.pth \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern,distributed,cerberus,edge_case \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"2,1,0,3,4\" \
    num_rounds=200 \
    dir_tag=cifar10_robust_aggregation_slr_1.0

python main.py -m -cn cifar10 \
    aggregator=weakdp \
    aggregator_config.weakdp.eta=1.0 \
    checkpoint=checkpoints/CIFAR10_weakdp_0.5/resnet18_round_1000_dir_0.9.pth \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern,distributed,cerberus,edge_case \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"3,2,1,0,4\" \
    num_rounds=200 \
    dir_tag=cifar10_robust_aggregation_slr_1.0

python main.py -m -cn cifar10 \
    aggregator=fltrust \
    aggregator_config.fltrust.eta=1.0 \
    checkpoint=checkpoints/CIFAR10_fltrust_0.5/resnet18_round_1000_dir_0.9.pth \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern,distributed,cerberus,edge_case \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"4,3,2,1,0\" \
    num_rounds=200 \
    dir_tag=cifar10_robust_aggregation_slr_1.0

python main.py -m -cn cifar10 \
    aggregator=robustlr \
    aggregator_config.robustlr.eta=1.0 \
    checkpoint=checkpoints/CIFAR10_robustlr_0.5/resnet18_round_1000_dir_0.9.pth \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern,distributed,cerberus,edge_case \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"4,3,2,1,0\" \
    num_rounds=200 \
    dir_tag=cifar10_robust_aggregation_slr_1.0


######### FEMNIST ##########
python main.py -m -cn femnist \
    aggregator=unweighted_fedavg \
    aggregator_config.unweighted_fedavg.eta=1.0 \
    checkpoint=checkpoints/FEMNIST_unweighted_fedavg_0.5/mnistnet_round_1000_dir_0.9.pth \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=pattern,distributed,cerberus,edge_case \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"0,1,2,3,4\" \
    num_rounds=200 \
    dir_tag=femnist_robust_aggregation_slr_1.0 &&

python main.py -m -cn femnist \
    aggregator=coordinate_median \
    aggregator_config.coordinate_median.eta=1.0 \
    checkpoint=checkpoints/FEMNIST_coordinate_median_0.5/mnistnet_round_1000_dir_0.9.pth \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=pattern,distributed,cerberus,edge_case \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"1,0,2,3,4\" \
    num_rounds=200 \
    dir_tag=femnist_robust_aggregation_slr_1.0

python main.py -m -cn femnist \
    aggregator=krum \
    aggregator_config.krum.eta=1.0 \
    checkpoint=checkpoints/FEMNIST_krum_0.5/mnistnet_round_1000_dir_0.9.pth \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=pattern,distributed,cerberus,edge_case \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"2,1,0,3,4\" \
    num_rounds=200 \
    dir_tag=femnist_robust_aggregation_slr_1.0

python main.py -m -cn femnist \
    aggregator=weakdp \
    aggregator_config.weakdp.eta=1.0 \
    checkpoint=checkpoints/FEMNIST_weakdp_0.5/mnistnet_round_1000_dir_0.9.pth \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=pattern,distributed,cerberus,edge_case \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"3,2,1,0,4\" \
    num_rounds=200 \
    dir_tag=femnist_robust_aggregation_slr_1.0

python main.py -m -cn femnist \
    aggregator=fltrust \
    aggregator_config.fltrust.eta=1.0 \
    checkpoint=checkpoints/FEMNIST_fltrust_0.5/mnistnet_round_1000_dir_0.9.pth \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=pattern,distributed,cerberus,edge_case \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"4,3,2,1,0\" \
    num_rounds=200 \
    dir_tag=femnist_robust_aggregation_slr_1.0

python main.py -m -cn femnist \
    aggregator=robustlr \
    aggregator_config.robustlr.eta=1.0 \
    checkpoint=checkpoints/FEMNIST_robustlr_0.5/mnistnet_round_1000_dir_0.9.pth \
    atk_config=femnist_multishot \
    atk_config.data_poison_method=pattern,distributed,cerberus,edge_case \
    atk_config.poison_start_round=1000 \
    atk_config.poison_end_round=1200 \
    cuda_visible_devices=\"2,3,1,0,4,5\" \
    num_rounds=200 \
    dir_tag=femnist_robust_aggregation_slr_1.0