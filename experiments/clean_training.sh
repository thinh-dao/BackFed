# *EMNIST
python main.py dataset=emnist_byclass model=mnistnet num_clients=3383 num_clients_per_round=30 num_rounds=1000 no_attack=True cuda_visible_devices=\"1,5,7,2,4\" "save_checkpoint_rounds=[200,400,600,800,1000,1200,1400,1600,1800,2000]" test_batch_size=5000 test_every=5 num_workers=8 save_checkpoint=True checkpoint=Null

# *FEMNIST
python main.py -cn femnist model=mnistnet num_clients_per_round=30 num_rounds=2000 no_attack=True cuda_visible_devices=\"1,5,7,2,4\" "save_checkpoint_rounds=[200,400,600,800,1000,1200,1400,1600,1800,2000]" test_batch_size=5000 test_every=5 num_workers=8 save_checkpoint=True checkpoint=Null dir_tag=clean_training

# *CIFAR10
python main.py aggregator=unweighted_fedavg dataset=cifar10 num_rounds=2000 num_clients=100 model=resnet18 no_attack=True cuda_visible_devices=\"1,2,3,4,5\" num_gpus=0.5 save_checkpoint=True "save_checkpoint_rounds=[200,400,600,800,1000,1200,1400,1600,1800,2000]" checkpoint=Null
python main.py aggregator=weighted_fedavg dataset=cifar10 num_rounds=2000 num_clients=100 model=resnet18 no_attack=True cuda_visible_devices=\"1,2,3,4,5\" num_gpus=0.5 save_checkpoint=True "save_checkpoint_rounds=[200,400,600,800,1000,1200,1400,1600,1800,2000]" checkpoint=Null

# *TINYIMAGENET
python main.py -cn tiny dataset=tinyimagenet num_rounds=2000 test_batch_size=1024 num_clients_per_round=10 model=resnet18 no_attack=True cuda_visible_devices=\"1,5,7,2,4\" save_checkpoint=True "save_checkpoint_rounds=[1000,1200,1400,1600,1800,2000]" checkpoint=Null test_every=2 dir_tag=clean_training
python main.py -cn tiny dataset=tinyimagenet num_rounds=2000 test_batch_size=1024 num_clients_per_round=20 model=vgg11 no_attack=True cuda_visible_devices=\"1,5,7,2,4\" save_checkpoint=True "save_checkpoint_rounds=[1000,1200,1400,1600,1800,2000]" checkpoint=Null test_every=2 dir_tag=clean_training

# *Sentiment140
python main.py -cn sentiment140 num_rounds=2000 no_attack=True save_checkpoint=True cuda_visible_devices=\"0,1,2,3\" "save_checkpoint_rounds=[250,500,750,1000,1250,1500,1750,2000]"

# *Reddit
python main.py -cn reddit num_rounds=5000 no_attack=True save_checkpoint=True cuda_visible_devices=\"4,5,6,7\" "save_checkpoint_rounds=[250,500,750,1000,1250,1500,1750,2000,2500,3000,3500,4000,4500,5000]"


python main.py -cn femnist model=mnistnet num_clients_per_round=30 num_rounds=1000 no_attack=True cuda_visible_devices=\"1,5,7,2,4\" "save_checkpoint_rounds=[200,400,600,800,1000,1200,1400,1600,1800,2000]" test_batch_size=5000 test_every=5 num_workers=8 save_checkpoint=True checkpoint=Null dir_tag=clean_training && \
python main.py -m -cn femnist \
    aggregator=unweighted_fedavg \
    atk_config=femnist_multishot \
    atk_config.model_poison_method=anticipate,neurotoxin,chameleon,base \
    atk_config.data_poison_method=pattern \
    checkpoint=1000 \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=femnist_durability_enhanced \
    cuda_visible_devices=\"0,1,2,3,4\" \