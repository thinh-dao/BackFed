# *EMNIST
python main.py dataset=emnist_byclass model=mnistnet num_clients=3383 num_clients_per_round=30 num_rounds=1000 no_attack=True cuda_visible_devices=\"1,5,7,2,4\" "save_model_rounds=[200,400,600,800,1000,1200,1400,1600,1800,2000]" test_batch_size=5000 test_every=5 num_workers=8 save_checkpoint=True checkpoint=Null

# *CIFAR10
python main.py dataset=cifar10 num_rounds=2000 num_clients=100 model=resnet18 no_attack=True cuda_visible_devices=\"1,4,5,6,7\" save_checkpoint=True "save_model_rounds=[200,400,600,800,1000,1200,1400,1600,1800,2000]" checkpoint=Null

# *TINYIMAGENET
python main.py dataset=tinyimagenet client_config.local_epochs=5 num_rounds=3000 num_clients=200 test_batch_size=1024 test_every=5 num_clients_per_round=10 model=resnet18 no_attack=True cuda_visible_devices=\"1,5,7,2,4\" save_checkpoint=True "save_model_rounds=[200,400,600,800,1000]" checkpoint=Null pretrain_model_path=IMAGENET1K_V1

# *Sentiment140
python main.py -cn sentiment140 num_rounds=500 no_attack=True "save_model_rounds=[200,400,600,800,1000,1250,1500,1750,2000]" cuda_visible_devices=\"0,1,2,3,4,5,6,7\"

# *Reddit
python main.py -cn reddit num_rounds=2000 no_attack=True cuda_visible_devices=\"1,2,3,4,5,0,7\" save_checkpoint=True "save_model_rounds=[250,500,750,1000,1250,1500,1750,2000]"