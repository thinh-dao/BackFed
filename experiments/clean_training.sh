# *EMNIST
python main.py dataset=emnist_byclass model=mnistnet num_clients=3383 num_rounds=1000 no_attack=True cuda_visible_devices=\"1,4,5,6,7\" save_checkpoint=True

# *CIFAR10
python main.py dataset=cifar10 num_rounds=2000 model=resnet18 no_attack=True cuda_visible_devices=\"1,4,5,6,7\" save_checkpoint=True "save_model_rounds=[1000,1200,1400,1600,1800,2000]"