python main.py -m -cn cifar10 dir_tag=cifar_runtime_sequential training_mode=sequential no_attack=True num_rounds=5 num_clients_per_round=20,40,60,80,100 
python main.py -m -cn cifar10 dir_tag=cifar_runtime_parallel training_mode=parallel no_attack=True num_rounds=5 num_clients_per_round=20,40,60,80,100 cuda_visible_devices=\"1,2,3,4,5,6\" num_gpus=0.4 num_cpus=1

python main.py -m dir_tag=
