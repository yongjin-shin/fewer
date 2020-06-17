
for model in mnistcnn cifarcnn testcnn
do
CUDA_VISIBLE_DEVICES=0 python main.py --config_file config.yaml --nb_exp_reps 1 --nb_devices 100 --lr 0.15 --model ${model} --pruning_type baseline
CUDA_VISIBLE_DEVICES=0 python main.py --config_file config.yaml --nb_exp_reps 1 --nb_devices 100 --lr 0.15 --model ${model} --pruning_type server_pruning
CUDA_VISIBLE_DEVICES=0 python main.py --config_file config.yaml --nb_exp_reps 1 --nb_devices 100 --lr 0.15 --model ${model} --pruning_type server_pruning --plan_type reverse
CUDA_VISIBLE_DEVICES=0 python main.py --config_file config.yaml --nb_exp_reps 1 --nb_devices 100 --lr 0.15 --model ${model} --pruning_type local_pruning
CUDA_VISIBLE_DEVICES=0 python main.py --config_file config.yaml --nb_exp_reps 1 --nb_devices 100 --lr 0.15 --model ${model} --pruning_type local_pruning_half
done

