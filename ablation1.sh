
CUDA_VISIBLE_DEVICES=0 python ablation.py --config_file config8.yaml --nb_exp_reps 3 --nb_devices 1 --lr 0.1 --model cifarcnn --pruning_type server_pruning --plan_type reverse --decay_type linear

