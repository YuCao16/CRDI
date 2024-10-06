python scripts/fs_gradient_train.py \
	--t_start 5 --t_end 20 --num_gradient 15 \
	--random_q_noise True --epochs 120 --learning_rate 0.05 \
	--category babies --print_config True

python scripts/fs_gradient_evaluate.py \
	--t_start 5 --t_end 20 --num_gradient 15 \
	--anneal_ptb True --anneal_scale 0.05 \
	--use_x_0 True --random_q_noise True --print_config True \
	--category babies --num_evaluate 5000 --lpips_cluster_size 50 \
	--experiment_gradient_path checkpoints/model_babies.pth
