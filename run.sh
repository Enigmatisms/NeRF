CUDA_VISIBLE_DEVICES=0 python3 ./train.py -s -w --dataset_name lego --opt_mode native
CUDA_VISIBLE_DEVICES=0 python3 ./ddp_train.py -s -w --dataset_name lego --opt_mode native --nr 0 --nodes 4
CUDA_VISIBLE_DEVICES=0 python3 ./ddp_train.py -s -w --dataset_name lego --opt_mode native --nr 0 --nodes 4 --epochs 500 --sample_ray_num 2048 --lr 3e-4 --output_time 50