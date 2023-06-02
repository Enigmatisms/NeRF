python3 ./train.py -s -w --dataset_name lego-test --opt_mode native
python3 ./ddp_train.py -s -w --dataset_name lego-test --opt_mode native --nr 0 --nodes 2