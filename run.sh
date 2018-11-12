#!/usr/bin/env bash
smart-dispatch --queue=gpu_1 --walltime=24:00:00 launch python -u main_ext.py --rouge_metric [lf avg_f] --batch_size [5 20] --rl_baseline_method=[greedy global_avg batch_avg] --oracle_l=3 --rl_loss_method=[1 2] --load_ext
smart-dispatch --queue=gpu_1 --walltime=24:00:00 launch python -u main_ext.py --rouge_metric avg_f --batch_size 20 --rl_baseline_method=None --oracle_l=3 --rl_loss_method=2 --device 0 --load_ext
smart-dispatch --queue=gpu_1 --walltime=24:00:00 launch python -u main_ext.py --rouge_metric avg_f --batch_size 20 --rl_baseline_method=batch_avg --oracle_l=3 --rl_loss_method=2 --train_example_quota [5000 10000 50000 100000] --device 0 --load_ext
smart-dispatch --queue=gpu_1 --walltime=24:00:00 launch python -u main_ext.py --rouge_metric avg_f --batch_size 20 --rl_baseline_method=batch_avg --oracle_l=3 --rl_loss_method=2 --data_dir ../data/DM_pickle_data/chunked/ --length_limit [75 275] --device 0 --load_ext
