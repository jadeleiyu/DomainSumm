#!/usr/bin/env bash
#CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --epochs_ext 10 --ext_model gm  --num_topics 10  > ~/PycharmProjects/E_Yue/outputs/gm_10.out & #dp-gpu2
#CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --epochs_ext 10 --ext_model fs  --num_topics 10 --resume --start_epoch 4  > ~/PycharmProjects/E_Yue/outputs/fs_10.out &
CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --epochs_ext 15 --ext_model dm  --num_topics 10 --resume --start_epoch 8  > ~/PycharmProjects/E_Yue/outputs/dm_10.out & #agent-server-1
#CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --epochs_ext 10 --ext_model ps  --num_topics 10 --resume --start_epoch 3  > ~/PycharmProjects/E_Yue/outputs/ps_10.out &



