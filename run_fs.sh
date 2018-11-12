#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --epochs_ext 15 --ext_model fs  --num_topics 100 --category_size 100 --data nyt  > ~/PycharmProjects/E_Yue/outputs/nyt_fs_100.out &  #
CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --epochs_ext 15 --ext_model fs  --num_topics 500 --category_size 500 --data nyt  > ~/PycharmProjects/E_Yue/outputs/nyt_fs_500.out &  #
#CUDA_VISIBLE_DEVICES=3 nohup python -u main.py --epochs_ext 15 --ext_model fs  --num_topics 1000 --category_size 1000 --data nyt  > ~/PycharmProjects/E_Yue/outputs/nyt_fs_1000.out &