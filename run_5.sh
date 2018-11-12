#!/usr/bin/env bash
#CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --epochs_ext 10 --ext_model gm --num_topics 5 --resume --start_epoch 3  > ~/PycharmProjects/E_Yue/outputs/gm_5.out &                      #
#CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --epochs_ext 40 --ext_model dm --num_topics 5 --resume --start_epoch 19 --data nyt >> ~/PycharmProjects/E_Yue/outputs/nyt_dm_5.out &      # agent2 running
#CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --epochs_ext 10 --ext_model ps --num_topics 5  ~/PycharmProjects/E_Yue/outputs/ps_5.out &                                                 #
#CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --epochs_ext 10 --ext_model fs --num_topics 5 --resume --start_epoch 1  > ~/PycharmProjects/E_Yue/outputs/fs_5.out &                      #

CUDA_VISIBLE_DEVICES=3 nohup python -u main.py --epochs_ext 15 --ext_model gm  --num_topics 5 --data cnn_dm  > ~/PycharmProjects/E_Yue/outputs/cnn_gm_5.out &         # garden-path running
#CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --epochs_ext 15 --ext_model dm  --num_topics 5 --data cnn_dm  > ~/PycharmProjects/E_Yue/outputs/cnn_dm_5.out &          # agent1 running
#CUDA_VISIBLE_DEVICES=3 nohup python -u main.py --epochs_ext 15 --ext_model ps  --num_topics 5 --data cnn_dm  > ~/PycharmProjects/E_Yue/outputs/cnn_ps_5.out &          # agent1 running
#CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --epochs_ext 10 --ext_model fs  --num_topics 5 --data cnn_dm  > ~/PycharmProjects/E_Yue/outputs/cnn_fs_5.out &          # agent1 running


#CUDA_VISIBLE_DEVICES=1 nohup python -u test.py --ext_model gm --num_topics 5 --data nyt > ~/PycharmProjects/E_Yue/outputs/nyt_gm_5_eval.out &  #