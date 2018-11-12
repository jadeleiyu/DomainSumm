#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --epochs_ext 10 --ext_model fs --num_topics 0   > ~/PycharmProjects/E_Yue/outputs/fs_0.out & #
#CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --epochs_ext 15 --resume --start_epoch 9 --ext_model dm --num_topics 0   > ~/PycharmProjects/E_Yue/outputs/dm_0.out & #agent-server-1

nohup python -u test.py >> cnn_docs_samples.out &