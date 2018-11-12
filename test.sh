#!/usr/bin/env bash
#CUDA_VISIBLE_DEVICES=4 nohup python -u main.py --epochs_ext 10 --ext_model gm --lr 5e-6 --log_file ../log/07_10_5_model > ~/PycharmProjects/E_Yue/outputs/18_07_10_t5.out &
#python -u test.py > lei_docs_exp.log &

CUDA_VISIBLE_DEVICES=1 nohup python -u test.py --data nyt >  ~/PycharmProjects/E_Yue/outputs/lstm_test.out &
