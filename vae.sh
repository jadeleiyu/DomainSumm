#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 nohup python -u vae.py --epochs_ext 5  --data nyt   > ~/PycharmProjects/E_Yue/outputs/vae_nyt.out & #garden-path
