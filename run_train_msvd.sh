#!/usr/bin/env zsh
CUDA_VISIBLE_DEVICES=1 python train.py --dataset=msvd --model=RMN \
 --result_dir=results/msvd/2222222 --use_lin_loss \
 --learning_rate_decay --learning_rate_decay_every=10 \
 --learning_rate_decay_rate=10 \
 --use_loc --use_rel --use_func \
 --learning_rate=1e-4 --attention=soft \
 --hidden_size=512 --att_size=512 \
 --train_batch_size=64 --test_batch_size=32 --beam_size=5