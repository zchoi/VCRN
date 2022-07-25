#!/usr/bin/env zsh
CUDA_VISIBLE_DEVICES=3 python train.py --dataset=vatex --model=RMN \
 --result_dir=results/vatex/trainval_vatex --use_lin_loss \
 --learning_rate_decay --learning_rate_decay_every=5 \
 --learning_rate_decay_rate=3 \
 --use_loc --use_rel --use_func \
 --learning_rate=1e-4 --attention=myatt \
 --hidden_size=1024  --att_size=1024 \
 --train_batch_size=128 --test_batch_size=32 --beam_size=5 --max_words=30