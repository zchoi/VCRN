#!/usr/bin/env zsh
CUDA_VISIBLE_DEVICES=0 python evaluate.py --dataset=msvd --model=RMN \
    --result_dir=results/msvd/msvd_1000_2048hidden_beam5_gate --attention=myatt \
    --use_loc --use_rel --use_func \
    --hidden_size=512 --att_size=512 \
    --test_batch_size=32 --beam_size=5 \
    --eval_metric=CIDEr --topk=18 --max_words=26
