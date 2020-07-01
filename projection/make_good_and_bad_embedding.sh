#!/bin/bash

set -e

python3 make_embedding.py \
    --model_file /users5/kliao/code/CommonsenseERL_EMNLP_2019/model/publish_2020/yago_stack_prop_lowrank10_batch128_sigmoid_0.3_0.3_e9_b708_hard_79.13.pt \
    --output_file good_embedding.pt

python3 make_embedding.py \
    --model_file /users5/kliao/code/CommonsenseERL_EMNLP_2019/model/pretrain_nyt/lowrank_ntn_10/LowRankNeuralTensorNetwork_2007.pt \
    --output_file bad_embedding.pt
