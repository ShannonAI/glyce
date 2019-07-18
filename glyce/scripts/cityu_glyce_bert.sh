#!/usr/bin/env bash 
# -*- coding: utf-8 -*- 




# Author: Xiaoy LI 
# Last update: 2019.03.20 
# First create: 2019.03.23 
# Description:
# 



repo_path=/home/yinfan/bert_cws



data_sign=ctb6_cws
data_dir=/data/nfsdata/xiaoya/datasets/zh_tagging/CTB6CWS
config_path=${repo_path}/configs/ctb6cws_glyph_bert.json
bert_model=/data/nfsdata/nlp/BERT_BASE_DIR/chinese_L-12_H-768_A-12
task_name=cws
output_dir=/home/yinfan/bert_glyce
max_seq_len=200
train_batch=16
dev_batch=32
test_batch=32
learning_rate=3e-5
num_train_epochs=20
warmup=-1
local_rank=-1
seed=3310
checkpoint=100
gradient_accumulation_steps=2

CUDA_VISIBLE_DEVICES=0  python3 ${repo_path}/run/run_glyph_tagger.py \
--data_sign ${data_sign} \
--config_path ${config_path} \
--data_dir ${data_dir} \
--bert_model ${bert_model} \
--task_name ${task_name} \
--max_seq_length ${max_seq_len} \
--do_train \
--do_eval \
--seed ${seed} \
--train_batch_size ${train_batch} \
--dev_batch_size ${dev_batch} \
--test_batch_size ${test_batch} \
--learning_rate ${learning_rate} \
--num_train_epochs ${num_train_epochs} \
--checkpoint ${checkpoint} \
--warmup_proportion ${warmup} \
--gradient_accumulation_steps ${gradient_accumulation_steps}
