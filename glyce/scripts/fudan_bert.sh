#!/usr/bin/env bash 
# -*- coding: utf-8 -*-


# Author: Fei Wang, Xiaoy LI 
# Last update: 2020.02.08 
# First create: 2020.02.08 
# Description:
# glyce-bert for cls 


repo_path=/data/xiaoya/work/gitrepo/glyce/glyce/

data_sign=fudan
data_dir=/data/nfsdata2/xiaoya/data_repo/glyce/sent_single/fudan
output_dir=/data/xiaoya/export-models
config_path=/data/xiaoya/work/gitrepo/glyce/glyce/configs/bert.json
bert_model=/data/nfsdata/nlp/BERT_BASE_DIR/chinese_L-12_H-768_A-12
checkpoint=10
task_name=clf
max_seq_len=128
train_batch=32
dev_batch=32
test_batch=32
learning_rate=2e-5
num_train_epochs=4
warmup=0.1
local_rank=-1
seed=3306

CUDA_VISIBLE_DEVICES=2 python3 ${repo_path}/bin/run_bert_classifier.py \
--data_sign ${data_sign} \
--config_path ${config_path} \
--data_dir ${data_dir} \
--bert_model ${bert_model} \
--task_name ${task_name} \
--max_seq_length ${max_seq_len} \
--do_train \
--do_eval \
--train_batch_size ${train_batch} \
--dev_batch_size ${dev_batch} \
--test_batch_size ${test_batch} \
--learning_rate ${learning_rate} \
--num_train_epochs ${num_train_epochs} \
--checkpoint ${checkpoint} \
--warmup_proportion ${warmup} \
--output_dir ${output_dir}
