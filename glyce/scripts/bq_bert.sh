#!/usr/bin/env bash 
# -*- coding: utf-8 -*- 



repo_path=/home/ubuntu/glyce/glyce
data_sign=bq
data_dir=/data/bq
output_dir=/home/ubuntu/train_logs/models/bq # change save_path !!!


config_path=/home/ubuntu/glyce/configs/bert.json
bert_model=/data/bert_base_chinese


task_name=clf
max_seq_len=64
train_batch=64
dev_batch=64
test_batch=64
learning_rate=2e-5
num_train_epochs=4
warmup=0.1
local_rank=-1
seed=3306
checkpoint=1500


python3 ${repo_path}/bin/run_text_classification.py \
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
