#!/usr/bin/env bash
# -*- coding: utf-8 -*-


repo_path=/home/ubuntu/glyce



data_sign=nlpcc-dbqa
data_dir=/data/nfsdata/nlp/datasets/sentence_pair/nlpcc-dbqa

config_path=/glyce/configs/glyph_bert.json

bert_model=/data/nfsdata/data/train_logs/dbqa_best_bert
task_name=clf
output_dir=/data/nfsdata/data/train_ilogs/dbqa_together_lr4e-6_warmup0.2
max_seq_len=128
train_batch=32
dev_batch=64
test_batch=64
learning_rate=4e-6
num_train_epochs=6
warmup=0.2
local_rank=-1
seed=3306
checkpoint=1000


CUDA_VISIBLE_DEVICES=1 python3 ${repo_path}/bin/run_bert_glyce_classifier.py \
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
