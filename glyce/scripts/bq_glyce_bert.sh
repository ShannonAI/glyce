# -*- coding: utf-8 -*-
repo_path=/data/xiaoya/work/gitrepo/glyce/glyce

data_sign=bq
data_dir=/data/nfsdata/nlp/datasets/sentence_pair/bq_corpus
output_dir=/data/xiaoya/export-models  # change save_path !!!

config_path=/data/xiaoya/work/gitrepo/glyce/glyce/configs/bq_glyce_bert.json
bert_model=/data/nfsdata/nlp/BERT_BASE_DIR/chinese_L-12_H-768_A-12

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
checkpoint=100


CUDA_VISIBLE_DEVICES=0 python3 ${repo_path}/bin/run_bert_glyce_classifier.py \
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
