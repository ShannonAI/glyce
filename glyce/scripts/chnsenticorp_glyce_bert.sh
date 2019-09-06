# -*- coding: utf-8 -*-
repo_path=/home/glyce/glyce
data_sign=chnsenticorp
data_dir=/data/nfsdata/nlp/datasets/sentence_pair/chnsenticorp
output_dir=/data/nfsdata2/train_logs/chnsenticorp_glyph_test # change save_path !!!
config_path=/home/bert_glyce/configs/glyph_bert.json
bert_model=/data/nfsdata2//train_logs/chnsenticorp_bert/checkpoint1
task_name=clf

max_seq_len=256
train_batch=8
dev_batch=8
test_batch=8
learning_rate=2e-5
num_train_epochs=3
warmup=0.1
local_rank=-1
seed=3308
checkpoint=100


CUDA_VISIBLE_DEVICES=0,1,2 python3 ${repo_path}/bin/run_bert_glyce_classifier.py \
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
