# -*- coding: utf-8 -*-
repo_path=/home/ubuntu/glyce/glyce

data_sign=dianping
data_dir=/home/ubuntu/data/sentiment_analysis/dianping
output_dir=/home/ubuntu/train_logs/models/dianping_glyph_lr1e-5_ratio1 # change save_path !!!

config_path=/home/ubuntu/glyce/configs/glyph_bert.json
bert_model=/home/ubuntu/train_logs/models/dianping_1e-5/

task_name=clf
max_seq_len=256
train_batch=32
dev_batch=32
test_batch=32
learning_rate=1e-5
num_train_epochs=4
warmup=0.1
local_rank=-1
seed=3306
checkpoint=20000

CUDA_VISIBLE_DEVICES=2 python3 ${repo_path}/bin/run_bert_glyce_classifier.py \
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
