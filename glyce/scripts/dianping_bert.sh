# -*- coding: utf-8 -*-
repo_path=/home/ubuntu/glyce/glyce

data_sign=dianping
data_dir=/data/sentiment_analysis/dianping
output_dir=/home/ubuntu/train_logs/models/dianping_1e-5  # change save_path !!!

config_path=/home/ubuntu/glyce/glyce/configs/bert.json
bert_model=/data/bert_base_chinese

task_name=clf
max_seq_len=256
train_batch=48
dev_batch=48
test_batch=48
learning_rate=1e-5
num_train_epochs=4
warmup=0.1
local_rank=-1
seed=3306
checkpoint=10000
gpus=12-13-14


python3 ${repo_path}/run/run_text_classification.py \
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
--output_dir ${output_dir} \
--gpus ${gpus}
