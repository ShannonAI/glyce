# -*- coding: utf-8 -*-


repo_path=/home/glyce



data_sign=nlpcc-dbqa
data_dir=/data/nfsdata/nlp/datasets/sentence_pair/nlpcc-dbqa

config_path=/home/mengyuxian/bert_glyce/configs/bert.json

bert_model=/data/nfsdata/nlp/BERT_BASE_DIR/chinese_L-12_H-768_A-12
task_name=clf
output_dir=/data/nfsdata/data/yuxian/train_logs/dbqa_bert_lr4e-6
max_seq_len=128
train_batch=24
dev_batch=48
test_batch=48
learning_rate=8e-7
num_train_epochs=6
warmup=-1
local_rank=-1
seed=3306
checkpoint=2000
nworkers=2




CUDA_VISIBLE_DEVICES=2
python3 ${repo_path}/run/run_classifier.py \
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
