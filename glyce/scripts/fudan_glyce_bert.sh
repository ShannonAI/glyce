# -*- coding: utf-8 -*-
repo_path=/data/nfsdata/nieping/gitlab/bert_glyce_work/

data_sign=fudan
data_dir=/data/nfsdata/nlp/datasets/sentiment_analysis/char/fudan_nieping/large/train0
output_dir=/data/nfsdata/nieping/models/fudan_bs32_glyph # change save_path !!!

config_path=/data/nfsdata/nieping/gitlab/bert_glyce_work/configs/fudan/glyph_bert.json
bert_model=/data/nfsdata/nieping/models/fudan_bs32_1

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
checkpoint=80
gpus=7
glyph_decay=0.1
glyph_ratio=0.1
glyph_output_size=768
glyph_embsize=96
num_fonts_concat=8
font_channels=8
glyph_hidden_dropout_prob=0.5
glyph_dropout=0.5
fc_merge=false


python3 ${repo_path}/run/run_glyph_text_classification.py \
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
--gpus ${gpus} \
--glyph_decay ${glyph_decay} \
--glyph_ratio ${glyph_ratio} \
--glyph_output_size ${glyph_output_size} \
--num_fonts_concat ${num_fonts_concat} \
--font_channels ${font_channels} \
--glyph_hidden_dropout_prob ${glyph_hidden_dropout_prob} \
--glyph_embsize ${glyph_embsize} \
--fc_merge ${fc_merge} \
--glyph_dropout ${glyph_dropout}
