# -*- coding: utf-8 -*-
repo_path=/home/ubuntu/glyce/glyce/
data_sign=lcqmc
data_dir=/data/sentence_pair/LCQMC
output_dir=/home/wangfei/train_logs/models/lcqmc_glyph_replay # change save_path !!!
config_path=/data/nfsdata/nieping/gitlab/bert_glyce_new/configs/glyph_bert.json
bert_model=/data/nfsdata2/wangfei/sever5/train_logs/lcqmc_bert_1800/checkpoint6
task_name=clf


max_seq_len=64
train_batch=32
dev_batch=32
test_batch=32
learning_rate=3e-4
num_train_epochs=4
warmup=0.1
gradient_accumulation_steps=2
local_rank=-1
seed=3306
checkpoint=600
gpus=2-3-0





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
--cnn_dropout ${cnn_dropout} \
--glyph_hidden_dropout_prob ${glyph_hidden_dropout_prob} \
--glyph_embsize ${glyph_embsize} \
--glyph_dropout ${glyph_dropout} \
--fc_merge ${fc_merge} \
--bert_frozen ${bert_frozen} \
--bert_hidden_dropout ${bert_hidden_dropout} \
--transformer_hidden_dropout_prob ${transformer_hidden_dropout_prob} \
--transformer_feedforward_dropout ${transformer_feedforward_dropout} \
--transformer_attention_probs_dropout_prob ${transformer_attention_probs_dropout_prob} \
--gradient_accumulation_steps ${gradient_accumulation_steps}


