# -*- coding: utf-8 -*-
repo_path=/home/wangfei/bert_glyce_new/
data_sign=chnsenticorp
data_dir=/data/nfsdata/nlp/datasets/sentence_pair/chnsenticorp
output_dir=/data/nfsdata2/wangfei/sever5/train_logs/chnsenticorp_glyph_test # change save_path !!!
config_path=/home/wangfei/bert_glyce_new/configs/glyph_bert.json
bert_model=/data/nfsdata2/wangfei/sever5/train_logs/chnsenticorp_bert/checkpoint1
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
gpus=0-3
glyph_decay=0.1
glyph_ratio=1.0
glyph_output_size=768
glyph_embsize=128
num_fonts_concat=6
font_channels=6
cnn_dropout=0.1
glyph_hidden_dropout_prob=0.1
glyph_dropout=0.1
fc_merge=0
bert_frozen=0
bert_hidden_dropout=0.1
transformer_hidden_dropout_prob=0.1
transformer_feedforward_dropout=0.1
transformer_attention_probs_dropout_prob=0.1



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
--transformer_attention_probs_dropout_prob ${transformer_attention_probs_dropout_prob}


