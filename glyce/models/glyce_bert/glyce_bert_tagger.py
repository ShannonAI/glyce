#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# Author: Xiaoy LI 
# Last update: 2019.03.25 
# First create: 2019.03.25 
# Description:
# glyph_transformer_tagging.py



import os 
import sys 


root_path = "/".join(os.path.realpath(__file__).split("/")[:-4])
if root_path not in sys.path:
    sys.path.insert(0, root_path)


import torch 
import torch.nn as nn 
from torch.nn import CrossEntropyLoss 


from glyce.layers.classifier import * 
from glyce.layers.bert_basic_model import * 
from glyce.layers.glyce_transformer import GlyceTransformer 




class GlyceBertTagger(nn.Module):
    def __init__(self, config, num_labels=4):
        super(GlyceBertTagger, self).__init__()
        self.num_labels = num_labels 
        self.glyph_transformer = GlyphTransformer(config, num_labels=num_labels)
        # config involves here 
        # 1. config.glyph_config 
        # 2. config.bert_config 
        # 3. transformer_config 
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if config.classifier_sign == "single_linear":
            self.classifier = SingleLinearClassifier(config.hidden_size, self.num_labels)
        elif config.classifier_sign == "multi_nonlinear":
            self.classifier = MultiNonLinearClassifier(config.hidden_size, self.num_labels)
        else:
            raise ValueError 

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, \
        labels=None):

        encoded_layers, pooled_output, glyph_cls_loss = self.glyph_transformer(input_ids, \
            token_type_ids=token_type_ids, attention_mask=attention_mask)

        features_output = encoded_layers[-1]
        features_output = self.dropout(features_output)
        logits = self.classifier(features_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, glyph_cls_loss  
        else:
            return logits, glyph_cls_loss  

         

