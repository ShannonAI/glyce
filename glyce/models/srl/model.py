#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 


# Author: Xiaoy LI 
# Last update: 2019.02.25 
# First create: 2019.02.25 
# Description:
# 

import os 
import sys 


root_path = "/".join(os.path.realpath(__file__).split("/")[:-4])
if root_path not in sys.path:
    sys.path.insert(0, root_path)


import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


import numpy as np


from glyce import GlyceConfig 
from glyce import WordGlyceEmbedding 
from glyce.models.srl.highway import HighwayMLP
from glyce.models.srl.utils import USE_CUDA
from glyce.models.srl.utils import get_torch_variable_from_np, get_data

from allennlp.modules.elmo import Elmo
from allennlp.data import dataset as Dataset 
from allennlp.data import Token, Vocabulary, Instance
from allennlp.data.fields import TextField
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer




indexer = ELMoTokenCharactersIndexer()
def batch_to_ids(batch):
    """
    Given a batch (as list of tokenized sentences), return a batch
    of padded character ids.
    """
    instances = []
    for sentence in batch:
        tokens = [Token(token) for token in sentence]
        field = TextField(tokens, {'character_ids': indexer})
        instance = Instance({"elmo": field})
        instances.append(instance)

    dataset = Dataset(instances)
    vocab = Vocabulary()
    # dataset.index_instances(vocab)
    for instance in dataset.instances:
        instance.index_fields(vocab)
    return dataset.as_tensor_dict()['elmo']['character_ids']



class End2EndModel(nn.Module):
    def __init__(self, model_params, config):
        super(End2EndModel,self).__init__()
        self.dropout = model_params['dropout']
        self.batch_size = model_params['batch_size']

        self.word_vocab_size = model_params['word_vocab_size']
        self.lemma_vocab_size = model_params['lemma_vocab_size']
        self.pos_vocab_size = model_params['pos_vocab_size']
        self.deprel_vocab_size = model_params['deprel_vocab_size']
        self.pretrain_vocab_size = model_params['pretrain_vocab_size']

        self.flag_emb_size = model_params['flag_embedding_size']
        self.word_emb_size = model_params['word_emb_size']
        self.lemma_emb_size = model_params['lemma_emb_size']
        self.pos_emb_size = model_params['pos_emb_size']
        self.deprel_emb_size = model_params['deprel_emb_size']
        self.pretrain_emb_size = model_params['pretrain_emb_size']
        self.elmo_emb_size = model_params['elmo_embedding_size']

        self.pretrain_emb_weight = model_params['pretrain_emb_weight']

        self.bilstm_num_layers = model_params['bilstm_num_layers']
        self.bilstm_hidden_size = model_params['bilstm_hidden_size']
        self.target_vocab_size = model_params['target_vocab_size']
        self.highway_layers = model_params['highway_layers']
        self.use_highway = model_params['use_highway']
        self.use_deprel = model_params['use_deprel']
        self.use_elmo = model_params['use_elmo']
        self.use_flag_embedding = model_params['use_flag_embedding']

        self.deprel2idx = model_params['deprel2idx']

        # use glyph
        # self.glyph = WordGlyphEmbedding(config)
        self.glyph = WordGlyceEmbedding(config)

        if self.use_flag_embedding:
            self.flag_embedding = nn.Embedding(2, self.flag_emb_size)
            self.flag_embedding.weight.data.uniform_(-1.0,1.0)

        self.lemma_embedding = nn.Embedding(self.lemma_vocab_size, self.lemma_emb_size)
        self.lemma_embedding.weight.data.uniform_(-1.0,1.0)

        self.pos_embedding = nn.Embedding(self.pos_vocab_size, self.pos_emb_size)
        self.pos_embedding.weight.data.uniform_(-1.0,1.0)

        if self.use_deprel:
            self.deprel_embedding = nn.Embedding(self.deprel_vocab_size, self.deprel_emb_size)
            self.deprel_embedding.weight.data.uniform_(-1.0,1.0)

        self.pretrained_embedding = nn.Embedding(self.pretrain_vocab_size,self.pretrain_emb_size)
        self.pretrained_embedding.weight.data.copy_(torch.from_numpy(self.pretrain_emb_weight))

        input_emb_size = 0

        if self.use_flag_embedding:
            input_emb_size += self.flag_emb_size
        else:
            input_emb_size += 1

        # glyph output_size
        input_emb_size += config.output_size

        if self.use_deprel:
            input_emb_size +=  self.pretrain_emb_size + self.lemma_emb_size + self.pos_emb_size + self.deprel_emb_size
        else:
            input_emb_size +=  self.pretrain_emb_size + self.lemma_emb_size + self.pos_emb_size

        
        if self.use_elmo:
            input_emb_size += self.elmo_emb_size
            self.elmo_mlp = nn.Sequential(nn.Linear(1024, self.elmo_emb_size), nn.ReLU())
            self.elmo_w = nn.Parameter(torch.Tensor([0.5,0.5]))
            self.elmo_gamma = nn.Parameter(torch.ones(1))

        if USE_CUDA:
            self.bilstm_hidden_state0 = (Variable(torch.zeros(2 * self.bilstm_num_layers, self.batch_size, self.bilstm_hidden_size),requires_grad=True).cuda(),
                                        Variable(torch.zeros(2 * self.bilstm_num_layers, self.batch_size, self.bilstm_hidden_size),requires_grad=True).cuda())
        else:
            self.bilstm_hidden_state0 = (Variable(torch.zeros(2 * self.bilstm_num_layers, self.batch_size, self.bilstm_hidden_size),requires_grad=True),
                                        Variable(torch.zeros(2 * self.bilstm_num_layers, self.batch_size, self.bilstm_hidden_size),requires_grad=True)) #randn


        self.bilstm_layer = nn.LSTM(input_size=input_emb_size,
                                    hidden_size = self.bilstm_hidden_size, num_layers = self.bilstm_num_layers,
                                    dropout = self.dropout, bidirectional = True,
                                    bias = True, batch_first=True)


        if self.use_highway:
            self.highway_layers = nn.ModuleList([HighwayMLP(self.bilstm_hidden_size*2, activation_function=F.relu)
                                                for _ in range(self.highway_layers)])

            self.output_layer = nn.Linear(self.bilstm_hidden_size*2, self.target_vocab_size)
        else:
            self.output_layer = nn.Linear(self.bilstm_hidden_size*2,self.target_vocab_size)


    def softmax(self,input, axis=1):
        """
        Softmax applied to axis=n
 
        Args:
           input: {Tensor,Variable} input on which softmax is to be applied
           axis : {int} axis on which softmax is to be applied
 
        Returns:
            softmaxed tensors
       
        """
 
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size)-1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = F.softmax(input_2d)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size)-1)
        

    def forward(self, batch_input, elmo):
        
        flag_batch = get_torch_variable_from_np(batch_input['flag'])
        # word_batch: (batch_size, max_len)
        word_batch = get_torch_variable_from_np(batch_input['word'])
        # char_batch: (batch_size, 4 * max_len), for each word we need 4 characters (pad or truncate)
        char_batch = get_torch_variable_from_np(batch_input['char'])
        
        lemma_batch = get_torch_variable_from_np(batch_input['lemma'])
        pos_batch = get_torch_variable_from_np(batch_input['pos'])
        deprel_batch = get_torch_variable_from_np(batch_input['deprel'])
        pretrain_batch = get_torch_variable_from_np(batch_input['pretrain'])
        origin_batch = batch_input['origin']
        origin_deprel_batch = batch_input['deprel']


        if self.use_flag_embedding:
            flag_emb = self.flag_embedding(flag_batch)
        else:
            flag_emb = flag_batch.view(flag_batch.shape[0],flag_batch.shape[1], 1)
        
        lemma_emb = self.lemma_embedding(lemma_batch)
        pos_emb = self.pos_embedding(pos_batch)
        pretrain_emb = self.pretrained_embedding(pretrain_batch)

        if self.use_deprel:
            deprel_emb = self.deprel_embedding(deprel_batch)

        # glyph embedding
        glyph_emb, glyph_loss = self.glyph((word_batch.t().contiguous(), char_batch.t().contiguous())) # firstly we don't use the classification loss
        glyph_emb = glyph_emb.transpose(0, 1)

        if self.use_deprel:
            input_emb = torch.cat([flag_emb, pretrain_emb, lemma_emb, pos_emb, deprel_emb, glyph_emb], 2)
        else:
           input_emb = torch.cat([flag_emb, pretrain_emb, lemma_emb, pos_emb, glyph_emb], 2)


        if self.use_elmo:

            # input_emb = self.input_layer(input_emb)

            # Finally, compute representations.
            # The input is tokenized text, without any normalization.
            batch_text = batch_input['text']

            # character ids is size (3, 11, 50)
            character_ids = batch_to_ids(batch_text)
            if USE_CUDA:
                character_ids = character_ids.cuda()

            representations = elmo(character_ids)
            # representations['elmo_representations'] is a list with two elements,
            #   each is a tensor of size (3, 11, 1024).  Sequences shorter then the
            #   maximum sequence are padded on the right, with undefined value where padded.
            # representations['mask'] is a (3, 11) shaped sequence mask.

            elmo_representations = representations['elmo_representations']

            w = F.softmax(self.elmo_w, dim=0)

            elmo_emb = self.elmo_gamma * (w[0] * elmo_representations[0] + w[1] * elmo_representations[1])

            elmo_emb = self.elmo_mlp(elmo_emb)

            input_emb = torch.cat([input_emb,elmo_emb],2)

        bilstm_output, (_, bilstm_final_state) = self.bilstm_layer(input_emb,self.bilstm_hidden_state0)

        
        bilstm_output = bilstm_output.contiguous()

        hidden_input = bilstm_output.view(bilstm_output.shape[0]*bilstm_output.shape[1],-1)

        if self.use_highway:
            for current_layer in self.highway_layers:
                hidden_input = current_layer(hidden_input)

            output = self.output_layer(hidden_input)
        else:
            output = self.output_layer(hidden_input)

        return output, glyph_loss

