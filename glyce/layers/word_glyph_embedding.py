#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 

import os 
import sys 


root_path = "/".join(os.path.realpath(__file__).split("/")[:-3])
if root_path not in sys.path:
    sys.path.insert(0, root_path)


import torch
import torch.nn as nn
import torch.nn.functional as F


import random 


from glyce.glyph_cnn_models.glyph_group_cnn import GlyphGroupCNN
from glyce.utils.random_erasing import RandomErasing
from glyce.glyph_cnn_models.word_encoder import Encoder 
from glyce.utils.components import SubCharComponent, Highway
from glyce.layers.mask_cross_entropy import MaskCrossEntropy
from glyce.utils.render import multiple_glyph_embeddings, get_font_names



class WordGlyphEmbedding(nn.Module):
    """输入token_id，输出其对应的word embedding　glyph embedding或者两者的结合. config中的参数定义如下：
    dropout: float, dropout rate
    idx2word: dict, 单词到对应token_id的映射
    word_embsize: int, word embedding size
    idx2char:  dict, char到对应token_id的映射
    char_embsize: int, char embedding size
    glyph_embsize: int, glyph embedding size
    pretrained_word_embedding: list, pretrained word embedding
    font_channels: int, 塞到channel里的字体的数目，如果random_fonts > 0，则代表可供随机选择的总字体数
    random_fonts: int, 每个batch都random sample　n个不同字体塞到n个channel里
    font_name: str, 形如'CJK/NotoSansCJKsc-Regular.otf'的字体名称，当font_channels=1时有效
    font_size: int, 字体大小
    use_traditional: bool, 是否用繁体字代替简体字
    font_normalize: bool, 是否对字体输入的灰度值归一化，即减去均值，除以标准差
    subchar_type: str, 是否使用拼音('pinyin')或者五笔('wubi')
    subchar_embsize: int, 拼音或五笔的embedding_size
    random_erase: bool, 是否随机block掉字体灰度图中的一小块
    num_fonts_concat: int, 将一个字对应的n个字体分别过CNN后得到的特征向量concat在一起
    glyph_cnn_type: str, 用于抽取glyph信息的cnn模型的名称
    cnn_dropout: float, glyph cnn dropout rate
    char_drop: float, char embedding dropout rate
    use_batch_norm: bool, 是否使用batch normalization
    use_layer_norm: bool, 是否使用layer normalization
    use_highway: bool, 是否将concat之后的向量过highway
    yuxian_merge: bool, 是否将concat之后的向量过yuxian_merge
    fc_merge: bool, 是否将concat之后的向量过全连接
    output_size: bool, 输出向量的维度
    char2word_dim: int, char变为word后的维度
    use_maxpool: bool, use maxpool to merge char embedding and glyce embedding

    """

    def __init__(self, model_config):
        super(WordGlyphEmbedding, self).__init__()
        self.config = model_config
        all_fonts = get_font_names()
        self.drop = nn.Dropout(self.config.dropout)
        self.char_drop = nn.Dropout(self.config.char_drop)

        if self.config.word_embsize:
            self.word_embedding = nn.Embedding(len(self.config.idx2word), self.config.word_embsize)
        if self.config.char_embsize:
            self.char_embedding = nn.Embedding(len(self.config.idx2char), self.config.char_embsize)

        self.glyph_embeddings = nn.ParameterList([nn.Parameter(multiple_glyph_embeddings(
            self.config.font_channels, all_fonts[i] if i else self.config.font_name, self.config.idx2char,
            self.config.font_size, self.config.use_traditional, self.config.font_normalize), requires_grad=False) for i
            in range(self.config.num_fonts_concat)])

        if self.config.subchar_type:
            self.subchar_component = SubCharComponent(encoding_type=self.config.subchar_type,
                                                      composing_func='GRU',
                                                      embedding_size=self.config.subchar_embsize,
                                                      hidden_size=self.config.glyph_embsize // 2,
                                                      num_layers=1,
                                                      )

        if self.config.random_erase:
            self.random_erasing = RandomErasing()

        self.token_size = self.config.word_embsize
        if self.config.subchar_type:
            self.token_size += self.config.glyph_embsize

        self.chars2vec_dim = self.config.char_embsize + self.config.glyph_embsize * self.config.num_fonts_concat
        num_filters = 0 if self.config.use_maxpool else self.config.char2word_dim
        if self.chars2vec_dim:
            self.chars2vec = Encoder(self.chars2vec_dim, num_filters=num_filters,
                                     output_dim=self.config.char2word_dim, ngram_filter_sizes=(2,),
                                     groups=1)
            self.token_size += self.chars2vec.get_output_dim()

        if not (self.config.use_highway or self.config.yuxian_merge or self.config.fc_merge):
            assert self.token_size == self.config.output_size, '没有用后处理，token_size {}应该等于output_size {}'.format(self.token_size, self.config.output_size)

        if self.config.num_fonts_concat:
            self.glyph_cnn_model = GlyphGroupCNN(
            # self.glyph_cnn_model = getattr(glyph_cnn_models, self.config.glyph_cnn_type)(
                num_features=self.config.glyph_embsize, cnn_drop=self.config.cnn_dropout,
                font_channels=self.config.random_fonts or self.config.font_channels, groups=self.config.glyph_groups)
            self.glyph_classifier = nn.Linear(self.config.glyph_embsize, len(self.config.idx2char))
            self.glyph_classification_criterion = MaskCrossEntropy(self.config.loss_mask_ids)

        if self.config.use_layer_norm:
            self.layer_norm = nn.LayerNorm(self.token_size)
        if self.config.use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(self.token_size)

        if self.config.use_highway:
            self.highway = Highway(self.token_size, self.config.output_size, 1, F.relu)
        if self.config.yuxian_merge:
            self.merge_feature = nn.Conv1d(self.token_size, self.config.output_size, kernel_size=2, groups=64)
        if self.config.fc_merge:
            self.fc_concat = nn.Linear(self.token_size, self.config.output_size)

        self.init_weights()  # 模型初始化时权值也要初始化

    def forward(self, data):  # 前向传播，输入输出加dropout，words:  (seq_len, batch)   chars: seqlen*4, batch
        words, chars = data
        seq_length, batch = words.size()
        num_char = chars.size(0) // seq_length
        all_embeddings = []
        char_embeddings = []
        glyph_loss = []
        seq_length, batch = words.size()
        if self.config.word_embsize:
            all_embeddings.append(self.drop(self.word_embedding(words.flatten())))  # emb: (-1, emb_size)
        if self.config.char_embsize:
            char_embeddings.append(self.char_drop(self.char_embedding(chars)))   # seqlen*4, batch, chardim

        if self.config.num_fonts_concat:  # (seq_len*batch, num_channels, fontsize, fontsize)
            for glyph_embedding in self.glyph_embeddings:
                glyph_emb = glyph_embedding.index_select(0, chars.flatten())  # (seq_len*4*batch, nfonts, fontsize, fontsize)
                if self.config.random_fonts:
                    idxes = data.new(random.sample(range(self.config.font_channels), self.config.random_fonts))
                    glyph_emb = glyph_emb.index_select(1, idxes)
                if self.config.random_erase and self.training:
                    glyph_emb = self.random_erasing(glyph_emb)
                glyph_feat = self.glyph_cnn_model(glyph_emb)   # (seq_len*4*batch, featsize)
                glyph_logit = self.glyph_classifier(glyph_feat)   # (seq_len*4*batch, ntokens)
                char_embeddings.append(glyph_feat.view(seq_length*num_char, batch, -1))
                glyph_loss.append(self.glyph_classification_criterion(glyph_logit, chars))

        if char_embeddings:
            char_embeds = torch.cat(char_embeddings, 2)  # seqlen*4, batch, featsize2
            char_embeds = char_embeds.view(seq_length, -1, batch, self.chars2vec_dim).transpose(1, 2).contiguous().\
                view(seq_length*batch, -1, self.chars2vec_dim)  #seqlen*batch, maxchar, featsize2
            char_embeds = self.chars2vec(char_embeds)  #seqlen*batch, featsize3
            all_embeddings.append(char_embeds)

        if self.config.subchar_type:
            all_embeddings.append(self.subchar_component(words))

        emb = torch.cat(all_embeddings, -1)  # seql, batch, feat*2

        if self.config.use_batch_norm:
            emb = self.batch_norm(emb)
        if self.config.use_layer_norm:
            emb = self.layer_norm(emb)

        if self.config.use_highway:
            emb = self.highway(emb)
        elif self.config.yuxian_merge:
            emb = self.merge_feature(emb.view(-1, self.config.output_size, self.token_size // self.config.output_size))
        elif self.config.fc_merge:
            emb = F.relu(self.fc_concat(emb))

        glyph_classification_loss = sum(glyph_loss) / len(glyph_loss) if glyph_loss else 0
        out_shape = list(data[0].size())
        out_shape.append(self.config.output_size)
        return emb.view(*out_shape), glyph_classification_loss

    def init_weights(self):
        initrange = 0.1
        if self.config.word_embsize:
            self.word_embedding.weight.data.uniform_(-initrange, initrange)
            if self.config.pretrained_word_embedding:
                self.word_embedding.weight = nn.Parameter(torch.FloatTensor(self.config.pretrained_word_embedding))
        if self.config.char_embsize:
            self.char_embedding.weight.data.uniform_(-initrange, initrange)
            if self.config.pretrained_char_embedding:
                self.char_embedding.weight = nn.Parameter(torch.FloatTensor(self.config.pretrained_char_embedding))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.uniform_(-initrange, initrange)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(std=0.001)
                if m.bias is not None:
                    m.bias.data.zero_()



if __name__ == '__main__':
    pass
