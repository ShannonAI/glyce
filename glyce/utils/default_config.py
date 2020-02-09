# encoding: utf-8
"""
@author: wuwei 
@contact: wu.wei@pku.edu.cn

@version: 1.0
@license: Apache Licence
@file: default_config.py
@time: 19-1-19 上午11:16

"""

import os 
import sys 


root_path = "/".join(os.path.realpath(__file__).split("/")[:-3])
if root_path not in sys.path:
    sys.path.insert(0, root_path)


import json
import copy


class GlyphEmbeddingConfig(object):
    """Configuration class to store the configuration of a `GlyphEmbedding`.
    """

    def __init__(self, cnn_dropout=0.3, dropout=0.2, use_batch_norm=False, use_highway=False, use_layer_norm=False,
                 fc_merge=False, font_channels=1, font_name='CJK/NotoSansCJKsc-Regular.otf', font_normalize=False,
                 glyph_embsize=1024, num_fonts_concat=1, output_size=2048,  font_size=12,
                 pretrained_char_embedding='', random_erase=False, random_fonts=0, subchar_embsize=512, subchar_type='',
                 use_traditional=False, word_embsize=1024, yuxian_merge=False, idx2word=None, level='char', idx2char=None,
                 char_drop=0.3, char_embsize=1024, char2word_dim=1024, pretrained_word_embedding='', use_maxpool=True,
                 glyph_groups=16, loss_mask_ids=(0, 1)):
        """Constructs GlyphEmbeddingConfig.
        必调参数：
            dropout: drop out of word_embedding
            cnn_dropout: glyce dropout
            char_dropout: char_embedding dropout
            另推荐单开一个image classification 作为 regularizer, ratio和decay比较重要，需要调一下
            font_size(recommended by wuwei): 字号大小，注意是N**2。也许可以用来节约内存？
            font_channels: 多少个字体


        Args:
            dropout: float, dropout rate
            idx2word: dict, 单词的token_id到对应词的映射
            idx2char: dict, 字的token_id到字的映射
            word_embsize: int, word embedding size
            glyph_embsize: int, glyph embedding size
            pretrained_char_embedding: list, pretrained char embedding
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
            cnn_dropout: float, glyph cnn dropout rate
            use_batch_norm: bool, 是否使用batch normalization
            use_layer_norm: bool, 是否使用layer normalization
            use_highway: bool, 是否将concat之后的向量过highway
            yuxian_merge: bool, 是否将concat之后的向量过yuxian_merge
            fc_merge: bool, 是否将concat之后的向量过全连接
            output_size: bool, 输出向量的维度
            char_embsize: int, char embedding的维度
            level: str, char or word
            char_drop: float, dropout applied to char_embedding
            char2word_dim: int, char变为word后的维度
            use_maxpool: bool, use maxpool to merge char embedding and glyce embedding
            glyph_groups: groups of glyph cnn
            loss_mask_ids: list[int], token id that not used in glyph loss

        """
        self.dropout = dropout
        self.idx2word = idx2word
        self.idx2char = idx2char
        self.word_embsize = word_embsize
        self.glyph_embsize = glyph_embsize
        self.pretrained_char_embedding = pretrained_char_embedding
        self.pretrained_word_embedding = pretrained_word_embedding
        self.font_channels = font_channels
        self.random_fonts = random_fonts
        self.font_name = font_name
        self.font_size = font_size
        self.use_traditional = use_traditional
        self.font_normalize = font_normalize
        self.subchar_type = subchar_type
        self.subchar_embsize = subchar_embsize
        self.random_erase = random_erase
        self.num_fonts_concat = num_fonts_concat
        self.cnn_dropout = cnn_dropout
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_highway = use_highway
        self.yuxian_merge = yuxian_merge
        self.fc_merge = fc_merge
        self.output_size = output_size
        self.char_embsize = char_embsize
        self.level = level
        self.char_drop = char_drop
        self.char2word_dim = char2word_dim
        self.use_maxpool = use_maxpool
        self.glyph_groups = glyph_groups
        self.loss_mask_ids = loss_mask_ids or (0, 1)

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `GlyphEmbeddingConfig` from a Python dictionary of parameters."""
        config = GlyphEmbeddingConfig()
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `GlyphEmbeddingConfig` from a json file of parameters."""
        with open(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
