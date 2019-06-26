# encoding: utf-8
"""
@author: Yuxian Meng 
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: word_data
@time: 2019/1/15 18:43

    这一行开始写关于本文件的说明与解释
"""

import os
import sys 


root_path = "/".join(os.path.realpath(__file__).split("/")[:-3]) 
if root_path not in sys.path:
    sys.path.insert(0, root_path)


import torch


import json
import jieba
from collections import Counter



def cut_data():
    old_data_dir = '/data/nfsdata/nlp/datasets/language_modeling/ctb_v6/data/utf8/raw/'
    new_data_dir = '/data/nfsdata/nlp/datasets/language_modeling/ctb_v6/data/utf8/yuxian_tokens'

    if not os.path.exists(new_data_dir):
        os.mkdir(new_data_dir)

    for file_path in os.listdir(old_data_dir):
        if file_path.endswith('.raw'):
            tokens = []
            with open(os.path.join(old_data_dir, file_path)) as fin:
                for line in fin:
                    if not line.startswith('<'):
                        chars = list(line.strip().lower())
                        words = list(jieba.cut(''.join(chars), cut_all=False)) + ['<eos>']
                        tokens += words
            json.dump(tokens, open(os.path.join(new_data_dir, file_path)[:-3]+'json', 'w'), ensure_ascii=False)


class Dictionary(object):
    def __init__(self, valid_words=None):
        self.word2idx = {'<UNK>': 0}
        self.idx2word = ['<UNK>']
        self.char2idx = {'<PAD>': 0}
        self.idx2char = ['<PAD>']
        self.valid_words=valid_words

    def add_word(self, word):
        for char in word:
            if char not in self.char2idx:
                self.idx2char.append(char)
                self.char2idx[char] = len(self.idx2char) - 1
        char_ids = [self.char2idx[char] for char in word]
        if self.valid_words and word not in self.valid_words:
            word = '<UNK>'
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word], char_ids


class WordCorpus(object):
    def __init__(self, path='/data/nfsdata/nlp/datasets/language_modeling/ctb_v6/data/utf8/yuxian_tokens',
                 reverse=False, max_char=4, max_words=25857):
        self.max_char = max_char
        self.files = [f for f in os.listdir(path) if f.startswith('chtb')]
        train_files = self.files[:int(len(self.files)*0.8)]
        test_files = self.files[int(len(self.files)*0.8):int(len(self.files)*0.9)]
        valid_files = self.files[int(len(self.files)*0.9):]
        self.max_words = max_words
        self.valid_words = self.word_counter([os.path.join(path, f) for f in train_files])
        self.dictionary = Dictionary(valid_words=set(self.valid_words))
        self.train = self.tokenize([os.path.join(path, f) for f in train_files], reverse)
        self.valid = self.tokenize([os.path.join(path, f) for f in valid_files], reverse)
        self.test = self.tokenize([os.path.join(path, f) for f in test_files], reverse)
        with open(os.path.join(path, 'dictionary.json'), 'w') as fo:
            json.dump({
                'idx2word': self.dictionary.idx2word, 'word2idx': self.dictionary.word2idx,
                'idx2char': self.dictionary.idx2char, 'char2idx': self.dictionary.char2idx,
            }, fo)

    def tokenize(self, paths, reverse=False):
        """Tokenizes a text file."""
        tokens = []
        for path in paths:
            # print(F'handle {path}')
            tokens += json.load(open(path, 'r', encoding='utf8'))
        word_ids = torch.LongTensor(len(tokens))
        char_ids = torch.LongTensor(len(tokens), self.max_char)
        if reverse:
            tokens.reverse()
        for i, token in enumerate(tokens):
            word_id, char_id = self.dictionary.add_word(token)
            word_ids[i] = word_id
            char_id = char_id[:self.max_char]
            char_id = char_id + [0]*(self.max_char-len(char_id))
            char_ids[i] = torch.Tensor(char_id)
        return word_ids, char_ids

    def word_counter(self, paths):
        counter = Counter()
        tokens = []
        for path in paths:
            tokens += json.load(open(path, 'r', encoding='utf8'))
        counter.update(tokens)
        print(len(counter))
        return [x[0] for x in counter.most_common(self.max_words)]


if __name__ == '__main__':
    corpus = WordCorpus()
    print(corpus.valid)


