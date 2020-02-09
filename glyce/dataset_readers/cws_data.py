#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# Author: Xiaoy LI 
# Last update: 2019.05.13 
# First create: 2019.05.13 
# Description;
# 


import os
import sys 


root_path = "/".join(os.path.realpath(__file__).split("/")[:-3]) 
if root_path not in sys.path:
    sys.path.insert(0, root_path)


import torch
from torch.utils import data as D


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, reverse=False):
        self.dictionary = Dictionary()
        self.files = [f for f in os.listdir(path) if f.endswith('.raw')]
        train_files = self.files[:int(len(self.files) * 0.8)]
        test_files = self.files[int(len(self.files) * 0.8):int(len(self.files) * 0.9)]
        valid_files = self.files[int(len(self.files) * 0.9):]
        self.train = self.tokenize([os.path.join(path, f) for f in train_files], reverse)
        self.valid = self.tokenize([os.path.join(path, f) for f in valid_files], reverse)
        self.test = self.tokenize([os.path.join(path, f) for f in test_files], reverse)

    def tokenize(self, paths, reverse=False):
        """Tokenizes a text file."""
        tokens = []
        for path in paths:
            # print(F'handle {path}')
            with open(path, 'r', encoding='utf8') as fi:
                for line in fi:
                    if not line.startswith('<'):
                        tokens += list(line.strip().lower()) + ['<eos>']
        ids = torch.LongTensor(len(tokens))
        if reverse:
            tokens.reverse()
        for i, token in enumerate(tokens):
            ids[i] = self.dictionary.add_word(token)
        return ids


class CWSDataSet(D.Dataset):
    """
    A customized data loader.
    """

    def __init__(self, paths):
        """ Intialize the dataset
        """
        self.paths = paths
        self.samples = []
        features, labels = self.generate_corpus(paths, fine_grain=True)
        self.f_map, self.l_map = CWSDataSet.get_maps()
        for feature, label in zip(features, labels):
            mapped_feature, mapped_label = [self.f_map[i] for i in feature], [self.l_map[i] for i in label]
            padded_feature = CWSDataSet.pad_sequence(mapped_feature, 50)
            padded_label = CWSDataSet.pad_sequence(mapped_label, 50)
            assert len(padded_feature) == len(padded_label) == 50, F"{len(padded_feature), len(padded_label)}"
            self.samples.append({'feature': padded_feature, 'label': padded_label})

    def __getitem__(self, index):
        """ Get a sample from the dataset
        """
        return self.samples[index]

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return len(self.samples)

    @staticmethod
    def generate_corpus(paths, fine_grain=False):
        features = []
        labels = []
        for path in paths:
            with open(path) as fi:
                for line in fi:
                    if not line.startswith('<'):
                        tags = []
                        line = line.lstrip('，：；、')
                        for i, c in enumerate(line.strip()):
                            if c != ' ':
                                if i == 0:
                                    if line[i + 1] == ' ':
                                        tags.append('S')
                                    else:
                                        tags.append('B')
                                else:
                                    if line[i + 1] in ' \n' and line[i - 1] == ' ':
                                        tags.append('S')
                                    elif line[i + 1] in ' \n':  # 下一个为空，前一个不为空，就是End
                                        tags.append('E')
                                    elif line[i - 1] == ' ':
                                        tags.append('B')
                                    else:
                                        tags.append('I')
                        sentence = line.strip().replace(' ', '')
                        if sentence.startswith('，'):
                            print(line, sentence, tags)
                        assert len(sentence) == len(tags), F'{line, sentence, tags}'
                        if fine_grain:
                            idxes = [i for i, c in enumerate(sentence) if c in '，：；、']
                            for i, idx in enumerate(idxes):
                                if i:
                                    features.append(sentence[idxes[i - 1] + 1: idx + 1])
                                    labels.append(tags[idxes[i - 1] + 1: idx + 1])
                        else:
                            features.append(sentence)
                            labels.append(tags)
        return features, labels

    @staticmethod
    def get_maps():
        path = '/data/nfsdata/nlp/datasets/language_modeling/ctb_v6/data/utf8/segmented'
        files = [f for f in os.listdir(path) if f.endswith('.seg')]
        all_features, all_labels = CWSDataSet.generate_corpus([os.path.join(path, f) for f in files])
        f_map = {f: i for i, f in enumerate(set().union(*all_features))}
        l_map = {l: i for i, l in enumerate(set().union(*all_labels))}
        return f_map, l_map

    @staticmethod
    def pad_sequence(sequence, max_len):
        if len(sequence) > max_len:
            return torch.LongTensor(sequence[: max_len])
        else:
            return torch.LongTensor(sequence + [-1] * (max_len - len(sequence)))


def get_dataloaders(batch_size):
    path = '/data/nfsdata/nlp/datasets/language_modeling/ctb_v6/data/utf8/segmented'
    files = [f for f in os.listdir(path) if f.endswith('.seg')]
    train_files = files[:int(len(files) * 0.8)]
    test_files = files[int(len(files) * 0.8):int(len(files) * 0.9)]
    valid_files = files[int(len(files) * 0.9):]
    train_dataset = CWSDataSet([os.path.join(path, f) for f in train_files])
    valid_dataset = CWSDataSet([os.path.join(path, f) for f in valid_files])
    test_dataset = CWSDataSet([os.path.join(path, f) for f in test_files])
    train_loader = D.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = D.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = D.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    path = '/data/nfsdata/nlp/datasets/language_modeling/ctb_v6/data/utf8/segmented'
    files = [f for f in os.listdir(path) if f.endswith('.seg')]
    dataset = CWSDataSet([os.path.join(path, f) for f in files])
    ls = []
    for d in dataset:
        print(d['feature'], d['label'])
