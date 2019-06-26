#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



import os
import sys 


root_path = "/".join(os.path.realpath(__file__).split("/")[:-3]) 
if root_path not in sys.path:
    sys.path.insert(0, root_path)


import torch
from torch.utils import data as D
from torchvision import transforms as TF


from PIL import Image


from glyce.utils.render import vocab_glyph_embedding


# import torchvision.transforms.functional as TF

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


class LMDataSet(D.Dataset):
    """
    A customized data loader.
    """

    def __init__(self, data_tensor, bptt):
        """ Intialize the dataset
        """
        self.samples = []
        for idx in range(0, len(data_tensor) - 1 - bptt, bptt):
            data = data_tensor[idx: idx + bptt]
            target = data_tensor[idx + 1: idx + 1 + bptt]
            self.samples.append({'data': data, 'target': target})

    def __getitem__(self, index):
        """ Get a sample from the dataset
        """
        return self.samples[index]

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return len(self.samples)


class YuxianDataSet(D.Dataset):
    """
    A customized data loader.
    """

    def __init__(self, data_tensor, bptt=35, batch_size=200, font_names=None, font_size=12, use_traditional=False,
                 font_normalize=False, augment=False):
        """ Intialize the dataset
        """
        self.data = batchify(data_tensor, batch_size)
        # self.data = self.data.transpose(0, 1)
        self.bptt = bptt
        self.batch_size = batch_size
        self.font_names = font_names or ['CJK/NotoSansCJKsc-Regular.otf']
        self.font_size = font_size
        self.augment = augment
        self.transform_funcs = TF.Compose([
            TF.ToPILImage(),
            TF.RandomResizedCrop(size=self.font_size, scale=(0.8, 1.0), ratio=(0.8, 1.2)),
            TF.ToTensor()
        ])
        self.glyph_embeddings = {font_name: torch.from_numpy(
            vocab_glyph_embedding(font_name, int(font_size * 1.0), use_traditional, font_normalize)).float()
                                 for font_name in self.font_names}
        self.nfonts = len(self.glyph_embeddings)

    def __getitem__(self, index):
        """ Get a sample from the dataset
        """
        num_block = index // self.batch_size
        row_idx = index - self.batch_size * num_block
        column_start = num_block * self.bptt
        column_range = min(self.bptt, self.data.shape[1] - column_start - 1)
        inputs = self.data[row_idx, column_start:column_start + column_range]
        targets = self.data[row_idx, column_start + 1:column_start + column_range + 1]
        images = self.inputs2images(inputs)
        return inputs, targets, images

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        length = self.data.shape[1] // self.bptt
        if length * self.bptt != self.data.shape[1]:
            length += 1
        return length * self.batch_size

    def inputs2images(self, inputs):
        """将输入转化为图片
        Args:
            tensor of shape (seq,)
        Returns:
            tensor of shape (seq, c, h, w), here c == 1. todo:兼容起见可以没有c
        """
        # seq_length = len(inputs)
        images = [embedding.index_select(0, inputs) for embedding in
                  self.glyph_embeddings]  # nfonts, seqlen, fontsize, fontsize
        images = torch.cat(images, 0)  # seqlen * nfonts, fontsize, fontsize
        if self.augment:
            images = [self.transforms(images[idx:idx + 1]) for idx in
                      range(images.shape[0])]  # num, (1, fontsize, fontsize)
            images = torch.cat(images, 0)  # num, fontsize, fontsize
        return images

    def transforms(self, image):
        """
        Augmentation
        :param image: np.array of shape 1, fontsize, fontsize
        :return: augmented image
        """
        image = Image.fromarray(image)
        image = self.transform_funcs(image)
        return image


def batchify(data, bsz):
    """
    取训练集中所有的token，整理成batch_size个长Tensor
    :param data: 整个数据集的所有token
    :param bsz: batch_size
    :return: (long_seq_len, batch_size)
    """
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).contiguous()
    return data


if __name__ == '__main__':
    corpus = Corpus('/data/nfsdata/nlp/datasets/language_modeling/ctb_v6/data/utf8/raw/')
    batch_size = 200
    dataset = YuxianDataSet(corpus.valid, bptt=35, batch_size=batch_size)
    train_loader = D.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print(len(dataset))

    old_dataset = batchify(corpus.valid, batch_size).t()

    #  验证YuxianDataset的遍历和原始一模一样
    def get_batch(source, idx, bptt=35):
        """
        取source中第i个滑动窗口的数据作为data, 第i+1个滑动窗口的数据作为target, (seq_len, batch_size)
        :param source: 整理成batch的数据
        :param idx: 在长tensor上的index
        :return: 在长tensor上以index开头，最长为bptt，总共batch_size个样例组成的输入，以及下一个time_step作为gold
        """
        seq_len = min(bptt, len(source) - 1 - idx)  # 长度一般是bptt，在结尾处委屈求全一下
        data = source[idx: idx + seq_len]
        target = source[idx + 1: idx + 1 + seq_len]
        return data, target


    for idx, sample in enumerate(train_loader):
        data, target, images = sample
        data = data.transpose(0, 1)
        data2, target2 = get_batch(old_dataset, idx=35 * idx, bptt=35)
        print(data[:, 1])
        print(data2[:, 1])
        print('-------')
