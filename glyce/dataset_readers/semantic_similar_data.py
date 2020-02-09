#!/usr/bin/env python
"""
# encoding: utf-8
# @author  : NiePing
# @contact : ping.nie@pku.edu.cn
# @versoin : 1.0
# @license: Apache Licence
# @file    : bq_data.py.py
# @time    : 2019-01-19 13:16
"""

import os
import sys 


root_path = "/".join(os.path.realpath(__file__).split("/")[:-3]) 
if root_path not in sys.path:
    sys.path.insert(0, root_path)


import torch
from torchtext import data
from torchtext.vocab import Vectors


class BQ_torchtext():
    def __init__(self, args):
        self.RAW = data.RawField()
        self.RAW.is_target = False
        tokenize = lambda x: list(x)
        self.TEXT = data.Field(batch_first=True, tokenize=tokenize)
        self.LABEL = data.Field(sequential=False, unk_token=None)
        self.train, self.dev, self.test = data.TabularDataset.splits(
            path='/data/nfsdata/nlp/datasets/sentence_pair/bq_corpus_torch10',
            train='BQ_train.json',
            validation='BQ_dev.json',
            test='BQ_test.json',
            format='json',
            fields={"gold_label": ("label", self.LABEL),
                    "sentence1": ("q1", self.TEXT),
                    "sentence2": ("q2", self.TEXT),
                    "ID": ("id", self.RAW)})

        self.TEXT.build_vocab(self.train, self.dev, self.test, vectors=Vectors("BQ300", args.data))
        self.LABEL.build_vocab(self.train)

        sort_key = lambda x: data.interleave_keys(len(x.q1), len(x.q2))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_iter = data.BucketIterator(self.train, batch_size=args.batch_size, device=device, sort_key=sort_key, sort=True)
        self.dev_iter = data.BucketIterator(self.dev, batch_size=args.batch_size, device=device, sort_key=sort_key, sort=True)
        self.test_iter = data.BucketIterator(self.test, batch_size=args.batch_size, device=device, sort_key=sort_key, sort=True)


def get_data_loader(args):
    bq_data = BQ_torchtext(args)
    return bq_data.train_iter, bq_data.dev_iter, bq_data.test_iter, bq_data.TEXT.vocab, bq_data


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    import torch
    import random
    random.seed(111)
    torch.manual_seed(111)
    print("test: bq_torchtext:===============")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    args = parser.parse_args()
    bq_data = BQ_torchtext(args)
    bq_data.train_iter.shuffle = False
    for _, batch in enumerate(bq_data.test_iter):
        print("test")
        s1 = getattr(batch, 'q1')
        s2 = getattr(batch, 'q2')
        id = getattr(batch, 'id')
        label = getattr(batch, 'label')
        print(id)
        print("test")
    for _, batch in enumerate(bq_data.dev_iter):
        s1 = getattr(batch, 'q1')
        s2 = getattr(batch, 'q2')
        id = getattr(batch, 'id')
        label = getattr(batch, 'label')
        print(id)
        print("dev")
    for _, batch in enumerate(bq_data.train_iter):
        s1 = getattr(batch, 'q1')
        s2 = getattr(batch, 'q2')
        id = getattr(batch, 'id')
        label = getattr(batch, 'label')
        print(id)
        print("train")