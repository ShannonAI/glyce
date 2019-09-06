#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 




# Author: Xiaoy LI 
# Last update: 2019.04.01 
# First create: 2019.04.01 
# Description:
# dataset_processor.py 


import os 
import sys 


root_path = "/".join(os.path.realpath(__file__).split("/")[:-3]) 
if root_path not in sys.path:
    sys.path.insert(0, root_path)


from torch.utils.data import TensorDataset, DataLoader, RandomSampler, \
SequentialSampler 


import csv 
import logging 
import argparse 
import random 
import numpy as np 
from tqdm import tqdm 

from glyce.dataset_readers.bert_data_utils import * 



class MsraNERProcessor(DataProcessor):
    # processor for the MSRA data set 
    def get_train_examples(self, data_dir):
        # see base class 
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.char.bmes")), "train")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.char.bmes")), "test")


    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.char.bmes")), "dev")


    def get_labels(self):
        # see base class 
        return ["S-NS", "B-NS", "M-NS", "E-NS", "S-NR", "B-NR", "M-NR", "E-NR", \
        "S-NT", "B-NT", "M-NT", "E-NT", "O"]

    def _create_examples(self, lines, set_type):
        # create examples for the training and dev sets. 
        examples = []
        for (i, line) in enumerate(lines):
            if line == "\n":
                continue 

            text_a = line[0]
            text_b = None 
            label = line[1]
            label = label.split(" ")
            guid = "{}_{}".format("msra.ner", str(i))
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples 



class OntoNotesNERProcessor(DataProcessor):
    # processor for OntoNotes dataset 
    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.char.bmes")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.char.bmes")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.char.bmes")), "test")

    def get_labels(self):
        # see base class 
        # {'M-GPE', 'S-LOC', 'M-PER', 'B-LOC', 'E-PER', 'M-LOC', 'B-PER', 'B-GPE', 
        # 'S-ORG', 'M-ORG', 'B-ORG', 'S-GPE', 'O', 'E-GPE', 'E-LOC', 'S-PER', 'E-ORG'}
        # GPE, LOC, PER, ORG, O
        return ["O", "S-LOC", "B-LOC", "M-LOC", "E-LOC", \
        "S-PER", "B-PER", "M-PER", "E-PER", \
        "S-GPE", "B-GPE", "M-GPE", "E-GPE", \
        "S-ORG", "B-ORG", "M-ORG", "E-ORG"]

    def _create_examples(self, lines, set_type):
        # create examples for the training and dev sets 
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 0:
            # continue 
            if line == "\n":
                continue 

            # print("check the content of line")
            # print(line)
            # line.split("\t")
            text_a = line[0]
            text_b = None 
            label = line[1]
            label = label.split(" ")
            guid = "{}_{}".format("ontonotes.ner", str(i))
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples 



class ResumeNERProcessor(DataProcessor):
    # processor for the Resume dataset 
    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.char.bmes")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.char.bmes")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.char.bmes")), "test")

    def get_labels(self):
        return ["O", "S-ORG", "S-NAME", "S-RACE", "B-TITLE", "M-TITLE", "E-TITLE", "B-ORG", "M-ORG", "E-ORG", "B-EDU", "M-EDU", "E-EDU", "B-LOC", "M-LOC", "E-LOC", "B-PRO", "M-PRO", "E-PRO", "B-RACE", "M-RACE", "E-RACE", "B-CONT", "M-CONT", "E-CONT", "B-NAME", "M-NAME", "E-NAME", ]


    def _create_examples(self, lines, set_type):
        # create examples for the training and dev sets 
        examples = []
        for (i, line) in enumerate(lines):
            if line == "\n":
                continue 
            text_a = line[0]
            text_b = None 
            label = line[1]
            label = label.split(" ")
            guid = "{}_{}".format("resume.ner", str(i))
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples 
