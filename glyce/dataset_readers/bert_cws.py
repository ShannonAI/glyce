#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# Author: Xiaoy LI 
# Last update: 2019.04.05 
# First create: 2019.04.05 
# Description:
# cws_dataset_processor.py


import os 
import sys 


root_path = "/".join(os.path.realpath(__file__).split("/")[:-3]) 
if root_path not in sys.path:
    sys.path.insert(0, root_path)


from torch.utils.data import TensorDataset, DataLoader, RandomSampler, \
SequentialSampler


import csv 
import json
import logging 
import random 
import argparse  
import numpy as np
from tqdm import tqdm 


from glyce.dataset_readers.bert_data_utils import *  



class Ctb6CWSProcessor(DataProcessor):
    # processor for CTB6 CWS dataset 
    def get_train_examples(self, data_dir):
        # see base class 
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.char.bmes")), "train")


    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.char.bmes")), "test")


    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.char.bmes")), "dev")


    def get_labels(self):
        return ['E-SEG', 'S-SEG', 'B-SEG', 'M-SEG', ]


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
            guid = "{}_{}".format("ctb6.cws", str(i))
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples 


class PkuCWSProcessor(DataProcessor):
    # processor for PKU CWS dataset 
    def get_train_examples(self, data_dir):
        # see base class 
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.char.bmes")), "train")


    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.char.bmes")), "test")


    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.char.bmes")), "dev") 


    def get_labels(self):
        return ['B-SEG', 'M-SEG', 'S-SEG', 'E-SEG',]


    def _create_examples(self, lines, set_type):
        # create examples for the trainng and dev sets 
        examples = []
        for (i, line) in enumerate(lines):
            if line == "\n":
                continue 

            text_a = line[0]
            text_b = None 
            label = line[1]
            label = label.split(" ")
            guid = "{}_{}".format("pku.cws", str(i))
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples 



class MsrCWSProcessor(DataProcessor):
    # processor for MSR CWS dataset 
    def get_train_examples(self, data_dir):
        # see base class 
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.char.bmes")), "train")

    def get_test_examples(self, data_dir):
        # see base class 
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.char.bmes")), "test")

    def get_dev_examples(self, data_dir):
        # see base class 
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.char.bmes")), "dev")


    def get_labels(self):
        return ['S-SEG', 'M-SEG', 'B-SEG', 'E-SEG',]


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
            guid = "{}_{}".format("mrs.cws", str(i))
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples 

