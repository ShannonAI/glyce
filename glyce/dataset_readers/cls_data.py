#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
"""
Model Description

@author: Xiaofei Sun
@contact: adoni1203@gmail.com
@version: 0.1
@license: Apache Licence
@file: csv_reader.py
@time: 21/01/2019
"""


import os 
import sys 


root_path = "/".join(os.path.realpath(__file__).split("/")[:-3]) 
if root_path not in sys.path:
    sys.path.insert(0, root_path)


import json 
import logging
import pandas
from tqdm import tqdm
from typing import Dict
from overrides import overrides


from allennlp.common.util import prepare_environment
from allennlp.common.params import Params
prepare_environment(Params(params={}))
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.common import Params
from allennlp.data.tokenizers import Tokenizer, WordTokenizer, CharacterTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("csv_dataset_reader")
class CsvDatasetReader(DatasetReader):
    def __init__(self,
                 max_sentence_length: int,
                 tokenizer: Tokenizer,
                 max_instance: int,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 ) -> None:
        if max_instance > 100000:
            super().__init__(False)
        else:
            super().__init__(False)
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_sentence_length = max_sentence_length
        self.trimmed_count = 0
        self.max_instance = max_instance

    @overrides
    def _read(self, file_path):
        logger.info(f"Loading csv file from {file_path}")
        df = pandas.read_csv(file_path, header=None)
        df = df.fillna("")

        values = df.values[:self.max_instance]
        # values = sorted(values, key=lambda v: len(" ".join(v[1:]).strip()))

        for idx, value in tqdm(enumerate(values), total=min(self.max_instance, len(values))):
            if idx >= self.max_instance:
                break
            label = str(value[0])
            sentence = " ".join(value[1:]).strip()
            # print(sentence)
            if sentence:
                yield self.text_to_instance(sentence, label)
        logger.info(f"{self.trimmed_count} sentences are trimmed to length {self.max_sentence_length}")

    @overrides
    def text_to_instance(self, sentence: str, label: str) -> Instance:
        sentence_tokens = self._tokenizer.tokenize(sentence)
        if len(sentence) > self.max_sentence_length:
            sentence_tokens = sentence_tokens[:self.max_sentence_length]
            self.trimmed_count += 1
        sentence_field = TextField(sentence_tokens, self._token_indexers)
        fields = {'sentence': sentence_field}
        if label is not None:
            fields['label'] = LabelField(label)
        return Instance(fields)
