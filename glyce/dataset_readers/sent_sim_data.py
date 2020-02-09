# encoding: utf-8
"""
@author: Yuxian Meng 
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: data_reader
@time: 2019/1/23 11:50

    这一行开始写关于本文件的说明与解释
"""

import os 
import sys 


root_path = "/".join(os.path.realpath(__file__).split("/")[:-3]) 
if root_path not in sys.path:
    sys.path.insert(0, root_path)


import json
import logging
from typing import Dict, List
from overrides import overrides


from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter, ChineseSimpleWordSplitter
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer


logger = logging.getLogger(__name__)


@DatasetReader.register("sim_sentence_reader")
class SimSentenceReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_tokens: int = 1000) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer(word_splitter=ChineseSimpleWordSplitter())
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_tokens = max_tokens

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file:
                line = line.strip("\n")
                if not line:
                    continue
                article_json = json.loads(line)
                s1 = article_json['sentence1'][: self.max_tokens]
                s2 = article_json['sentence2'][: self.max_tokens]
                label = str(article_json['gold_label'])
                yield self.text_to_instance(s1, s2, label)

    @overrides
    def text_to_instance(self, s1: str, s2: str, label: str = None, split_token: str = None) -> Instance:  # type: ignore
        tokens1 = self._tokenizer.tokenize(s1)
        tokens2 = self._tokenizer.tokenize(s2)
        # self._truncate_seq_pair(tokens1, tokens2, self.max_tokens)
        tokens1_field = TextField(tokens1, self._token_indexers)
        tokens2_field = TextField(tokens2, self._token_indexers)
        fields = {'premise': tokens1_field, 'hypothesis': tokens2_field}
        if label is not None:
            fields['label'] = LabelField(label)
        return Instance(fields)

    @staticmethod
    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()