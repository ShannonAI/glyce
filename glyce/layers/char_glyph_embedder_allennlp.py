# encoding: utf-8
"""
Model Description

@author: Xiaofei Sun
@contact: adoni1203@gmail.com
@version: 0.1
@license: Apache Licence
@file: glyph_embedding.py
@time: 22/01/2019
"""

import os 
import sys 


root_path = "/".join(os.path.realpath(__file__).split("/")[:-3])
if root_path not in sys.path:
    sys.path.insert(0, root_path)


import torch 
from torch.nn.functional import embedding


import io
import re
import tarfile
import zipfile
import numpy
import logging
import warnings
import itertools
from overrides import overrides
from typing import Optional, Tuple, Sequence, cast, IO, Iterator, Any, NamedTuple


from allennlp.nn import util
from allennlp.data import Vocabulary
from allennlp.common.params import Params
from allennlp.common.util import prepare_environment
prepare_environment(Params(params={}))
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder


from glyce import CharGlyceEmbedding 
from glyce import GlyceConfig 


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name



@TokenEmbedder.register("glyph_char_embedding")
class GlyphCharEmbedding(TokenEmbedder):
    def __init__(self,
                 embedding_dim: int,
                 model_config,
                 vocab_namespace: str = None) -> None:
        super(GlyphCharEmbedding, self).__init__()
        self._vocab_namespace = vocab_namespace
        self.glyph_embedding = CharGlyphEmbedding(model_config)
        self.output_dim = embedding_dim

    @overrides
    def get_output_dim(self) -> int:
        return self.output_dim

    @overrides
    def forward(self, inputs):  # pylint: disable=arguments-differ
        # inputs may have extra dimensions (batch_size, d1, ..., dn, sequence_length),
        # but embedding expects (batch_size, sequence_length), so pass inputs to
        # util.combine_initial_dims (which is a no-op if there are no extra dimensions).
        # Remember the original size.
        original_size = inputs.size()
        inputs = util.combine_initial_dims(inputs)

        emb, glyph_classification_loss = self.glyph_embedding(inputs)


        # Now (if necessary) add back in the extra dimensions.
        embedded = util.uncombine_initial_dims(emb, original_size)

        return embedded, glyph_classification_loss

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'GlyphCharEmbedding':  # type: ignore
        """
        We need the vocabulary here to know how many items we need to embed, and we look for a
        ``vocab_namespace`` key in the parameter dictionary to know which vocabulary to use.  If
        you know beforehand exactly how many embeddings you need, or aren't using a vocabulary
        mapping for the things getting embedded here, then you can pass in the ``num_embeddings``
        key directly, and the vocabulary will be ignored.
        In the configuration file, a file containing pretrained embeddings can be specified
        using the parameter ``"pretrained_file"``.
        It can be the path to a local file or an URL of a (cached) remote file.
        Two formats are supported:
            * hdf5 file - containing an embedding matrix in the form of a torch.Tensor;
            * text file - an utf-8 encoded text file with space separated fields::
                    [word] [dim 1] [dim 2] ...
              The text file can eventually be compressed with gzip, bz2, lzma or zip.
              You can even select a single file inside an archive containing multiple files
              using the URI::
                    "(archive_uri)#file_path_inside_the_archive"
              where ``archive_uri`` can be a file system path or a URL. For example::
                    "(http://nlp.stanford.edu/data/glove.twitter.27B.zip)#glove.twitter.27B.200d.txt"
        """
        # pylint: disable=arguments-differ
        vocab_namespace = params.pop("vocab_namespace", "tokens")
        embedding_dim = params.pop_int('embedding_dim')
        config = GlyceConfig.from_dict(params.pop('config'))
        params.assert_empty(cls.__name__)
        print(vocab._index_to_token.keys())
        config.idx2char = vocab._index_to_token["tokens"]
        return cls(embedding_dim=embedding_dim,
                   vocab_namespace=vocab_namespace,
                   model_config=config)
