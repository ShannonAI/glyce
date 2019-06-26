# encoding: utf-8
"""
Model Description

@author: Xiaofei Sun
@contact: adoni1203@gmail.com
@version: 0.1
@license: Apache Licence
@file: basic_classifier.py
@time: 21/01/2019
"""


import os 
import sys 


root_path = "/".join(os.path.realpath(__file__).split("/")[:-4])
if root_path not in sys.path:
    sys.path.insert(0, root_path)


import torch
import torch.nn.functional as F


import numpy
from overrides import overrides
from typing import Dict, Optional


from allennlp.common.util import prepare_environment
from allennlp.common.params import Params
prepare_environment(Params(params={}))
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy, Average


@Model.register("basic_classifier")
class BasicClassifier(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 sentence_encoder: Seq2VecEncoder,
                 classifier_feedforward: FeedForward,
                 label_weight: Dict[str, float] = None,
                 use_label_distribution: bool = False,
                 image_classification_ratio: float = 0.0,
                 decay_every_i_step=100000,
                 decay_ratio=0.8,
                 instance_count=100000,
                 max_epoch=10,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None
                 ) -> None:
        super(BasicClassifier, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.sentence_encoder = sentence_encoder
        self.classifier_feedforward = classifier_feedforward

        if text_field_embedder.get_output_dim() != sentence_encoder.get_input_dim():
            raise ConfigurationError("The output dimension of the text_field_embedder must match the "
                                     "input dimension of the title_encoder. Found {} and {}, "
                                     "respectively.".format(text_field_embedder.get_output_dim(),
                                                            sentence_encoder.get_input_dim()))
        self.metrics = {
                "accuracy": CategoricalAccuracy(),
                "cnn_loss": Average()
        }
        if not use_label_distribution:
            self.loss = torch.nn.CrossEntropyLoss()
        else:
            self.loss = torch.nn.CrossEntropyLoss()
        self.image_classification_ratio = image_classification_ratio
        self.decay_every_i_step = decay_every_i_step
        self.decay_ratio = decay_ratio
        self.training_step = 0
        self.current_ratio = image_classification_ratio
        self.total_steps = max_epoch*instance_count//64
        self.step_every_epoch = instance_count // 64

        print("每个epoch的step数量", self.step_every_epoch)

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                sentence: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        embedded_sentence, multi_task_loss = self.text_field_embedder(sentence)
        sentence_mask = util.get_text_field_mask(sentence)
        encoded_sentence = self.sentence_encoder(embedded_sentence, sentence_mask)
        logits = self.classifier_feedforward(encoded_sentence)
        output_dict = {'logits': logits}
        if label is not None:
            loss = self.loss(logits, label)
            for metric_name in self.metrics:
                if metric_name == "cnn_loss":
                    self.metrics[metric_name](sum(multi_task_loss.values()).item())
                else:
                    self.metrics[metric_name](logits, label)
            output_dict["loss"] = loss + self.current_ratio * sum(multi_task_loss.values())
        if self.training:
            self.training_step += 1
            # self.current_ratio = max(0, self.image_classification_ratio * (1 - self.training_step/self.total_steps))
            if self.training_step % self.step_every_epoch == 0:
                self.current_ratio = self.current_ratio * self.decay_ratio
                print("\n\n\n")
                print("Ration now: ", self.current_ratio)
                print("\n\n\n")
            # if self.training_step > 1563:
            #     self.current_ratio = 0
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``"label"`` key to the dictionary with the result.
        """
        class_probabilities = F.softmax(output_dict['logits'], dim=-1)
        output_dict['class_probabilities'] = class_probabilities

        predictions = class_probabilities.cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in argmax_indices]
        output_dict['label'] = labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}
