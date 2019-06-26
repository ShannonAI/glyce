# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-06-14 17:34:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-01-29 15:26:51


import os 
import sys 


root_path = "/".join(os.path.realpath(__file__).split("/")[:-5])
if root_path not in sys.path:
    sys.path.insert(0, root_path)  


import logging


from glyce.models.latticeLSTM.utils.functions import *
from glyce.models.latticeLSTM.utils.gazetteer import Gazetteer

# START = "</s>"
# UNKNOWN = "</unk>"
PADDING = "</pad>"
NULLKEY = "-null-"
logger = logging.getLogger(__name__)


class Data(object):
    def __init__(self):
        self.MAX_SENTENCE_LENGTH = 250
        self.MAX_WORD_LENGTH = -1
        self.number_normalized = True
        self.norm_word_emb = True
        self.norm_biword_emb = True
        self.norm_gaz_emb = False
        self.word_alphabet = Alphabet('word')
        self.biword_alphabet = Alphabet('biword')
        self.char_alphabet = Alphabet('character')
        # self.word_alphabet.add(START)
        # self.word_alphabet.add(UNKNOWN)
        # self.char_alphabet.add(START)
        # self.char_alphabet.add(UNKNOWN)
        # self.char_alphabet.add(PADDING)
        self.label_alphabet = Alphabet('label', True)
        self.gaz_lower = False
        self.gaz = Gazetteer(self.gaz_lower)
        self.gaz_alphabet = Alphabet('gaz')
        self.HP_fix_gaz_emb = False
        self.HP_use_gaz = True

        self.tagScheme = "NoSeg"
        self.char_features = "LSTM"

        self.train_texts = []
        self.dev_texts = []
        self.test_texts = []
        self.raw_texts = []

        self.train_Ids = []
        self.dev_Ids = []
        self.test_Ids = []
        self.raw_Ids = []
        self.use_bigram = True
        self.word_emb_dim = 50
        self.biword_emb_dim = 50
        self.char_emb_dim = 30
        self.gaz_emb_dim = 50
        self.gaz_dropout = 0.5
        self.pretrain_word_embedding = None
        self.pretrain_biword_embedding = None
        self.pretrain_gaz_embedding = None
        self.label_size = 0
        self.word_alphabet_size = 0
        self.biword_alphabet_size = 0
        self.char_alphabet_size = 0
        self.label_alphabet_size = 0

        #  hyperparameters
        self.HP_iteration = 50
        self.HP_batch_size = 1
        self.HP_char_hidden_dim = 50
        self.HP_hidden_dim = 200
        self.HP_dropout = 0.5
        self.HP_lstm_layer = 1
        self.HP_bilstm = True
        self.HP_use_char = False
        self.HP_gpu = True
        self.HP_lr = 0.015
        self.HP_lr_decay = 0.05
        self.HP_clip = 5.0
        self.HP_momentum = 0
        self.HP_use_glyph = True
        self.HP_glyph_ratio = 0.05
        self.HP_font_channels = 2
        self.HP_glyph_highway = False
        self.HP_glyph_embsize = 64
        self.HP_glyph_output_size = 64
        self.HP_glyph_dropout = 0.7
        self.HP_glyph_cnn_dropout = 0.5
        self.HP_glyph_layernorm = False
        self.HP_glyph_batchnorm = False

    def show_data_summary(self):
        logger.info("DATA SUMMARY START:")
        logger.info(("     Tag          scheme: %s" % (self.tagScheme)))
        logger.info(("     MAX SENTENCE LENGTH: %s" % (self.MAX_SENTENCE_LENGTH)))
        logger.info(("     MAX   WORD   LENGTH: %s" % (self.MAX_WORD_LENGTH)))
        logger.info(("     Number   normalized: %s" % (self.number_normalized)))
        logger.info(("     Use          bigram: %s" % (self.use_bigram)))
        logger.info(("     Word  alphabet size: %s" % (self.word_alphabet_size)))
        logger.info(("     Biword alphabet size: %s" % (self.biword_alphabet_size)))
        logger.info(("     Char  alphabet size: %s" % (self.char_alphabet_size)))
        logger.info(("     Gaz   alphabet size: %s" % (self.gaz_alphabet.size())))
        logger.info(("     Label alphabet size: %s" % (self.label_alphabet_size)))
        logger.info(("     Word embedding size: %s" % (self.word_emb_dim)))
        logger.info(("     Biword embedding size: %s" % (self.biword_emb_dim)))
        logger.info(("     Char embedding size: %s" % (self.char_emb_dim)))
        logger.info(("     Gaz embedding size: %s" % (self.gaz_emb_dim)))
        logger.info(("     Norm     word   emb: %s" % (self.norm_word_emb)))
        logger.info(("     Norm     biword emb: %s" % (self.norm_biword_emb)))
        logger.info(("     Norm     gaz    emb: %s" % (self.norm_gaz_emb)))
        logger.info(("     Norm   gaz  dropout: %s" % (self.gaz_dropout)))
        logger.info(("     Train instance number: %s" % (len(self.train_texts))))
        logger.info(("     Dev   instance number: %s" % (len(self.dev_texts))))
        logger.info(("     Test  instance number: %s" % (len(self.test_texts))))
        logger.info(("     Raw   instance number: %s" % (len(self.raw_texts))))
        logger.info(("     Hyperpara  iteration: %s" % (self.HP_iteration)))
        logger.info(("     Hyperpara  batch size: %s" % (self.HP_batch_size)))
        logger.info(("     Hyperpara          lr: %s" % (self.HP_lr)))
        logger.info(("     Hyperpara    lr_decay: %s" % (self.HP_lr_decay)))
        logger.info(("     Hyperpara     HP_clip: %s" % (self.HP_clip)))
        logger.info(("     Hyperpara    momentum: %s" % (self.HP_momentum)))
        logger.info(("     Hyperpara  hidden_dim: %s" % (self.HP_hidden_dim)))
        logger.info(("     Hyperpara     dropout: %s" % (self.HP_dropout)))
        logger.info(("     Hyperpara  lstm_layer: %s" % (self.HP_lstm_layer)))
        logger.info(("     Hyperpara      bilstm: %s" % (self.HP_bilstm)))
        logger.info(("     Hyperpara         GPU: %s" % (self.HP_gpu)))
        logger.info(("     Hyperpara     use_gaz: %s" % (self.HP_use_gaz)))
        logger.info(("     Hyperpara fix gaz emb: %s" % (self.HP_fix_gaz_emb)))
        logger.info(("     Hyperpara    use_char: %s" % (self.HP_use_char)))
        if self.HP_use_char:
            logger.info(("             Char_features: %s" % (self.char_features)))
        logger.info("DATA SUMMARY END.")
        sys.stdout.flush()

    def refresh_label_alphabet(self, input_file):
        old_size = self.label_alphabet_size
        self.label_alphabet.clear(True)
        in_lines = open(input_file, 'r').readlines()
        for line in in_lines:
            if len(line) > 2:
                pairs = line.strip().split()
                label = pairs[-1]
                self.label_alphabet.add(label)
        self.label_alphabet_size = self.label_alphabet.size()
        startS = False
        startB = False
        for label, _ in self.label_alphabet.items():
            if "S-" in label.upper():
                startS = True
            elif "B-" in label.upper():
                startB = True
        if startB:
            if startS:
                self.tagScheme = "BMES"
            else:
                self.tagScheme = "BIO"
        self.fix_alphabet()
        logger.info(("Refresh label alphabet finished: old:%s -> new:%s" % (old_size, self.label_alphabet_size)))

    def build_alphabet(self, input_file):
        in_lines = open(input_file, encoding="UTF-8").readlines()
        for idx in range(len(in_lines)):
            line = in_lines[idx]
            if len(line) > 2:
                pairs = line.strip().split()
                # word = pairs[0].decode('utf-8')
                word = pairs[0]
                if self.number_normalized:
                    word = normalize_word(word)
                label = pairs[-1]
                self.label_alphabet.add(label)
                self.word_alphabet.add(word)
                if idx < len(in_lines) - 1 and len(in_lines[idx + 1]) > 2:
                    # biword = word + in_lines[idx+1].strip().split()[0].decode('utf-8')
                    biword = word + in_lines[idx + 1].strip().split()[0]
                else:
                    biword = word + NULLKEY
                self.biword_alphabet.add(biword)
                for char in word:
                    self.char_alphabet.add(char)
        self.word_alphabet_size = self.word_alphabet.size()
        self.biword_alphabet_size = self.biword_alphabet.size()
        self.char_alphabet_size = self.char_alphabet.size()
        self.label_alphabet_size = self.label_alphabet.size()
        startS = False
        startB = False
        logger.info(self.label_alphabet)
        for label, _ in self.label_alphabet.instance2index.items():
            # for label, _ in self.label_alphabet.:
            if "S-" in label.upper():
                startS = True
            elif "B-" in label.upper():
                startB = True
        if startB:
            if startS:
                self.tagScheme = "BMES"
            else:
                self.tagScheme = "BIO"

    def build_gaz_file(self, gaz_file):
        if gaz_file:
            fins = open(gaz_file, encoding="UTF-8").readlines()
            for fin in fins:
                # fin = fin.strip().split()[0].decode('utf-8')
                fin = fin.strip().split()[0]
                if fin:
                    self.gaz.insert(fin, "one_source")
            logger.info(F"Load gaz file: {gaz_file} total size: {self.gaz.size()}")
        else:
            logger.info("Gaz file is None, load nothing")

    def build_gaz_alphabet(self, input_file):
        in_lines = open(input_file, encoding="UTF-8").readlines()
        word_list = []
        for line in in_lines:
            if len(line) > 3:
                # word = line.split()[0].decode('utf-8')
                word = line.split()[0]
                if self.number_normalized:
                    word = normalize_word(word)
                word_list.append(word)
            else:
                w_length = len(word_list)
                for idx in range(w_length):
                    matched_entity = self.gaz.enumerateMatchList(word_list[idx:])
                    for entity in matched_entity:
                        # logger.info entity, self.gaz.searchId(entity),self.gaz.searchType(entity)
                        self.gaz_alphabet.add(entity)
                word_list = []
        logger.info(F"gaz alphabet size: {self.gaz_alphabet.size()}")

    def fix_alphabet(self):
        self.word_alphabet.close()
        self.biword_alphabet.close()
        self.char_alphabet.close()
        self.label_alphabet.close()
        self.gaz_alphabet.close()

    def build_word_pretrain_emb(self, emb_path):
        logger.info("build word pretrain emb...")
        self.pretrain_word_embedding, self.word_emb_dim = build_pretrain_embedding(emb_path, self.word_alphabet,
                                                                                   self.word_emb_dim,
                                                                                   self.norm_word_emb)

    def build_biword_pretrain_emb(self, emb_path):
        logger.info("build biword pretrain emb...")
        self.pretrain_biword_embedding, self.biword_emb_dim = build_pretrain_embedding(emb_path, self.biword_alphabet,
                                                                                       self.biword_emb_dim,
                                                                                       self.norm_biword_emb)

    def build_gaz_pretrain_emb(self, emb_path):
        logger.info("build gaz pretrain emb...")
        self.pretrain_gaz_embedding, self.gaz_emb_dim = build_pretrain_embedding(emb_path, self.gaz_alphabet,
                                                                                 self.gaz_emb_dim, self.norm_gaz_emb)

    def generate_instance(self, input_file, name):
        self.fix_alphabet()
        if name == "train":
            self.train_texts, self.train_Ids = read_seg_instance(input_file, self.word_alphabet, self.biword_alphabet,
                                                                 self.char_alphabet, self.label_alphabet,
                                                                 self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "dev":
            self.dev_texts, self.dev_Ids = read_seg_instance(input_file, self.word_alphabet, self.biword_alphabet,
                                                             self.char_alphabet, self.label_alphabet,
                                                             self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "test":
            self.test_texts, self.test_Ids = read_seg_instance(input_file, self.word_alphabet, self.biword_alphabet,
                                                               self.char_alphabet, self.label_alphabet,
                                                               self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "raw":
            self.raw_texts, self.raw_Ids = read_seg_instance(input_file, self.word_alphabet, self.biword_alphabet,
                                                             self.char_alphabet, self.label_alphabet,
                                                             self.number_normalized, self.MAX_SENTENCE_LENGTH)
        else:
            logger.info(("Error: you can only generate train/dev/test instance! Illegal input:%s" % (name)))

    def generate_instance_with_gaz(self, input_file, name):
        self.fix_alphabet()
        if name == "train":
            self.train_texts, self.train_Ids = read_instance_with_gaz(input_file, self.gaz, self.word_alphabet,
                                                                      self.biword_alphabet, self.char_alphabet,
                                                                      self.gaz_alphabet, self.label_alphabet,
                                                                      self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "dev":
            self.dev_texts, self.dev_Ids = read_instance_with_gaz(input_file, self.gaz, self.word_alphabet,
                                                                  self.biword_alphabet, self.char_alphabet,
                                                                  self.gaz_alphabet, self.label_alphabet,
                                                                  self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "test":
            self.test_texts, self.test_Ids = read_instance_with_gaz(input_file, self.gaz, self.word_alphabet,
                                                                    self.biword_alphabet, self.char_alphabet,
                                                                    self.gaz_alphabet, self.label_alphabet,
                                                                    self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "raw":
            self.raw_texts, self.raw_Ids = read_instance_with_gaz(input_file, self.gaz, self.word_alphabet,
                                                                  self.biword_alphabet, self.char_alphabet,
                                                                  self.gaz_alphabet, self.label_alphabet,
                                                                  self.number_normalized, self.MAX_SENTENCE_LENGTH)
        else:
            logger.info(("Error: you can only generate train/dev/test instance! Illegal input:%s" % (name)))

    def write_decoded_results(self, output_file, predict_results, name):
        fout = open(output_file, 'w')
        sent_num = len(predict_results)
        content_list = []
        if name == 'raw':
            content_list = self.raw_texts
        elif name == 'test':
            content_list = self.test_texts
        elif name == 'dev':
            content_list = self.dev_texts
        elif name == 'train':
            content_list = self.train_texts
        else:
            logger.info("Error: illegal name during writing predict result, name should be within train/dev/test/raw !")
        assert (sent_num == len(content_list))
        for idx in range(sent_num):
            sent_length = len(predict_results[idx])
            for idy in range(sent_length):
                ## content_list[idx] is a list with [word, char, label]
                # fout.write(content_list[idx][0][idy].encode('utf-8') + " " + predict_results[idx][idy] + '\n')
                fout.write(content_list[idx][0][idy] + " " + predict_results[idx][idy] + '\n')

            fout.write('\n')
        fout.close()
        logger.info(("Predict %s result has been written into file. %s" % (name, output_file)))
