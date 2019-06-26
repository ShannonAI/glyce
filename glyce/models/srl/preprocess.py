# Implemented by Charles(charlee@sjtu.edu.cn) & Shexia He(heshexia@sjtu.edu.cn).
# This file is used for data process.

import os
import sys 


root_path = "/".join(os.path.realpath(__file__).split("/")[:-4])
if root_path not in sys.path:
    sys.path.insert(0, root_path)


from glyce.models.srl.data_utils import *



def make_dataset():

    base_path = os.path.join(os.path.dirname(__file__),'data/CoNLL-2009-Datasets')

    # because the train and dev file is with full format, wo just copy them
    raw_train_file = os.path.join(base_path,'conll2009-eng/CoNLL2009-ST-English-train.txt')
    raw_dev_file = os.path.join(base_path,'CoNLL2009-ST-dev-LDC2009E31B/connl_09_st_eng_dev/CoNLL2009-ST-English-development.txt')
    
    # because the eval file is lack of 9, 11, 14, 15 so we need to merge them
    raw_eval_file = os.path.join(base_path,'CoNLL2009-ST-test-only-LDC2009E50/CoNLL2009-ST-eval-Eng-SRL/CoNLL2009-ST-evaluation-English-SRLonly.txt')
    raw_eval_file_head = os.path.join(base_path,'CoNLL2009-ST-Gold-Both_tasks/CoNLL2009-ST-evaluation-English.9.HEAD.txt')
    raw_eval_file_deprel = os.path.join(base_path,'CoNLL2009-ST-Gold-Both_tasks/CoNLL2009-ST-evaluation-English.11.DEPREL.txt')
    raw_eval_file_pred_apreds = os.path.join(base_path,'CoNLL2009-ST-Gold-Both_tasks/CoNLL2009-ST-evaluation-English.14-.PRED_APREDs.txt')

    raw_eval_ood_file = os.path.join(base_path,'CoNLL2009-ST-test-only-LDC2009E50/CoNLL2009-ST-eval-Eng-SRL/CoNLL2009-ST-evaluation-English-SRLonly-ood.txt')
    raw_eval_ood_file_head = os.path.join(base_path,'CoNLL2009-ST-Gold-Both_tasks/CoNLL2009-ST-evaluation-English-ood.9.HEAD.txt')
    raw_eval_ood_file_deprel = os.path.join(base_path,'CoNLL2009-ST-Gold-Both_tasks/CoNLL2009-ST-evaluation-English-ood.11.DEPREL.txt')
    raw_eval_ood_file_pred_apreds = os.path.join(base_path,'CoNLL2009-ST-Gold-Both_tasks/CoNLL2009-ST-evaluation-English-ood.14-.PRED_APREDs.txt')


    train_file = os.path.join(os.path.dirname(__file__),'data/conll09-english/conll09_train.dataset')
    dev_file = os.path.join(os.path.dirname(__file__),'data/conll09-english/conll09_dev.dataset')
    test_file = os.path.join(os.path.dirname(__file__),'data/conll09-english/conll09_test.dataset')
    test_ood_file = os.path.join(os.path.dirname(__file__),'data/conll09-english/conll09_test_ood.dataset')

    # for train
    with open(raw_train_file,'r') as fin:
        with open(train_file,'w') as fout:
            while True:
                line = fin.readline()
                if len(line) == 0:
                    break
                fout.write(line)

    # for dev
    with open(raw_dev_file,'r') as fin:
        with open(dev_file,'w') as fout:
            while True:
                line = fin.readline()
                if len(line) == 0:
                    break
                fout.write(line)

    # for test
    with open(raw_eval_file,'r') as fin:
         with open(raw_eval_file_head,'r') as fhead:
              with open(raw_eval_file_deprel,'r') as fdeprel:
                   with open(raw_eval_file_pred_apreds,'r') as fpredapreds:
                        with open(test_file,'w') as fout:
                            while True:
                                raw_line = fin.readline()
                                if len(raw_line) == 0:
                                    break
                                head_line = fhead.readline()
                                deprel_line = fdeprel.readline()
                                pred_apreds_line = fpredapreds.readline()
                                if len(raw_line.strip()) > 0:
                                    raw_line = raw_line.strip().split('\t')
                                    head_line = head_line.strip().split('\t')
                                    deprel_line = deprel_line.strip().split('\t')
                                    pred_apreds_line = pred_apreds_line.strip().split('\t')
                                    raw_line[8] = head_line[0]
                                    raw_line[10] = deprel_line[0]
                                    raw_line += pred_apreds_line
                                    fout.write('\t'.join(raw_line))
                                    fout.write('\n')
                                else:
                                    fout.write(raw_line)

    # for test ood
    with open(raw_eval_ood_file,'r') as fin:
         with open(raw_eval_ood_file_head,'r') as fhead:
              with open(raw_eval_ood_file_deprel,'r') as fdeprel:
                   with open(raw_eval_ood_file_pred_apreds,'r') as fpredapreds:
                        with open(test_ood_file,'w') as fout:
                            while True:
                                raw_line = fin.readline()
                                if len(raw_line) == 0:
                                    break
                                head_line = fhead.readline()
                                deprel_line = fdeprel.readline()
                                pred_apreds_line = fpredapreds.readline()
                                if len(raw_line.strip()) > 0:
                                    raw_line = raw_line.strip().split('\t')
                                    head_line = head_line.strip().split('\t')
                                    deprel_line = deprel_line.strip().split('\t')
                                    pred_apreds_line = pred_apreds_line.strip().split('\t')
                                    raw_line[8] = head_line[0]
                                    raw_line[10] = deprel_line[0]
                                    raw_line += pred_apreds_line
                                    fout.write('\t'.join(raw_line))
                                    fout.write('\n')
                                else:
                                    fout.write(raw_line)


    # because the train and dev file is with full format, wo just copy them
    raw_train_file = os.path.join(base_path,'CoNLL2009-ST-Chinese-train/CoNLL2009-ST-Chinese-train.txt')
    raw_dev_file = os.path.join(base_path,'CoNLL2009-ST-Chinese-dev/CoNLL2009-ST-Chinese-development.txt')
    
    # because the eval file is lack of 9, 11, 14, 15 so we need to merge them
    raw_eval_file = os.path.join(base_path,'CoNLL2009-ST-eval-Ch-SRL/CoNLL2009-ST-evaluation-Chinese-SRLonly.txt')
    raw_eval_file_head = os.path.join(base_path,'CoNLL2009-ST-Gold-Both_tasks/CoNLL2009-ST-evaluation-Chinese.9.HEAD.txt')
    raw_eval_file_deprel = os.path.join(base_path,'CoNLL2009-ST-Gold-Both_tasks/CoNLL2009-ST-evaluation-Chinese.11.DEPREL.txt')
    raw_eval_file_pred_apreds = os.path.join(base_path,'CoNLL2009-ST-Gold-Both_tasks/CoNLL2009-ST-evaluation-Chinese.14-.PRED_APREDs.txt')

    train_file = os.path.join(os.path.dirname(__file__),'data/conll09-chinese/conll09_train.dataset')
    dev_file = os.path.join(os.path.dirname(__file__),'data/conll09-chinese/conll09_dev.dataset')
    test_file = os.path.join(os.path.dirname(__file__),'data/conll09-chinese/conll09_test.dataset')


    # for train
    with open(raw_train_file,'r') as fin:
        with open(train_file,'w') as fout:
            while True:
                line = fin.readline()
                if len(line) == 0:
                    break
                fout.write(line)

    # for dev
    with open(raw_dev_file,'r') as fin:
        with open(dev_file,'w') as fout:
            while True:
                line = fin.readline()
                if len(line) == 0:
                    break
                fout.write(line)

    # for test
    with open(raw_eval_file,'r') as fin:
         with open(raw_eval_file_head,'r') as fhead:
              with open(raw_eval_file_deprel,'r') as fdeprel:
                   with open(raw_eval_file_pred_apreds,'r') as fpredapreds:
                        with open(test_file,'w') as fout:
                            while True:
                                raw_line = fin.readline()
                                if len(raw_line) == 0:
                                    break
                                head_line = fhead.readline()
                                deprel_line = fdeprel.readline()
                                pred_apreds_line = fpredapreds.readline()
                                if len(raw_line.strip()) > 0:
                                    raw_line = raw_line.strip().split('\t')
                                    head_line = head_line.strip().split('\t')
                                    deprel_line = deprel_line.strip().split('\t')
                                    pred_apreds_line = pred_apreds_line.strip().split('\t')
                                    raw_line[8] = head_line[0]
                                    raw_line[10] = deprel_line[0]
                                    raw_line += pred_apreds_line
                                    fout.write('\t'.join(raw_line))
                                    fout.write('\n')
                                else:
                                    fout.write(raw_line)

def stat_dataset(dataset_path):

    with open(dataset_path, 'r') as f:
        data = f.readlines()

        # read data
        sentence_data = []
        sentence = []
        for line in data:
            if len(line.strip()) > 0:
                line = line.strip().split('\t')
                sentence.append(line)
            else:
                sentence_data.append(sentence)
                sentence = []

    predicate_number = 0
    non_predicate_number = 0
    argument_number = 0
    non_argument_number = 0
    predicate_dismatch = 0
    uas_correct = 0
    las_correct = 0
    syntactic_sum = 0
    for sentence in sentence_data:
        for item in sentence:
            syntactic_sum += 1
            if item[8] == item[9]:
                uas_correct += 1
            if item[8] == item[9] and item[10] == item[11]:
                las_correct += 1
            if item[12] == 'Y':
                predicate_number += 1
            else:
                non_predicate_number += 1
            if (item[12] == 'Y' and item[12] == '_') or (item[12] == '_' and item[12] != '_'):
                predicate_dismatch += 1
            for i in range(len(item)-14):
                if item[14+i] != '_':
                    argument_number += 1
                else:
                    non_argument_number += 1
    
    # sentence number
    # predicate number
    # argument number
    print('\tsentence:{} \n\tpredicate:{} non predicate:{} predicate dismatch:{} \n\targument:{} non argument:{} \n\tUAS:{:.2f} LAS:{:.2f}'
            .format(len(sentence_data), predicate_number, non_predicate_number, predicate_dismatch, argument_number, non_argument_number, uas_correct / syntactic_sum * 100 , las_correct / syntactic_sum * 100))

if __name__ == '__main__':

    # make train/dev/test dataset
    # make_dataset()

    train_file = os.path.join(os.path.dirname(__file__),'data/conll09-english/conll09_train.dataset')
    dev_file = os.path.join(os.path.dirname(__file__),'data/conll09-english/conll09_dev.dataset')
    test_file = os.path.join(os.path.dirname(__file__),'data/conll09-english/conll09_test.dataset')
    test_ood_file = os.path.join(os.path.dirname(__file__),'data/conll09-english/conll09_test_ood.dataset')

    # replace_syn_dataset(train_file, os.path.join(os.path.dirname(__file__),'data/biaffine/conll2009_deep_biaffine_train_92.96_98.33.txt'),
    #     os.path.join(os.path.dirname(__file__),'data/conll09-english/conll09_train_92.96.dataset'))
    # replace_syn_dataset(dev_file, os.path.join(os.path.dirname(__file__),'data/biaffine/conll2009_deep_biaffine_dev_88.89_95.02.txt'),
    #     os.path.join(os.path.dirname(__file__),'data/conll09-english/conll09_dev_88.89.dataset'))
    # replace_syn_dataset(test_file, os.path.join(os.path.dirname(__file__),'data/biaffine/conll2009_deep_biaffine_test_90.22_95.56.txt'),
    #     os.path.join(os.path.dirname(__file__),'data/conll09-english/conll09_test_90.22.dataset'))

    # train_file = os.path.join(os.path.dirname(__file__),'data/conll09-english/conll09_train_92.96.dataset')
    # dev_file = os.path.join(os.path.dirname(__file__),'data/conll09-english/conll09_dev_88.89.dataset')
    # test_file = os.path.join(os.path.dirname(__file__),'data/conll09-english/conll09_test_90.22.dataset')

    # stat_max_order(train_file)
    # stat_max_order(dev_file)
    # stat_max_order(test_file)

    # word_filter = load_word_filter(os.path.join(os.path.dirname(__file__),'data/word_filter.vocab'))

    # without pruning
    # make_dataset_input

    make_dataset_input(train_file, os.path.join(os.path.dirname(__file__),'temp/train.input'), unify_pred=False)
    make_dataset_input(dev_file, os.path.join(os.path.dirname(__file__),'temp/dev.input'), unify_pred=False)
    make_dataset_input(test_file, os.path.join(os.path.dirname(__file__),'temp/test.input'), unify_pred=False)
    make_dataset_input(test_ood_file, os.path.join(os.path.dirname(__file__),'temp/test_ood.input'), unify_pred=False)

    # make_k_order_pruning_dataset_input(train_file, os.path.join(os.path.dirname(__file__),'temp/train.1.order.pruning.input'), 1, unify_pred=False)
    # make_k_order_pruning_dataset_input(dev_file, os.path.join(os.path.dirname(__file__),'temp/dev.1.order.pruning.input'), 1, unify_pred=False)
    # make_k_order_pruning_dataset_input(test_file, os.path.join(os.path.dirname(__file__),'temp/test.1.order.pruning.input'), 1, unify_pred=False)
    # make_k_order_pruning_dataset_input(test_ood_file, os.path.join(os.path.dirname(__file__),'temp/test_ood.10.order.pruning.input'), 10, unify_pred=False)

    #statistic train/dev/test predicate information
    # print('-- statistic dataset information --')
    # print('train:')
    # stat_dataset(train_file)
    # print('dev:')
    # stat_dataset(dev_file)
    # print('test:')
    # stat_dataset(test_file)

    # make word/pos/lemma/deprel/argument vocab
    print('\n-- making (word/lemma/pos/argument) vocab --')
    vocab_path = os.path.join(os.path.dirname(__file__),'temp')
    print('word:')
    make_word_vocab(train_file,vocab_path, unify_pred=False)
    print('pos:')
    make_pos_vocab(train_file,vocab_path, unify_pred=False)
    print('lemma:')
    make_lemma_vocab(train_file,vocab_path, unify_pred=False)
    print('deprel:')
    make_deprel_vocab(train_file,vocab_path, unify_pred=False)
    print('argument:')
    make_argument_vocab(train_file, dev_file, test_file, vocab_path, unify_pred=False)
    print('predicate:')
    make_pred_vocab(train_file, dev_file, test_file, vocab_path)

    # shrink pretrained embeding
    print('\n-- shrink pretrained embeding --')
    pretrain_file = os.path.join(os.path.dirname(__file__),'data/glove.6B.100d.txt')#words.vector
    pretrained_emb_size = 100
    pretrain_path = os.path.join(os.path.dirname(__file__),'temp')
    shrink_pretrained_embedding(train_file,dev_file,test_file,pretrain_file,pretrained_emb_size, pretrain_path)

    # 
    # make_pred_dataset_input(train_file, os.path.join(os.path.dirname(__file__),'temp/pred_train.input'))
    # make_pred_dataset_input(dev_file, os.path.join(os.path.dirname(__file__),'temp/pred_dev.input'))
    # make_pred_dataset_input(test_file, os.path.join(os.path.dirname(__file__),'temp/pred_test.input'))

    # make_pred_recog_dataset_input(train_file, os.path.join(os.path.dirname(__file__),'temp/pred_recog_train.input'))
    # make_pred_recog_dataset_input(dev_file, os.path.join(os.path.dirname(__file__),'temp/pred_recog_dev.input'))
    # make_pred_recog_dataset_input(test_file, os.path.join(os.path.dirname(__file__),'temp/pred_recog_test.input'))
    
