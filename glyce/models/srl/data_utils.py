# Implemented by Charles.

# the CoNLL2009 format is : # you can see here (http://ufal.mff.cuni.cz/conll2009-st/task-description.html)
# ID(1) FORM(2) LEMMA(3) PLEMMA(4) POS(5) PPOS(6) FEAT(7) PFEAT(8) HEAD(9) PHEAD(10) DEPREL(11) PDEPREL(12) FILLPRED(13) PRED(14) APREDs(15) 
# The train and development dataset is satisified with this format
# But in the evaluation part, this(CoNLL 2009) is divided along three dimensions
# -> 1. "Open" vs. "Closed" challenge (In our work we choose Closed)
# -> 2. Joint vs. SRL-only task (In our work we choose SRL-only task)
# -> 3. In-domain vs. out-of-domain data (for Cz, En, Ge only) (In our work we choose In-domain)
# We prepare for Closed SRL-only In-domain task in English and / or Chinese.
# In SRL-only evaluation data, the column 9, 11, 14 , 15 is not provided.
# So we download the evaluation golden data from here(http://ufal.mff.cuni.cz/conll2009-st/eval-data.html)
# the dataset puts in path : ./data/CoNLL2009-Datasets
# train dataset:                |-> conll2009-eng/CoNLL2009-ST-English-train.txt
# dev dataset:                  |-> CoNLL2009-ST-dev-LDC2009E31B/connl_09_st_eng_dev/CoNLL2009-ST-English-development.txt
# evl golden dataset:           |-> CoNLL2009-ST-Gold-Both_tasks/CoNLL2009-ST-evaluation-*.*.*.txt
# eval dataset:                 |-> CoNLL2009-ST-eval-Eng-SRL/CoNLL2009-ST-evaluation-English-SRLonly.txt


import os
import sys 


root_path = "/".join(os.path.realpath(__file__).split("/")[:-4])
if root_path not in sys.path:
    sys.path.insert(0, root_path)


import pickle
import collections
import numpy as np
import random
from collections import Counter
from tqdm import tqdm


_UNK_ = '<UNK>'
_PAD_ = '<PAD>'
_ROOT_ = '<ROOT>'
_NUM_ = '<NUM>'
_DUMMY_ = '<DUMMY>'


class Vertex:
    def __init__(self, id, head):
        self.id = id
        self.head = head
        self.children = []

def is_valid_tree(sentence, rd_node, cur_node):
    if rd_node == 0:
        return True
    if rd_node == cur_node:
        return False
    cur_head = int(sentence[rd_node-1][9])
    if cur_head == cur_node:
        return False
    while cur_head != 0:
        cur_head = int(sentence[cur_head-1][9])
        if cur_head == cur_node:
            return False
    return True

def is_scientific_notation(s):
    s = str(s)
    if s.count(',')>=1:
        sl = s.split(',')
        for item in sl:
            if not item.isdigit():
                return False
        return True   
    return False

def is_float(s):
    s = str(s)
    if s.count('.')==1:
        sl = s.split('.')
        left = sl[0]
        right = sl[1]
        if left.startswith('-') and left.count('-')==1 and right.isdigit():
            lleft = left.split('-')[1]
            if lleft.isdigit() or is_scientific_notation(lleft):
                return True
        elif (left.isdigit() or is_scientific_notation(left)) and right.isdigit():
            return True
    return False

def is_fraction(s):
    s = str(s)
    if s.count('\/')==1:
        sl = s.split('\/')
        if len(sl)== 2 and sl[0].isdigit() and sl[1].isdigit():
            return True  
    if s.count('/')==1:
        sl = s.split('/')
        if len(sl)== 2 and sl[0].isdigit() and sl[1].isdigit():
            return True    
    if s[-1]=='%' and len(s)>1:
        return True
    return False

def is_number(s):
    s = str(s)
    if s.isdigit() or is_float(s) or is_fraction(s) or is_scientific_notation(s):
        return True
    else:
        return False

def make_word_vocab(file_name, output_path, freq_lower_bound=0, quiet=False, use_lower_bound = False, unify_pred=False):

    with open(file_name,'r') as f:
        data = f.readlines()

    origin_data = []
    sentence = []
    for i in range(len(data)):
        if len(data[i].strip())>0:
            sentence.append(data[i].strip().split('\t'))
        else:
            origin_data.append(sentence)
            sentence = []

    if len(sentence) > 0:
        origin_data.append(sentence)

    word_data = []
    for sentence in origin_data:
        for line in sentence:
            if not is_number(line[1].lower()):
                word_data.append(line[1].lower())
                

    word_data_counter = collections.Counter(word_data).most_common()

    if unify_pred:
        word_vocab = [_PAD_,_UNK_,_ROOT_,_NUM_, _DUMMY_]
    else:
        word_vocab = [_PAD_,_UNK_,_ROOT_,_NUM_]

    if use_lower_bound:
        word_vocab = word_vocab + [item[0] for item in word_data_counter if item[1]>=freq_lower_bound]
    else:
        word_vocab = word_vocab + [item[0] for item in word_data_counter]


    word_to_idx = {word:idx for idx,word in enumerate(word_vocab)}

    idx_to_word = {idx:word for idx,word in enumerate(word_vocab)}


    if not quiet:
        print('\tword vocab size:{}'.format(len(word_vocab)))

    if not quiet:
        print('\tdump vocab at:{}'.format(output_path))

    vocab_path = os.path.join(output_path,'word.vocab')

    word2idx_path = os.path.join(output_path,'word2idx.bin')

    idx2word_path = os.path.join(output_path,'idx2word.bin')

    with open(vocab_path, 'w') as f:
        f.write('\n'.join(word_vocab))

    with open(word2idx_path,'wb') as f:
        pickle.dump(word_to_idx,f)

    with open(idx2word_path,'wb') as f:
        pickle.dump(idx_to_word,f)


def make_pos_vocab(file_name, output_path, freq_lower_bound=0, quiet=False, use_lower_bound = False, unify_pred=False):

    with open(file_name,'r') as f:
        data = f.readlines()

    origin_data = []
    sentence = []
    for i in range(len(data)):
        if len(data[i].strip())>0:
            sentence.append(data[i].strip().split('\t'))
        else:
            origin_data.append(sentence)
            sentence = []

    if len(sentence) > 0:
        origin_data.append(sentence)

    pos_data = []
    for sentence in origin_data:
        for line in sentence:
            pos_data.append(line[5])
                

    pos_data_counter = collections.Counter(pos_data).most_common()

    if unify_pred:
        pos_vocab = [_PAD_,_UNK_,_ROOT_, _DUMMY_]
    else:
        pos_vocab = [_PAD_,_UNK_,_ROOT_]

    if use_lower_bound:
        pos_vocab = pos_vocab + [item[0] for item in pos_data_counter if item[1]>=freq_lower_bound]
    else:
        pos_vocab = pos_vocab + [item[0] for item in pos_data_counter]


    pos_to_idx = {pos:idx for idx,pos in enumerate(pos_vocab)}

    idx_to_pos = {idx:pos for idx,pos in enumerate(pos_vocab)}


    if not quiet:
        print('\tpos tag vocab size:{}'.format(len(pos_vocab)))

    if not quiet:
        print('\tdump vocab at:{}'.format(output_path))

    vocab_path = os.path.join(output_path,'pos.vocab')

    pos2idx_path = os.path.join(output_path,'pos2idx.bin')

    idx2pos_path = os.path.join(output_path,'idx2pos.bin')

    with open(vocab_path, 'w') as f:
        f.write('\n'.join(pos_vocab))

    with open(pos2idx_path,'wb') as f:
        pickle.dump(pos_to_idx,f)

    with open(idx2pos_path,'wb') as f:
        pickle.dump(idx_to_pos,f)

def make_lemma_vocab(file_name, output_path, freq_lower_bound=0, quiet=False, use_lower_bound = False, unify_pred=False):

    with open(file_name,'r') as f:
        data = f.readlines()

    origin_data = []
    sentence = []
    for i in range(len(data)):
        if len(data[i].strip())>0:
            sentence.append(data[i].strip().split('\t'))
        else:
            origin_data.append(sentence)
            sentence = []

    if len(sentence) > 0:
        origin_data.append(sentence)

    lemma_data = []
    for sentence in origin_data:
        for line in sentence:
            if not is_number(line[3].lower()):
                lemma_data.append(line[3].lower())
                
    lemma_data_counter = collections.Counter(lemma_data).most_common()

    if unify_pred:
        lemma_vocab = [_PAD_,_UNK_,_ROOT_,_NUM_, _DUMMY_]
    else:
        lemma_vocab = [_PAD_,_UNK_,_ROOT_,_NUM_]

    if use_lower_bound:
        lemma_vocab = lemma_vocab + [item[0] for item in lemma_data_counter if item[1]>=freq_lower_bound]
    else:
        lemma_vocab = lemma_vocab + [item[0] for item in lemma_data_counter]


    lemma_to_idx = {lemma:idx for idx,lemma in enumerate(lemma_vocab)}

    idx_to_lemma = {idx:lemma for idx,lemma in enumerate(lemma_vocab)}


    if not quiet:
        print('\tlemma vocab size:{}'.format(len(lemma_vocab)))

    if not quiet:
        print('\tdump vocab at:{}'.format(output_path))

    vocab_path = os.path.join(output_path,'lemma.vocab')

    lemma2idx_path = os.path.join(output_path,'lemma2idx.bin')

    idx2lemma_path = os.path.join(output_path,'idx2lemma.bin')

    with open(vocab_path, 'w') as f:
        f.write('\n'.join(lemma_vocab))

    with open(lemma2idx_path,'wb') as f:
        pickle.dump(lemma_to_idx,f)

    with open(idx2lemma_path,'wb') as f:
        pickle.dump(idx_to_lemma,f)

def make_deprel_vocab(file_name, output_path, freq_lower_bound=0, quiet=False, use_lower_bound = False, unify_pred=False):

    with open(file_name,'r') as f:
        data = f.readlines()

    origin_data = []
    sentence = []
    for i in range(len(data)):
        if len(data[i].strip())>0:
            sentence.append(data[i].strip().split('\t'))
        else:
            origin_data.append(sentence)
            sentence = []

    if len(sentence) > 0:
        origin_data.append(sentence)

    deprel_data = []
    for sentence in origin_data:
        for line in sentence:
            deprel_data.append(line[11])
                
    deprel_data_counter = collections.Counter(deprel_data).most_common()

    if unify_pred:
        deprel_vocab = [_PAD_,_UNK_, _DUMMY_]
    else:
        deprel_vocab = [_PAD_,_UNK_]

    if use_lower_bound:
        deprel_vocab = deprel_vocab + [item[0] for item in deprel_data_counter if item[1]>=freq_lower_bound]
    else:
        deprel_vocab = deprel_vocab + [item[0] for item in deprel_data_counter]


    deprel_to_idx = {deprel:idx for idx,deprel in enumerate(deprel_vocab)}

    idx_to_deprel = {idx:deprel for idx,deprel in enumerate(deprel_vocab)}


    if not quiet:
        print('\tdeprel vocab size:{}'.format(len(deprel_vocab)))

    if not quiet:
        print('\tdump vocab at:{}'.format(output_path))

    vocab_path = os.path.join(output_path,'deprel.vocab')

    deprel2idx_path = os.path.join(output_path,'deprel2idx.bin')

    idx2deprel_path = os.path.join(output_path,'idx2deprel.bin')

    with open(vocab_path, 'w') as f:
        f.write('\n'.join(deprel_vocab))

    with open(deprel2idx_path,'wb') as f:
        pickle.dump(deprel_to_idx,f)

    with open(idx2deprel_path,'wb') as f:
        pickle.dump(idx_to_deprel,f)

def make_argument_vocab(train_file, dev_file, test_file, output_path, freq_lower_bound=0, quiet=False, use_lower_bound = False, unify_pred=False):

    argument_data = []

    with open(train_file,'r') as f:
        data = f.readlines()

    origin_data = []
    sentence = []
    for i in range(len(data)):
        if len(data[i].strip())>0:
            sentence.append(data[i].strip().split('\t'))
        else:
            origin_data.append(sentence)
            sentence = []

    if len(sentence) > 0:
        origin_data.append(sentence)

    
    for sentence in origin_data:
        for line in sentence:
            if unify_pred:
                if line[13] != '_':
                    argument_data.append(line[13].split('.')[1])
                else:
                    argument_data.append(line[13])
            for i in range(len(line)-14):
                argument_data.append(line[14+i])

    if dev_file is not None:
        with open(dev_file,'r') as f:
            data = f.readlines()

        origin_data = []
        sentence = []
        for i in range(len(data)):
            if len(data[i].strip())>0:
                sentence.append(data[i].strip().split('\t'))
            else:
                origin_data.append(sentence)
                sentence = []

        if len(sentence) > 0:
            origin_data.append(sentence)

        
        for sentence in origin_data:
            for line in sentence:
                if unify_pred:
                    if line[13] != '_':
                        argument_data.append(line[13].split('.')[1])
                    else:
                        argument_data.append(line[13])
                for i in range(len(line)-14):
                    argument_data.append(line[14+i])

    if test_file is not None:
        with open(test_file,'r') as f:
            data = f.readlines()

        origin_data = []
        sentence = []
        for i in range(len(data)):
            if len(data[i].strip())>0:
                sentence.append(data[i].strip().split('\t'))
            else:
                origin_data.append(sentence)
                sentence = []

        if len(sentence) > 0:
            origin_data.append(sentence)

        for sentence in origin_data:
            for line in sentence:
                if unify_pred:
                    if line[13] != '_':
                        argument_data.append(line[13].split('.')[1])
                    else:
                        argument_data.append(line[13])
                for i in range(len(line)-14):
                    argument_data.append(line[14+i])
                
    argument_data_counter = collections.Counter(argument_data).most_common()

    if use_lower_bound:
        argument_vocab = [_PAD_,_UNK_] + [item[0] for item in argument_data_counter if item[1]>=freq_lower_bound]
    else:
        argument_vocab = [_PAD_,_UNK_] + [item[0] for item in argument_data_counter]


    argument_to_idx = {argument:idx for idx,argument in enumerate(argument_vocab)}

    idx_to_argument = {idx:argument for idx,argument in enumerate(argument_vocab)}


    if not quiet:
        print('\targument vocab size:{}'.format(len(argument_vocab)))

    if not quiet:
        print('\tdump vocab at:{}'.format(output_path))

    vocab_path = os.path.join(output_path,'argument.vocab')

    argument2idx_path = os.path.join(output_path,'argument2idx.bin')

    idx2argument_path = os.path.join(output_path,'idx2argument.bin')

    with open(vocab_path, 'w') as f:
        f.write('\n'.join(argument_vocab))

    with open(argument2idx_path,'wb') as f:
        pickle.dump(argument_to_idx,f)

    with open(idx2argument_path,'wb') as f:
        pickle.dump(idx_to_argument,f)

def make_pred_vocab(train_file, dev_file, test_file, output_path, freq_lower_bound=0, quiet=False, use_lower_bound = False):
    
    pred_data = []

    with open(train_file,'r') as f:
        data = f.readlines()

    origin_data = []
    sentence = []
    for i in range(len(data)):
        if len(data[i].strip())>0:
            sentence.append(data[i].strip().split('\t'))
        else:
            origin_data.append(sentence)
            sentence = []

    if len(sentence) > 0:
        origin_data.append(sentence)

    
    for sentence in origin_data:
        for line in sentence:
            if line[12] == 'Y':
                pred_data.append(line[13].split('.')[1])
            else:
                pred_data.append(line[13])

    if dev_file is not None:
        with open(dev_file,'r') as f:
            data = f.readlines()

        origin_data = []
        sentence = []
        for i in range(len(data)):
            if len(data[i].strip())>0:
                sentence.append(data[i].strip().split('\t'))
            else:
                origin_data.append(sentence)
                sentence = []

        if len(sentence) > 0:
            origin_data.append(sentence)

        
        for sentence in origin_data:
            for line in sentence:
                if line[12] == 'Y':
                    pred_data.append(line[13].split('.')[1])
                else:
                    pred_data.append(line[13])

    if test_file is not None:
        with open(test_file,'r') as f:
            data = f.readlines()

        origin_data = []
        sentence = []
        for i in range(len(data)):
            if len(data[i].strip())>0:
                sentence.append(data[i].strip().split('\t'))
            else:
                origin_data.append(sentence)
                sentence = []

        if len(sentence) > 0:
            origin_data.append(sentence)

        for sentence in origin_data:
            for line in sentence:
                if line[12] == 'Y':
                    pred_data.append(line[13].split('.')[1])
                else:
                    pred_data.append(line[13])
                
    pred_data_counter = collections.Counter(pred_data).most_common()

    if use_lower_bound:
        pred_vocab = [_PAD_,_UNK_] + [item[0] for item in pred_data_counter if item[1]>=freq_lower_bound]
    else:
        pred_vocab = [_PAD_,_UNK_] + [item[0] for item in pred_data_counter]


    pred_to_idx = {label:idx for idx,label in enumerate(pred_vocab)}

    idx_to_pred = {idx:label for idx,label in enumerate(pred_vocab)}


    if not quiet:
        print('\tpred vocab size:{}'.format(len(pred_vocab)))

    if not quiet:
        print('\tdump vocab at:{}'.format(output_path))

    vocab_path = os.path.join(output_path,'pred.vocab')

    pred2idx_path = os.path.join(output_path,'pred2idx.bin')

    idx2pred_path = os.path.join(output_path,'idx2pred.bin')

    with open(vocab_path, 'w') as f:
        f.write('\n'.join(pred_vocab))

    with open(pred2idx_path,'wb') as f:
        pickle.dump(pred_to_idx,f)

    with open(idx2pred_path,'wb') as f:
        pickle.dump(idx_to_pred,f)

def count_sentence_predicate(sentence):
    count = 0
    for item in sentence:
        if item[12] == 'Y':
            assert item[12] == 'Y' and item[13] != '_'
            count += 1
    return count

def shrink_pretrained_embedding(train_file, dev_file, test_file, pretrained_file, pretrained_emb_size, output_path, quiet=False):
    word_set = set()
    with open(train_file,'r') as f:
        data = f.readlines()
        for line in data:
            if len(line.strip())>0:
                line = line.strip().split('\t')
                word_set.add(line[1].lower())
    with open(dev_file,'r') as f:
        data = f.readlines()
        for line in data:
            if len(line.strip())>0:
                line = line.strip().split('\t')
                word_set.add(line[1].lower())

    with open(test_file,'r') as f:
        data = f.readlines()
        for line in data:
            if len(line.strip())>0:
                line = line.strip().split('\t')
                word_set.add(line[1].lower())

    pretrained_vocab = [_PAD_,_UNK_,_ROOT_,_NUM_]
    pretrained_embedding = [
                            [0.0]*pretrained_emb_size,
                            [0.0]*pretrained_emb_size,
                            [0.0]*pretrained_emb_size,
                            [0.0]*pretrained_emb_size
                        ]

    with open(pretrained_file,'r',encoding='utf-8') as f:
        for line in f.readlines():
            row = line.strip().split(' ')
            word = row[0].lower()
            if not is_number(word):
                if word in word_set and word not in pretrained_vocab:
                    pretrained_vocab.append(word)
                    weight = [float(item) for item in row[1:]]
                    assert(len(weight)==pretrained_emb_size)
                    pretrained_embedding.append(weight)

    pretrained_embedding = np.array(pretrained_embedding,dtype=float)

    pretrained_to_idx = {word:idx for idx,word in enumerate(pretrained_vocab)}

    idx_to_pretrained = {idx:word for idx,word in enumerate(pretrained_vocab)}

    if not quiet:
        print('\tshrink pretrained vocab size:{}'.format(len(pretrained_vocab)))
        print('\tdataset sum:{} pretrained cover:{} coverage:{:.3}%'.format(len(word_set),len(pretrained_vocab),len(pretrained_vocab)/len(word_set)*100))

    if not quiet:
        print('\tdump vocab at:{}'.format(output_path))

    vocab_path = os.path.join(output_path,'pretrain.vocab')

    pretrain2idx_path = os.path.join(output_path,'pretrain2idx.bin')

    idx2pretrain_path = os.path.join(output_path,'idx2pretrain.bin')

    pretrain_emb_path = os.path.join(output_path,'pretrain.emb.bin')

    with open(vocab_path, 'w') as f:
        f.write('\n'.join(pretrained_vocab))

    with open(pretrain2idx_path,'wb') as f:
        pickle.dump(pretrained_to_idx,f)

    with open(idx2pretrain_path,'wb') as f:
        pickle.dump(idx_to_pretrained,f)

    with open(pretrain_emb_path,'wb') as f:
        pickle.dump(pretrained_embedding,f)

# argument model input
# SENTID(0), PREDID(1), SENTLEN(2), TOKENID(3), RINDEX(4), FLAG(5), FORM(6), LEMMA(7), POS(8), HEAD(9), RHEAD(10), DEPREL(11), LABEL(12)
# argument model input will copy the sentence by count of predicate.
# note: in original input, the RINDEX=TOKENID and HEAD=RHEAD because of no pruning.
def make_dataset_input(dataset_file, output_path, quiet=False, random_error_prob=0.0, deprel_vocab=None, unify_pred=False, predicate_recog_data=None, pickle_dump_path=None): #, use_golden_syn=False, word_filter=None

    if random_error_prob > 0.0:
        assert deprel_vocab is not None

    # load the original dataset
    with open(dataset_file,'r') as f:
        data = f.readlines()

    origin_data = []
    sentence = []
    for i in range(len(data)):
        if len(data[i].strip())>0:
            sentence.append(data[i].strip().split('\t'))
        else:
            origin_data.append(sentence)
            sentence = []

    if len(sentence) > 0:
        origin_data.append(sentence)

    # filter the predicate by recognition results in ConNLL2008 task
    if predicate_recog_data is not None:
        with open(predicate_recog_data, 'r') as f:
            pred_recog_data = f.readlines()

        pred_predict_data = []
        pre_sentence = []

        for i in range(len(pred_recog_data)):
            if len(pred_recog_data[i].strip())>0:
                pre_sentence.append(pred_recog_data[i].strip().split('\t'))
            else:
                pred_predict_data.append(pre_sentence)
                pre_sentence = []

        if len(pre_sentence) > 0:
            pred_predict_data.append(pre_sentence)

        assert len(origin_data) == len(pred_predict_data)

        pred_total = 0
        recog_correct = 0
        recog_label_correct = 0

        for i in range(len(origin_data)):
            assert len(origin_data[i]) == len(pred_predict_data[i])
            for j in range(len(origin_data[i])):
                if origin_data[i][j][12]=='Y':
                    pred_total += 1
                if origin_data[i][j][12]=='Y' and pred_predict_data[i][j][1]!='_':
                    recog_correct += 1
                if origin_data[i][j][12]=='Y' and origin_data[i][j][13].split('.')[1] == pred_predict_data[i][j][1]:
                    recog_label_correct += 1
                if origin_data[i][j][12]=='Y' and pred_predict_data[i][j][1]=='_':
                    origin_data[i][j][12] = '_'
        if not quiet:
            print('\t predicate recognition total:{} correct:{} label correct:{}'.format(pred_total, recog_correct, recog_label_correct))

    # show the output path
    if not quiet:
        print('\tdump dataset input at:{}'.format(output_path))

    uas_count = 0
    las_count = 0
    total = 0
    sentence_idx = 0
    predicate_sum = 0
    argument_sum = 0
    target_sum = 0

    uas_count = 0
    las_count = 0
    total = 0
    if random_error_prob > 0.0:
        for idx in tqdm(range(len(origin_data))):
            sentence = origin_data[idx]
            for item in sentence:
                item[9] = item[8]
                item[11] = item[10]
            for item in sentence:
                rd = random.randint(0,10000)/10000
                if rd < random_error_prob:
                    overflow_ctx = 0
                    while overflow_ctx < overflow_max:
                        rd_head = random.randint(0,len(sentence))
                        if is_valid_tree(sentence, rd_head, int(item[0])):
                            item[9] = str(rd_head)
                            break
                        overflow_ctx += 1
                    rd_rel = deprel_vocab[random.randint(0,len(deprel_vocab)-1)]
                    item[11] = rd_rel
                total += 1
                if item[8] == item[9]:
                    uas_count += 1
                if item[8] == item[9] and item[10] == item[11]:
                    las_count += 1
        if not quiet:
            print('\t after random error UAS:{:.2f} LAS:{:.2f}'.format(uas_count/total*100,las_count/total*100))


    output_data = []
    with open(output_path, 'w') as f:
        for sidx in tqdm(range(len(origin_data))):
            sentence = origin_data[sidx]
            predicate_idx = 0
            for i in range(len(sentence)):
                if sentence[i][12] == 'Y':

                    predicate_sum += 1

                    output_block = []

                    if unify_pred:
                        output_block.append([str(sentence_idx), str(predicate_idx), str(len(sentence)+1), '1', '1', '0', '<DUMMY>', '<DUMMY>', '<DUMMY>', str(int(sentence[i][0])+1), str(int(sentence[i][0])+1), '<DUMMY>', sentence[i][13].split('.')[1]])                    
                    
                    target_sum += len(sentence)
                    for j in range(len(sentence)):
                        ID = sentence[j][0] # ID
                        IS_PRED = 0
                        if i == j:
                            IS_PRED = 1
                        
                        word = sentence[j][1].lower() # FORM

                        if is_number(word):
                            word = _NUM_
                        
                        lemma = sentence[j][3].lower() # PLEMMA
                        if is_number(lemma):
                            lemma = _NUM_

                        pos = sentence[j][5] # PPOS

                        if use_golden_syn:
                            sentence[j][9] = sentence[j][8]
                            sentence[j][11] = sentence[j][10]

                        head = sentence[j][9] # PHEAD

                        deprel = sentence[j][11] # PDEPREL

                        tag = sentence[j][14+predicate_idx] # APRED

                        if tag != '_':
                            argument_sum += 1

                        if unify_pred:
                            output_block.append([str(sentence_idx), str(predicate_idx), str(len(sentence)+1), str(int(ID)+1), str(int(ID)+1), str(IS_PRED), word, lemma, pos, str(int(head)+1), str(int(head)+1), deprel, tag])
                        else:
                            output_block.append([str(sentence_idx), str(predicate_idx), str(len(sentence)), ID, ID, str(IS_PRED), word, lemma, pos, head, head, deprel, tag])
                        
                    if len(output_block)>0:
                        output_data.append(output_block)
                        for item in output_block:
                            f.write('\t'.join(item))
                            f.write('\n')
                        f.write('\n')
                    
                    predicate_idx += 1

            sentence_idx += 1

    if pickle_dump_path is not None:
        dump_data = {'predicate_sum':predicate_sum, 'target_sum':target_sum, 'out_of_target_sum':0, 'argument_sum':argument_sum, 'out_of_pruning_sum':0, 'input_data':output_data, 'K':0}
        with open(pickle_dump_path, 'wb') as df:
            pickle.dump(dump_data, df)

# perform k-order pruning
# SENTID(0), PREDID(1), SENTLEN(2), TOKENID(3), RINDEX(4), FLAG(5), FORM(6), LEMMA(7), POS(8), HEAD(9), RHEAD(10), DEPREL(11), LABEL(12)
# argument model input will copy the sentence by count of predicate.
# note: in pruning input, the RINDEX!=TOKENID and HEAD!=RHEAD because of pruning.
def make_k_order_pruning_dataset_input(dataset_file, output_path, K, quiet=False, random_error_prob=0.0, deprel_vocab=None, overflow_max=10, output_pruning_path=None, unify_pred=False, predicate_recog_data=None, pickle_dump_path=None): # , word_filter=None

    with open(dataset_file,'r') as f:
        data = f.readlines()

    # total_pos_counter = Counter()
    # argument_pos_counter = Counter()
    # pruning_word_pos_counter = Counter()

    # load the dataset
    origin_data = []
    sentence = []
    for i in range(len(data)):
        if len(data[i].strip())>0:
            sentence.append(data[i].strip().split('\t'))
        else:
            origin_data.append(sentence)
            sentence = []

    if len(sentence) > 0:
        origin_data.append(sentence)

    # filter the predicate by recognition results in ConNLL2008 task
    if predicate_recog_data is not None:
        with open(predicate_recog_data, 'r') as f:
            pred_recog_data = f.readlines()

        pred_predict_data = []
        pre_sentence = []

        for i in range(len(pred_recog_data)):
            if len(pred_recog_data[i].strip())>0:
                pre_sentence.append(pred_recog_data[i].strip().split('\t'))
            else:
                pred_predict_data.append(pre_sentence)
                pre_sentence = []

        if len(pre_sentence) > 0:
            pred_predict_data.append(pre_sentence)

        assert len(origin_data) == len(pred_predict_data)

        pred_total = 0
        recog_correct = 0
        recog_label_correct = 0

        for i in range(len(origin_data)):
            assert len(origin_data[i]) == len(pred_predict_data[i])
            for j in range(len(origin_data[i])):
                if origin_data[i][j][12]=='Y':
                    pred_total += 1
                if origin_data[i][j][12]=='Y' and pred_predict_data[i][j][1]!='_':
                    recog_correct += 1
                if origin_data[i][j][12]=='Y' and origin_data[i][j][13].split('.')[1] == pred_predict_data[i][j][1]:
                    recog_label_correct += 1
                if origin_data[i][j][12]=='Y' and pred_predict_data[i][j][1]=='_':
                    origin_data[i][j][12] = '_'
        if not quiet:
            print('\t predicate recognition total:{} correct:{} label correct:{}'.format(pred_total, recog_correct, recog_label_correct))

    if not quiet:
        print('\tdump dataset input at:{}'.format(output_path))

    out_of_pruning_sum = 0
    argument_sum = 0
    target_sum = 0
    out_of_target_sum = 0

    predicate_sum = 0

    # generate syntactic random error
    uas_count = 0
    las_count = 0
    total = 0
    if random_error_prob > 0.0:
        for idx in tqdm(range(len(origin_data))):
            sentence = origin_data[idx]
            for item in sentence:
                item[9] = item[8]
                item[11] = item[10]
            for item in sentence:
                rd = random.randint(0,10000)/10000
                if rd < random_error_prob:
                    overflow_ctx = 0
                    while overflow_ctx < overflow_max:
                        rd_head = random.randint(0,len(sentence))
                        if is_valid_tree(sentence, rd_head, int(item[0])):
                            item[9] = str(rd_head)
                            break
                        overflow_ctx += 1
                    rd_rel = deprel_vocab[random.randint(0,len(deprel_vocab)-1)]
                    item[11] = rd_rel
                total += 1
                if item[8] == item[9]:
                    uas_count += 1
                if item[8] == item[9] and item[10] == item[11]:
                    las_count += 1
        if not quiet:
            print('\t after random error UAS:{:.2f} LAS:{:.2f}'.format(uas_count/total*100,las_count/total*100))

    output_data = []
    sentence_idx = 0
    with open(output_path, 'w') as f:
        for sidx in tqdm(range(len(origin_data))):
            sentence = origin_data[sidx]
            predicate_idx = 0

            # record the syntactic son for every node.(include dummy ROOT)
            syntactic_son_list = [[[] for _ in range(len(sentence)+1)] for _ in range(K)]
            for oidx in range(K):
                for i in range(len(sentence)):
                    if oidx == 0:
                        syntactic_son_list[oidx][int(sentence[i][9])].append(int(sentence[i][0]))
                    else:
                        for k in range(len(syntactic_son_list[oidx-1])):
                            if int(sentence[i][9]) in syntactic_son_list[oidx-1][k]:
                                syntactic_son_list[oidx][k].append(int(sentence[i][0]))
                                break
            
            for i in range(len(sentence)):

                if sentence[i][12] == 'Y':

                    predicate_sum += 1

                    argument_set = set()
                    pruning_set = set()

                    # for this predicate we do pruning by syntactic grammar.
                    current_node_idx = int(sentence[i][0])

                    output_block = []

                    reserve_set = set()

                    while True:

                        for item in syntactic_son_list:
                            reserve_set.update(item[current_node_idx])

                        if current_node_idx != 0:
                            current_node_idx = int(sentence[current_node_idx-1][9])
                        else:
                            break  
                    
                    #
                    # pruning_word_set = set([h+1 for h in range(len(sentence))]) - reserve_set
                    # pruning_word_set = list(pruning_word_set)
                    # for j in range(len(pruning_word_set)):
                    #     pruning_word_pos_counter.update([sentence[pruning_word_set[j]-1][5]])

                    if unify_pred:
                        output_block.append([str(sentence_idx), str(predicate_idx), str(len(sentence)+1), '1', '1', '0', '<DUMMY>', '<DUMMY>', '<DUMMY>', str(int(sentence[i][0])+1), str(int(sentence[i][0])+1), '<DUMMY>', sentence[i][13].split('.')[1]])

                    reserve_set = list(reserve_set)

                    for j in range(len(reserve_set)):

                        item_pos = reserve_set[j]-1

                        ID = sentence[item_pos][0] # ID

                        IS_PRED = 0
                        if i == item_pos:
                            IS_PRED = 1
                        
                        word = sentence[item_pos][1].lower() # FORM

                        # if word_filter is not None and word_filter.get(word) is not None:
                        #     continue

                        if is_number(word):
                            word = _NUM_


                        pruning_set.add(item_pos)
                        
                        lemma = sentence[item_pos][3].lower() # PLEMMA
                        if is_number(lemma):
                            lemma = _NUM_

                        pos = sentence[item_pos][5] # PPOS

                        head = sentence[item_pos][9] #PHEAD

                        deprel = sentence[item_pos][11] # PDEPREL

                        tag = sentence[item_pos][14+predicate_idx] # APRED

                        if unify_pred:
                            output_block.append([str(sentence_idx), str(predicate_idx), str(len(sentence)+1), str(int(ID)+1), str(int(ID)+1), str(IS_PRED), word, lemma, pos, str(int(head)+1), str(int(head)+1), deprel, tag])         
                        else:
                            output_block.append([str(sentence_idx), str(predicate_idx), str(len(sentence)), ID, ID, str(IS_PRED), word, lemma, pos, head, head, deprel, tag])
                         
                    output_block.sort(key=lambda x:int(x[3]))

                    index_map = dict()
                    for idx in range(len(output_block)):
                        index_map[str(output_block[idx][3])] = str(idx+1)

                    for item in output_block:
                        item[4] = index_map[str(item[4])]
                        if item[10] != '0':
                            item[10] = index_map[str(item[10])]
                        f.write('\t'.join(item))
                        f.write('\n')
                    f.write('\n')

                    output_data.append(output_block)

                    target_sum += len(sentence)
                    if unify_pred:
                        out_of_target_sum += (len(sentence)+1-len(output_block))
                    else:
                        out_of_target_sum += (len(sentence)-len(output_block))

                    for j in range(len(sentence)):
                        if sentence[j][14+predicate_idx] != '_':
                            argument_set.add(j)

                    # statistic in argument but not in pruning
                    argument_sum += len(argument_set)
                    out_of_pruning = argument_set - pruning_set
                    out_of_pruning_sum += len(out_of_pruning)

                    predicate_idx += 1

            sentence_idx += 1

    if not quiet:
        print('\n\tpredicate sum:{} target sum:{} prune target sum:{} argument sum:{} prune argument sum:{} argument coverage:{:.2f}'.format(predicate_sum, target_sum, out_of_target_sum, argument_sum, out_of_pruning_sum, (argument_sum-out_of_pruning_sum)/argument_sum*100))  

    if pickle_dump_path is not None:
        dump_data = {'predicate_sum':predicate_sum, 'target_sum':target_sum, 'out_of_target_sum':out_of_target_sum, 'argument_sum':argument_sum, 'out_of_pruning_sum':out_of_pruning_sum, 'input_data':output_data, 'K':K}
        with open(pickle_dump_path, 'wb') as df:
            pickle.dump(dump_data, df)


# predicate model input 
# SENTID(0), SENTLEN(1), TOKENID(2), FLAG(3), FORM(4), LEMMA(5), POS(6), LABEL(7)
def make_pred_dataset_input(dataset_file, output_path, quiet=False):
    with open(dataset_file,'r') as f:
        data = f.readlines()

    origin_data = []
    sentence = []
    for i in range(len(data)):
        if len(data[i].strip())>0:
            sentence.append(data[i].strip().split('\t'))
        else:
            origin_data.append(sentence)
            sentence = []

    if len(sentence) > 0:
        origin_data.append(sentence)

    if not quiet:
        print('\tdump dataset input at:{}'.format(output_path))

    sentence_idx = 0
    with open(output_path, 'w') as f:
        for sentence in origin_data:
            for item in sentence:

                ID = item[0] # ID

                IS_PRED = 0

                if item[12] == 'Y':
                    IS_PRED = 1

                word = item[1].lower() # FORM
                if is_number(word):
                    word = _NUM_

                lemma = item[3].lower() # PLEMMA
                if is_number(lemma):
                    lemma = _NUM_

                pos = item[5] # PPOS

                if item[12] != 'Y':
                    tag = item[13] # PRED
                else:
                    tag = item[13].split('.')[1]

                f.write('\t'.join([str(sentence_idx) , str(len(sentence)), ID, str(IS_PRED), word, lemma, pos, tag]))

                f.write('\n')

            f.write('\n')

            sentence_idx += 1

# predicate recognition model input 
# SENTID(0), SENTLEN(1), TOKENID(2), FORM(3), LEMMA(4), POS(5), LABEL(6)
def make_pred_recog_dataset_input(dataset_file, output_path, quiet=False):
    with open(dataset_file,'r') as f:
        data = f.readlines()

    origin_data = []
    sentence = []
    for i in range(len(data)):
        if len(data[i].strip())>0:
            sentence.append(data[i].strip().split('\t'))
        else:
            origin_data.append(sentence)
            sentence = []

    if len(sentence) > 0:
        origin_data.append(sentence)

    if not quiet:
        print('\tdump dataset input at:{}'.format(output_path))

    sentence_idx = 0
    with open(output_path, 'w') as f:
        for sentence in origin_data:
            for item in sentence:

                ID = item[0] # ID

                word = item[1].lower() # FORM
                if is_number(word):
                    word = _NUM_

                lemma = item[3].lower() # PLEMMA
                if is_number(lemma):
                    lemma = _NUM_

                pos = item[5] # PPOS

                if item[12] != 'Y':
                    tag = item[13] # PRED
                else:
                    tag = item[13].split('.')[1]

                f.write('\t'.join([str(sentence_idx) , str(len(sentence)), ID, word, lemma, pos, tag]))

                f.write('\n')

            f.write('\n')

            sentence_idx += 1


def stat_max_order(dataset_file):
    with open(dataset_file,'r') as f:
        data = f.readlines()

    origin_data = []
    sentence = []
    for i in range(len(data)):
        if len(data[i].strip())>0:
            sentence.append(data[i].strip().split('\t'))
        else:
            origin_data.append(sentence)
            sentence = []

    if len(sentence) > 0:
        origin_data.append(sentence)

    max_order = 0

    for sidx in tqdm(range(len(origin_data))):
        sentence = origin_data[sidx]
        predicate_idx = 0

        for i in range(len(sentence)):
            if sentence[i][12] == 'Y':
                
                argument_set = set()
                for j in range(len(sentence)):
                    if sentence[j][14+predicate_idx] != '_':
                        argument_set.add(int(sentence[j][0]))
                
                cur_order = 1
                while True:
                    found_set = set()
                    son_data = []
                    order_idx = 0
                    while order_idx < cur_order:
                        son_order = [[] for _ in range(len(sentence)+1)]
                        for j in range(len(sentence)):
                            if len(son_data) == 0:
                                son_order[int(sentence[j][9])].append(int(sentence[j][0]))
                            else:
                                for k in range(len(son_data[-1])):
                                    if int(sentence[j][9]) in son_data[-1][k]:
                                        son_order[k].append(int(sentence[j][0]))
                                        break
                        son_data.append(son_order)
                        order_idx += 1
                    
                    current_node = int(sentence[i][0])
                    while True:
                        for item in son_data:
                            found_set.update(item[current_node])
                        if current_node != 0:
                            current_node = int(sentence[current_node-1][9])
                        else:
                            break
                    if len(argument_set - found_set) > 0:
                        cur_order += 1
                    else:
                        break
                if cur_order > max_order:
                    max_order = cur_order
                predicate_idx += 1

    print('max order:{}'.format(max_order))


def load_dataset_input(file_path):
    with open(file_path,'r') as f:
        data = f.readlines()

    origin_data = []
    sentence = []
    for i in range(len(data)):
        if len(data[i].strip())>0:
            sentence.append(data[i].strip().split('\t'))
        else:
            if len(sentence) > 0:
                origin_data.append(sentence)
            sentence = []

    if len(sentence) > 0:
        origin_data.append(sentence)

    return origin_data

def load_dump_data(path):
    return pickle.load(open(path,'rb'))

def load_deprel_vocab(path):
    with open(path,'r') as f:
        data = f.readlines()
    
    data = [item.strip() for item in data if len(item.strip())>0 and item.strip()!=_UNK_ and item.strip()!=_PAD_]

    return data

def output_predict(path, data):
    with open(path, 'w') as f:
        for sentence in data:
            for i in range(len(sentence[0])):
                line = [str(sentence[j][i]) for j in range(len(sentence))]
                f.write('\t'.join(line))
                f.write('\n')
            f.write('\n')



def load_word_filter(path):
    with open(path, 'r') as f:
        data = f.readlines()

    data = [item.strip() for item in data if len(item.strip())>0]


    word_filter = {data[idx]:idx for idx in range(len(data))}

    return word_filter

def replace_syn_dataset(dataset_file, syn_data_path, output_path):
    with open(dataset_file,'r') as f:
        data = f.readlines()

    origin_data = []
    sentence = []
    for i in range(len(data)):
        if len(data[i].strip())>0:
            sentence.append(data[i].strip().split('\t'))
        else:
            origin_data.append(sentence)
            sentence = []

    if len(sentence) > 0:
        origin_data.append(sentence)

    with open(syn_data_path,'r') as f:
        data = f.readlines()

    syn_data = []
    sentence = []
    for i in range(len(data)):
        if len(data[i].strip())>0:
            sentence.append(data[i].strip().split('\t'))
        else:
            syn_data.append(sentence)
            sentence = []

    if len(sentence) > 0:
        syn_data.append(sentence)

    assert len(origin_data) == len(syn_data)

    origin_uas_count = 0
    origin_las_count = 0
    total = 0
    new_uas_count = 0
    new_las_count = 0

    for i in range(len(origin_data)):
        assert len(origin_data[i]) == len(syn_data[i])
        for j in range(len(origin_data[i])):
            total += 1
            if origin_data[i][j][8] == origin_data[i][j][9]:
                origin_uas_count += 1
            if origin_data[i][j][8] == origin_data[i][j][9] and origin_data[i][j][10] == origin_data[i][j][11]:
                origin_las_count += 1

            origin_data[i][j][9] = syn_data[i][j][6]
            origin_data[i][j][11] = syn_data[i][j][7]

            if origin_data[i][j][8] == origin_data[i][j][9]:
                new_uas_count += 1
            if origin_data[i][j][8] == origin_data[i][j][9] and origin_data[i][j][10] == origin_data[i][j][11]:
                new_las_count += 1

    print('\t Origin UAS:{:.2f} LAS:{:.2f}'.format(origin_uas_count/total*100,origin_las_count/total*100))
    print('\t New UAS:{:.2f} LAS:{:.2f}'.format(new_uas_count/total*100,new_las_count/total*100))

    with open(output_path,'w') as f:
        for sentence in origin_data:
            for line in sentence:
                f.write('\t'.join(line))
                f.write('\n')
            f.write('\n')
