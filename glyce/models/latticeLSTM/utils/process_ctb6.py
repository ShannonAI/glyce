import os


def convert_file(file_path):
    converted = []
    with open(file_path) as fi:
        for line in fi:
            if line.strip(' \n　\t') and not line.startswith('<'):
                line = line.strip('　\xa0 \n')
                line_str = line.strip().replace(' ', '').replace('　', '')
                line += '\n'
                conv_res = []
                for i, c in enumerate(line):
                    if c not in ' \n　':
                        if i == 0 and line[i + 1] not in ' \n　':
                            conv_res.append('B-SEG')
                        elif i == 0:
                            conv_res.append('S-SEG')
                        elif line[i - 1] not in ' \n　' and line[i + 1] not in ' \n　':
                            conv_res.append('M-SEG')
                        elif line[i - 1] not in ' \n' and line[i + 1] in ' \n　':
                            conv_res.append('E-SEG')
                        elif line[i - 1] in ' \n　' and line[i + 1] not in ' \n　':
                            conv_res.append('B-SEG')
                        elif line[i - 1] in ' \n　' and line[i + 1] in ' \n　':
                            conv_res.append('S-SEG')
                out_line = '\n' + line
                for s, c in zip(line_str, conv_res):
                    out_line += s + ' ' + c + '\n'
                assert len(conv_res) == len(line_str), F'{len(conv_res), len(line_str), line_str, line} not equal v{out_line}v'
                converted.append((conv_res, line_str))
    return converted

def convert_pos(file_path, mode='word'):
    converted = []
    with open(file_path) as fi:
        for line in fi:
            if line.strip() and not line.strip().startswith('<'):
                conv_res = []
                for t in line.strip().split():
                    try:
                        word, pos = t.split('_')
                        if mode == 'word':
                            conv_res.append((word, pos))
                        elif mode == 'char':
                            if len(word) == 1:
                                conv_res.append((word, 'S-' + pos))
                            else:
                                conv_res.append((word[0], 'B-' + pos))
                                for i in range(1, len(word) - 1):
                                    conv_res.append((word[i], 'M-' + pos))
                                conv_res.append((word[-1], 'E-' + pos))
                    except:
                        print(t, line)
                converted.append(([i[1] for i in conv_res], [i[0] for i in conv_res]))
    return converted


def get_index_map():
    ret = []
    for mode in ['trn', 'dev', 'tst']:
        map_dir = F'/data/nfsdata/nlp/datasets/language_modeling/ctb_v6/ChtbSplit/chtb_v6.{mode}.idx'
        idxes = []
        with open(map_dir) as fi:
            for line in fi:
                idxes.append(line.strip())
        ret.append(idxes)
    return ret


def get_index_v9():
    ret = []
    with open('./ctb9_split.txt') as fi:
        for line in fi:
            idxes = []
            for i in line.strip().split(', '):
                if '-' in i:
                    start, end = i.split('-')
                    for j in range(int(start), int(end) + 1):
                        idxes.append('{:04}'.format(j))
                else:
                    idxes.append(i)
            ret.append(idxes)
    return ret


def write_file():
    train_map, dev_map, test_map = get_index_map()
    res = {'train': [], 'dev': [], 'test': []}
    dir_path = '/data/nfsdata/nlp/datasets/language_modeling/ctb_v6/data/utf8/postagged'
    out_path = '/data/nfsdata/nlp/datasets/sequence_labeling/CN_NER/CTB6POS'
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    for fname in os.listdir(dir_path):
        if fname.endswith('.pos'):
            converted = convert_pos(os.path.join(dir_path, fname), 'char')
            if fname[5: 9] in train_map:
                res['train'].extend(converted)
            elif fname[5: 9] in dev_map:
                res['dev'].extend(converted)
            elif fname[5: 9] in test_map:
                res['test'].extend(converted)
            else:
                res['train'].extend(converted)

    for name, res in res.items():
        with open(os.path.join(out_path, F'{name}.char.bmes'), 'w') as fo:
            for conv, sent in res:
                for c, s in zip(conv, sent):
                    fo.write(s + ' ' + c + '\n')
                fo.write('\n')


def write_ud():
    res = {'train': [], 'dev': [], 'test': []}
    dir_path = '/data/nfsdata/nlp/datasets/sequence_labeling/tagger/ud1'
    out_path = '/data/nfsdata/nlp/datasets/sequence_labeling/CN_NER/UD1POS'
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    for fname in os.listdir(dir_path):
        if fname.endswith('.txt'):
            converted = convert_pos(os.path.join(dir_path, fname), 'char')
            if fname == 'train.txt':
                res['train'].extend(converted)
            elif fname == 'dev.txt':
                res['dev'].extend(converted)
            elif fname == 'test.txt':
                res['test'].extend(converted)
            else:
                res['train'].extend(converted)

    for name, res in res.items():
        with open(os.path.join(out_path, F'{name}.char.bmes'), 'w') as fo:
            for conv, sent in res:
                for c, s in zip(conv, sent):
                    fo.write(s + ' ' + c + '\n')
                fo.write('\n')

if __name__ == '__main__':
    write_ud()
