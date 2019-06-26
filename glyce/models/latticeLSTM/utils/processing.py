import os


def convert_file(file_path):
    converted = []
    with open(file_path) as fi:
        for line in fi:
            if line.strip(' \n　\t'):
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


def write_file():
    dir_path = '/data/nfsdata/nlp/datasets/sequence_labeling/icwb2-data/training'
    out_path = '/data/nfsdata/nlp/datasets/sequence_labeling/CN_NER/MSRCWS'
    for fname in os.listdir(dir_path):
        if fname == 'msr_training.utf8':
            converted = convert_file(os.path.join(dir_path, fname))
            idx = int(len(converted) * 0.9)
            print(fname, idx)
            with open(os.path.join(out_path, fname.replace('utf8', 'train')), 'w') as fo:
                for conv, sent in converted[: idx]:
                    for c, s in zip(conv, sent):
                        fo.write(s + ' ' + c + '\n')
                    fo.write('\n')

            with open(os.path.join(out_path, fname.replace('utf8', 'dev')), 'w') as fo:
                for conv, sent in converted[idx: ]:
                    for c, s in zip(conv, sent):
                        fo.write(s + ' ' + c + '\n')
                    fo.write('\n')

if __name__ == '__main__':
    write_file()
