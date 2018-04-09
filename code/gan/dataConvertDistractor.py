from random import shuffle
import random
random.seed(0)


def recover(s, delimiter='_', unk='<a>'):
    s = s.split(delimiter)
    # s = [x.split('-') for x in s if x != unk]
    # s = [x for sublist in s for x in sublist]
    return s


def padding(s, max_length=100, unk='<a>'):
    # pad
    if len(s) > max_length:
        s = s[:max_length]
    else:
        s = s + [unk] * (max_length - len(s))
    s = delimiter.join(s)
    return s


def load_vocab(vocab_path):
    vocab = []
    with open(vocab_path, 'r') as fin:
        for line in fin:
            line = line.strip()
            line = line.split()
            vocab.append(line)
    return vocab

data_path = './distractorQA/mcq_neg/'
out_path = './distractorQA/mcq_neg/'
source_path = ['train.data', 'valid_neg_2nd_2500.data', 'test_neg_2nd_2500.data']
data_split = ['train', 'dev', 'test']
vocab_path = 'combine.vocab'
vocab = load_vocab(data_path + vocab_path)
max_length = 80
max_length_dis = 20
delimiter = '_'
unk = '<a>'
num_negative = len(vocab)

for it, source in enumerate(source_path):
    path = data_path + source
    dataset = data_split[it]
    with open(out_path + dataset, 'w') as fout:
        with open(path, 'r') as fin:
            q_prev = ''
            dlist = []
            for i, line in enumerate(fin):
                line = line.strip().lower()
                line = line.split(' ')
                label, q, a, dis = line[0], line[1], line[2], line[3]
                if i == 0:
                    print(len(q.split('_')), len(a.split('_')), len(dis.split('_')))
                # to original sentence
                q = recover(q, delimiter, unk)
                a = recover(a, delimiter, unk)
                dis = recover(dis, delimiter, unk)
                if q == q_prev:
                    dlist.append(dis)
                else:
                    q_prev = q
                    dlist = [dis]
                q = padding(q, max_length, unk)
                a = padding(a, max_length_dis, unk)
                dis = padding(dis, max_length_dis, unk)
                fout.write('{} {} {} {}\n'.format(label, q, a, dis))
                # # write
                # if dataset == 'train':
                #     # positive
                #     if label == '1':
                #         fout.write('{} {} {} {}\n'.format(label, q, a, dis))
                # else:
                #     # negative
                #     if len(dlist) == 1:
                #         count = 0
                #         shuffle(vocab)
                #         for v in vocab:
                #             if v not in dlist:
                #                 count += 1
                #                 v = padding(v, max_length_dis, unk)
                #                 fout.write('0 {} {} {}\n'.format(q, a, v))
                #                 if count >= num_negative:
                #                     break
                #     # positive
                #     if label == '1':
                #         fout.write('{} {} {} {}\n'.format(label, q, a, dis))
