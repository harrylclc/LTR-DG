import random
import os
import pickle
import math
import numpy as np


def loadGloVe(filename, vocab_exist=None):
    vocab = []
    vocab_dict = {}
    embd = []
    with open(filename, 'r') as fin:
        for line in fin:
            row = line.strip().split(' ')
            if vocab_exist is None or row[0] in vocab_exist:
                vocab.append(row[0])
                vocab_dict[row[0]] = len(vocab) - 1
                embd.append(row[1:])
        print('Loaded GloVe!')
    embd = np.array(embd)
    return vocab, vocab_dict, embd


def build_vocab(dataset, pretrained_embeddings_path):
    vocab = None
    if os.path.isfile('{}/vocab.pkl'.format(dataset)):
        print('loading saved vocab...')
        with open('{}/vocab.pkl'.format(dataset), 'rb') as fin:
            vocab = pickle.load(fin)
    else:
        code = int(0)
        vocab = {}
        vocab['UNKNOWN'] = code
        code += 1
        filenames = ['{}/train.data'.format(dataset), '{}/valid.data'.format(dataset), '{}/test.data'.format(dataset)]
        for filename in filenames:
            for line in open(filename):
                items = line.strip().split(' ')
                for i in range(2, 3):
                    words = items[i].split('_')
                    for word in words:
                        if word not in vocab:
                            vocab[word] = code
                            code += 1
    embd = None
    if os.path.isfile('{}/embd.pkl'.format(dataset)):
        print('loading saved embd...')
        with open('{}/embd.pkl'.format(dataset), 'rb') as fin:
            embd = pickle.load(fin)
    elif len(pretrained_embeddings_path) > 0:
        vocab_all, vocab_dict_all, embd_all = loadGloVe(pretrained_embeddings_path, vocab)
        embd = []
        for k, v in vocab.items():
            try:
                index = vocab_dict_all[k]
                embd.append(embd_all[index])
            except:
                embd.append(np.random.uniform(-0.05, 0.05, (embd_all.shape[1])))
        embd = np.array(embd)
    return vocab, embd


def loadTestSet(dataset, filename):
    testList = []
    for line in open('{}/{}'.format(dataset, filename)):
        testList.append(line.strip().lower())
    return testList


def loadCandidateSamples(q, a, distractor, candidates, vocab, max_sequence_length_q=100, max_sequence_length_a=100):
    samples = []
    for neg in candidates:
        samples.append((encode_sent(vocab, q, max_sequence_length_q),
                        encode_sent(vocab, a, max_sequence_length_a),
                        encode_sent(vocab, distractor, max_sequence_length_a),
                        encode_sent(vocab, neg, max_sequence_length_a)))
    return samples


def read_raw(dataset):
    raw = []
    raw_dict = {}  # pos_index for each answer grouped by questions
    with open('{}/train.data'.format(dataset), 'r') as fin:
        for i, line in enumerate(fin):
            items = line.strip().lower().split(' ')
            if items[0] == '1':  # valide q-a pair [label, q, a, d+]
                raw.append(items)
                if items[1] in raw_dict:
                    raw_dict[items[1]].append(items[3])
                else:
                    raw_dict[items[1]] = [items[3]]
    return raw, raw_dict


def read_alist(dataset):
    alist = []
    for line in open('{}/train.data'.format(dataset)):
        items = line.strip().lower().split(' ')
        alist.append(items[3])
    print('read_alist done ......')
    return alist


def read_alist_standalone(dataset, vocab, max_length, unk='<a>'):
    alist = []
    for line in open('{}/{}'.format(dataset, vocab)):
        items = line.strip().lower().split(' ')
        if len(items) > max_length:
            items = items[:max_length]
        else:
            items = items + [unk] * (max_length - len(items))
        alist.append('_'.join(items))
    print('read_alist done ......')
    return alist


def encode_sent(vocab, string, size):
    x = []
    words = string.split('_')
    for i in range(size):
        if i < len(words) and words[i] in vocab:
            x.append(vocab[words[i]])
        else:
            x.append(vocab['UNKNOWN'])
    return x


def load_val_batch(testList, vocab, index, batch_size, max_sequence_length_q=100, max_sequence_length_a=100):
    x_train_1 = []
    x_train_2 = []
    x_train_3 = []
    x_train_4 = []
    for i in range(0, batch_size):
        true_index = index + i
        if (true_index >= len(testList)):
            true_index = len(testList) - 1
        items = testList[true_index].split(' ')
        x_train_1.append(encode_sent(vocab, items[1], max_sequence_length_q))
        x_train_2.append(encode_sent(vocab, items[2], max_sequence_length_a))
        x_train_3.append(encode_sent(vocab, items[3], max_sequence_length_a))
        x_train_4.append(encode_sent(vocab, items[3], max_sequence_length_a))
    return np.array(x_train_1), np.array(x_train_2), np.array(x_train_3), np.array(x_train_4)


def batch_iter(data, batch_size, num_epochs=1, shuffle=False):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(math.ceil(len(data) / batch_size))
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            # start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[end_index-batch_size:end_index]
