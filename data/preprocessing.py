import json
import os
import re
import random
import json
import locale

import numpy as np
from nltk import word_tokenize


locale.setlocale(locale.LC_ALL, '')


# util functions
def dump_json(data, outpath):
    print 'Saving to', outpath
    with open(outpath, 'w') as out:
        json.dump(data, out, indent=4, separators=(',', ': '))


def load_vocab(path):
    ret = set()
    with open(path) as f:
        for line in f:
            ret.add(line.strip().decode('utf8'))
    return ret


def mkdirp(path_to_dir):
    import os
    import errno
    if not os.path.exists(path_to_dir):
        try:
            os.makedirs(path_to_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise e
            pass


def conv_sciq(out_dir='sciq_processed', in_dir='sciq'):
    r"""Convert original SciQ dataset to a pre-defined JSON format.

    Questions and answers are tokenized.

    Arguments:
        out_dir (str): output folder to store the converted files
        in_dir (str): folder where SciQ data is stored

    Returns:
        None
    """
    def normalize_distractor(wd):
        wd = wd.replace(u'\u200b', '')
        wd = wd.rstrip('?.')
        if not wd:
            return ''
        tokens = word_tokenize(wd)
        return ' '.join(tokens)

    for split in ['train', 'valid', 'test']:
        print '=' * 40, split, '=' * 40
        path = os.path.join(in_dir, split + '.json')
        data = json.load(open(path))
        results = []
        for k, d in enumerate(data):
            if k % 100 == 0:
                print '{}/{}'.format(k, len(data))
            tmp = {}
            tmp['sentence'] = ' '.join(word_tokenize(d['question']))
            tmp['answer'] = normalize_distractor(d['correct_answer'])
            tmp['distractors'] = [
                normalize_distractor(d['distractor{}'.format(i + 1)])
                for i in range(3)
            ]
            results.append(tmp)

        mkdirp(out_dir)
        output = os.path.join(out_dir, split + '.json')
        dump_json(results, output)


def conv_mcql(out_dir='mcql_processed', in_dir='mcql'):
    r"""Convert original MCQL dataset to a pre-defined JSON format.

    Questions and answers are tokenized. Split the data into train/valid/test.

    Arguments:
        outdir (str): output folder to store the converted files
        in_dir (str): folder where MCQL data is stored

    Returns:
        None
    """
    def normalize_distractor(wd):
        wd = wd.strip()
        wd = wd.replace(u'\u200b', '')
        wd = wd.replace('`', '')
        return ' '.join(word_tokenize(wd))

    results = []
    domains = ['biology', 'biology-olvl', 'chem', 'chem-olvl',
               'physics', 'physics-olvl']
    for domain in domains:
        print domain
        path = os.path.join(in_dir, 'mcqlearn_{}.json'.format(domain))
        data = json.load(open(path))
        for d in data:
            tmp = {}
            tmp['sentence'] = ' '.join(word_tokenize(d['sentence']))
            tmp['answer'] = normalize_distractor(d['answer'])
            tmp['distractors'] = [
                normalize_distractor(dis) for dis in d['distractors']
            ]
            results.append(tmp)
    print '# questions', len(results)
    random.seed(123321)
    random.shuffle(results)

    # export train/valid/test splits
    mkdirp(out_dir)
    N_VALID = 600
    N_TEST = 600
    output = os.path.join(out_dir, 'train.json')
    dump_json(results[:-(N_VALID + N_TEST)], output)

    output = os.path.join(out_dir, 'valid.json')
    dump_json(results[-(N_VALID + N_TEST):-N_TEST], output)

    output = os.path.join(out_dir, 'test.json')
    dump_json(results[-N_TEST:], output)


def export_vocab(dataset, data_dir, vocab_file='vocab.txt'):
    r"""Export the distractor vocabulary given a dataset.

    Arguments:
        dataset (str): specify dataset, choose from ['sciq', 'mcql']
        data_dir (str): folder which stores the preprocessed data
    """
    def is_valid_dis_sciq(dis):
        if not dis or dis in set(['a and b', 'b and c', 'a and c']):
            return False
        return True

    def is_valid_dis_mcqslearn(dis):
        if not dis:
            return ''
        if '?' in dis:  # parsing error for formula
            return False
        wds = dis.split()
        if len(wds) == 0 or wds[0] == 'all' or wds[0] == 'none' or wds[0] == 'both':
            return False
        return True

    if dataset == 'sciq':
        is_valid_dis = is_valid_dis_sciq
    elif dataset == 'mcql':
        is_valid_dis = is_valid_dis_mcqslearn
    else:
        raise ValueError('"dataset" can only be "sciq" or "mcql"')

    vocab = set()
    for split in ['train', 'valid', 'test']:
        path = os.path.join(data_dir, split + '.json')
        data = json.load(open(path))
        for d in data:
            for dis in d['distractors'] + [d['answer']]:
                dis = dis.lower()
                if is_valid_dis(dis):
                    vocab.add(dis.encode('utf8'))
    print '# vocab', len(vocab)
    output = os.path.join(data_dir, vocab_file)
    with open(output, 'w') as out:
        for wd in sorted(vocab, key=locale.strxfrm):
            out.write(wd + '\n')


def conv_to_ranking_data(data_dir='sciq_processed', split='test',
                         vocab_file='vocab.txt', n_neg=None):
    r"""Convert the processed dataset into data for ranking models.

    Arguments:
        data_dir (str): folder which stores the preprocessed data
        split (str): choose from ['train', 'valid', 'test']
        vocab (str): vocabulary's filename
        n_neg (int): number of negative samples. If set to 'None', the number
            of negative samples is set to the number of distractors (postive
            samples)
    """
    seed = 0
    np.random.seed(seed)

    vocab = load_vocab(os.path.join(data_dir, vocab_file))
    print '# vocab', len(vocab)
    path = os.path.join(data_dir, split + '.json')
    data = json.load(open(path))
    vocab_list = list(vocab)
    data_new = []
    BLANK = '<blank>'
    for k, d in enumerate(data):
        if k % 100 == 0:
            print '{}/{}'.format(k, len(data))
        ans = d['answer'].lower()
        if ans not in vocab:
            continue
        distractors = []
        for dis in d['distractors']:
            dis = dis.lower()
            if dis in vocab and dis != ans and dis not in distractors:
                distractors.append(dis)
        if len(distractors) == 0:
            continue
        sent = re.sub('_{3,}', BLANK, d['sentence']).lower()
        while True:
            n_samples = n_neg if n_neg is not None else len(distractors)
            neg_ids = np.random.choice(len(vocab), n_samples, replace=False)
            neg_samples = [vocab_list[id] for id in neg_ids]
            overlap = False
            for dis in distractors + [ans]:
                if dis in neg_samples:
                    overlap = True
                    break
            if not overlap:
                break
        dis_all = distractors + neg_samples + [ans]
        tmp = {}
        tmp['sentence'] = sent
        tmp['answer'] = ans
        tmp['distractors'] = distractors
        tmp['neg_samples'] = neg_samples
        data_new.append(tmp)

    # export as JSON format
    if n_neg is None:
        json_name = split + '_neg.json'
    else:
        json_name = split + '_neg_{}.json'.format(n_neg)
    mkdirp(data_dir)
    output_path = os.path.join(data_dir, json_name)
    dump_json(data_new, output_path)

