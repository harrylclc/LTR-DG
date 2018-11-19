import json
import random
random.seed(0)


def recover(s, delimiter='_', unk='<a>'):
    s = s.split(delimiter)
    s = [x for x in s if x != unk]
    return s


def padding(s, max_length=100, unk='<a>', delimiter='_'):
    if isinstance(s, str):
        s = s.split()
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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str, required=True, help='path to json')
    parser.add_argument('--output', type=str, required=True, help='path to output')
    parser.add_argument('--vocab_path', type=str, default='combine.vocab', help='path to vocabulary')
    parser.add_argument('--max_length', type=int, default=100, help='max length of questions')
    parser.add_argument('--max_length_dis', type=int, default=20, help='max length of distractors')
    parser.add_argument('--delimiter', type=str, default='_')
    parser.add_argument('--unk', type=str, default='<a>')
    parser.add_argument('--train', type=int, default=1, help='whether is train or val/test')
    args = parser.parse_args()

    # Get data
    data = json.load(open(args.json))
    vocab = load_vocab(args.vocab_path)

    # Get max length just for a reference
    max_dis_len = -1
    max_sent_len = -1
    for d in data:
        dis_all = d['distractors'] + d['neg_samples'] + [d['answer']]
        d_lens = [len(dis.split()) for dis in dis_all]
        max_dis_len = max(max_dis_len, max(d_lens))
        max_sent_len = max(max_sent_len, len(d['sentence'].split()))
    print('max sentence length', max_sent_len)
    print('max distractor length', max_dis_len)

    # Main logic
    with open(args.output, 'w') as out:
        for d in data:
            sent = padding(d['sentence'].lower(), args.max_length,
                           args.unk, args.delimiter).encode('utf8')
            answer = padding(d['answer'].lower(), args.max_length_dis,
                             args.unk, args.delimiter).encode('utf8')
            for key in ['distractors', 'neg_samples']:
                for dis in d[key]:
                    label = 1 if key == 'distractors' else 0
                    if (args.train == 1 and label == 1) or args.train != 1:
                        dis = padding(dis.lower(), max_dis_len, args.unk, args.delimiter).encode('utf8')
                        out.write('{} {} {} {}\n'.format(label, sent, answer, dis))
