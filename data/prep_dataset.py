from preprocessing import *


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', choices=['sciq', 'mcql'], default='sciq')
    parser.add_argument('-data_dir', default='sciq')
    parser.add_argument('-out_dir', default='sciq_processed')
    parser.add_argument('-vocab', default='vocab.txt')
    parser.add_argument('-n_neg', type=int)
    args = parser.parse_args()

    dataset = args.dataset
    raw_dir = args.data_dir
    prep_dir = args.out_dir

    if dataset == 'sciq':
        conv_sciq(in_dir=raw_dir, out_dir=prep_dir)
    elif dataset == 'mcql':
        conv_mcql(in_dir=raw_dir, out_dir=prep_dir)
    export_vocab(dataset, prep_dir, vocab_file=args.vocab)
    for split in ['train', 'valid', 'test']:
        conv_to_ranking_data(prep_dir, split, vocab_file=args.vocab,
                             n_neg=args.n_neg)

