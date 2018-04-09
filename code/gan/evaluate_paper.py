import json
import numpy as np
import numpy as np
from math import log


def eval_avg_precision(y, rank, topk=10, pos_tot=None):
    pos_tot = sum(y) if pos_tot is None else pos_tot
    pos_cnt = 0
    avg_p = 0.
    for i, idx in enumerate(rank[:topk]):
        if pos_cnt >= pos_tot:
            break
        if y[idx] == 1:
            pos_cnt += 1
            avg_p += float(pos_cnt) / (i + 1)
    avg_p = avg_p / pos_tot if pos_tot > 0 else 0
    return avg_p


def eval_precision(y, rank, topk=10, pos_tot=None):
    pos_tot = sum(y) if pos_tot is None else pos_tot
    pos_cnt = 0
    for i, idx in enumerate(rank[:topk]):
        if pos_cnt >= pos_tot:
            break
        if y[idx] == 1:
            pos_cnt += 1
    prec = float(pos_cnt) / topk
    return prec


def eval_recall(y, rank, topk=10, pos_tot=None):
    pos_tot = sum(y) if pos_tot is None else pos_tot
    pos_cnt = 0
    for i, idx in enumerate(rank[:topk]):
        if pos_cnt >= pos_tot:
            break
        if y[idx] == 1:
            pos_cnt += 1
    recall = float(pos_cnt) / pos_tot if pos_tot > 0 else 0.
    return recall


def eval_ndcg(y, rank, topk=10, pos_tot=None):
    pos_tot = sum(y) if pos_tot is None else pos_tot
    idcg = 0.
    for i in range(1, pos_tot + 1):
        idcg += (2 ** 1 - 1) / log(i + 1, 2)
    dcg = 0.
    pos_cnt = 0
    for i, idx in enumerate(rank[:topk]):
        if pos_cnt >= pos_tot:
            break
        if y[idx] == 1:
            pos_cnt += 1
            dcg += (2 ** 1 - 1) / log(i + 1 + 1, 2)
    return dcg / idcg


def eval_mrr(y, rank):
    rr = 0.
    for i, idx in enumerate(rank):
        if y[idx] == 1:
            rr = 1. / (i + 1)
            break
    return rr


def recover(q):
    qid_sent = q.split('_')
    qid_sent = ' '.join([x for x in qid_sent if x != '<a>'])
    return qid_sent


def test_rank(test_path, test_json, topk=10):
    print('=' * 40, 'TEST RANKING', '=' * 40)
    print('Testing...', test_path)
    prediction = None

    # json
    json_data = json.load(open(test_json))
    json_sent2idx = {}
    for cnt, item in enumerate(json_data):
        json_sent2idx[item['sentence']] = cnt

    # load results
    data = {}
    qids = set()
    with open(test_path, 'r') as fin:
        for line in fin:
            line = line.strip().split(' ')
            label_ = int(line[1])
            q_ = line[2]
            a_ = line[3]
            dis_ = line[4]
            score_ = float(line[5])

            q_sent_ = recover(q_)
            dis_sent_ = recover(dis_)
            label_real_ = int(dis_sent_ in json_data[json_sent2idx[q_sent_]]['distractors'])
            label_ = label_real_

            if q_ in qids:
                data['y_{}'.format(q_)].append(label_)
                data['y_pred_{}'.format(q_)].append(score_)
            else:
                qids.add(q_)
                data['y_{}'.format(q_)] = [label_]
                data['y_pred_{}'.format(q_)] = [score_]

    # evaluatin
    n_query = len(qids)
    map_k = 0.
    recall_k = 0.
    p1_tot = 0.
    p3_tot = 0.
    ndcg_k = 0.
    mrr = 0.
    for cnt, qid in enumerate(qids):
        if cnt % 100 == 0:
            print('Testing {}/{}'.format(cnt, n_query))
        y = np.array(data['y_{}'.format(qid)])

        qid_sent = recover(qid)
        if qid_sent not in json_sent2idx:
            print(qid_sent)
            continue
        d = json_data[json_sent2idx[qid_sent]]
        y_pred = data['y_pred_{}'.format(qid)]
        rank = np.argsort(y_pred)[::-1]
        rank = rank.tolist()

        pos_tot = len(d['distractors'])

        if pos_tot > 0:
            avg_p = eval_avg_precision(y, rank, topk=topk, pos_tot=pos_tot)
            recall = eval_recall(y, rank, topk=topk, pos_tot=pos_tot)
            p1 = eval_precision(y, rank, topk=1, pos_tot=pos_tot)
            p3 = eval_precision(y, rank, topk=3, pos_tot=pos_tot)
            ndcg = eval_ndcg(y, rank, topk=topk, pos_tot=pos_tot)
            rr = eval_mrr(y, rank)

            map_k += avg_p
            recall_k += recall
            ndcg_k += ndcg
            p1_tot += p1
            p3_tot += p3
            mrr += rr

    map_k /= n_query
    recall_k /= n_query
    ndcg_k /= n_query
    p1_tot /= n_query
    p3_tot /= n_query
    mrr /= n_query

    result = {
        'MAP@{}'.format(topk): map_k,
        'Recall@{}'.format(topk): recall_k,
        'Precision@1': p1_tot,
        'Precision@3': p3_tot,
        'NDCG@{}'.format(topk): ndcg_k,
        'MRR': mrr,
        'Prediction': prediction,
    }
    print(result)
    return result

if __name__ == '__main__':
    test_path = './log/distractorQA_mcq_results.txt' 
    test_json = './distractorQA/mcq_neg/test_neg_2nd_2500.json'
    test_rank(test_path, test_json)
