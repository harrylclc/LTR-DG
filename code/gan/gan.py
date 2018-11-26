# coding=utf-8

import sys
import os
import numpy as np
import pickle
import time
import math
import datetime
import tensorflow as tf
from functools import wraps

import Discriminator
import Generator

from data_helpers import encode_sent
import data_helpers as data_helpers

# import dataHelper
# Data
tf.flags.DEFINE_string("dataset", "semevalQA", "dataset path")
tf.flags.DEFINE_string("prefix", "semevalQA", "prefix")
# Model Hyperparameters
tf.flags.DEFINE_integer("max_sequence_length_q", 100, "Max sequence length fo sentence (default: 100)")
tf.flags.DEFINE_integer("max_sequence_length_a", 100, "Max sequence length fo sentence (default: 100)")
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 300)")
tf.flags.DEFINE_integer("hidden_size", 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 1e-6, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_float("learning_rate", 0.0001, "learning_rate (default: 0.1)")
tf.flags.DEFINE_string("padding", "<a>", "dataset path")

# Training parameters
tf.flags.DEFINE_string("pretrained_embeddings_path", "", "path to pretrained_embeddings")
tf.flags.DEFINE_string("pretrained_model_path", "", "path to pretrained_model")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 50)")
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("pools_size", 64, "The positive sample set, which is bigger than 500")
tf.flags.DEFINE_integer("gan_k", 16, "the number of samples of gan")
tf.flags.DEFINE_integer("g_epochs_num", 1, " the num_epochs of generator per epoch")
tf.flags.DEFINE_integer("d_epochs_num", 1, " the num_epochs of discriminator per epoch")
tf.flags.DEFINE_integer("sampled_temperature", 5, " the temperature of sampling")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)
assert(FLAGS.batch_size == FLAGS.pools_size)

print(("\nParameters:"))
for attr, value in sorted(FLAGS.__flags.items()):
        print(("{}={}".format(attr.upper(), value)))
print((""))

timeStamp = time.strftime("%Y%m%d%H%M%S", time.localtime(int(time.time())))

print(("Loading data..."))
vocab, embd = data_helpers.build_vocab(FLAGS.dataset, FLAGS.pretrained_embeddings_path)
if len(FLAGS.pretrained_embeddings_path) > 0:
    assert(embd.shape[1] == FLAGS.embedding_dim)
    with open('{}/embd.pkl'.format(FLAGS.dataset), 'wb') as fout:
        pickle.dump(embd, fout)
with open('{}/vocab.pkl'.format(FLAGS.dataset), 'wb') as fout:
    pickle.dump(vocab, fout)
alist = data_helpers.read_alist_standalone(FLAGS.dataset, "vocab.txt", FLAGS.max_sequence_length_a, FLAGS.padding)
raw, raw_dict = data_helpers.read_raw(FLAGS.dataset)
devList = data_helpers.loadTestSet(FLAGS.dataset, "valid.data")
testList = data_helpers.loadTestSet(FLAGS.dataset, "test.data")
testallList = data_helpers.loadTestSet(FLAGS.dataset, "test.data")  # testall

print("Load done...")
if not os.path.exists('./log/'):
    os.mkdir('./log/')
log_precision = 'log/{}.test.gan_precision.{}.log'.format(FLAGS.prefix, timeStamp)
log_loss = 'log/{}.test.gan_loss.{}.log'.format(FLAGS.prefix, timeStamp)


def log_time_delta(func):
    @wraps(func)
    def _deco(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()
        delta = end - start
        print("%s runed %.2f seconds" % (func.__name__, delta))
        return ret
    return _deco


def generate_gan(sess, model, negative_size=FLAGS.gan_k):
    '''used for generate negative samples for the Discriminator'''
    samples = []
    for _index, pair in enumerate(raw):
        if _index % 5000 == 0:
            print("have sampled %d pairs" % _index)
        q = pair[1]
        a = pair[2]
        distractor = pair[3]

        neg_alist_index = [i for i in range(len(alist))]
        sampled_index = np.random.choice(neg_alist_index, size=[FLAGS.pools_size], replace=False)
        pools = np.array(alist)[sampled_index]  # it's possible that true positive samples are selected
        # TODO: remove true positives

        # [q, a, distractor, negative sample]
        canditates = data_helpers.loadCandidateSamples(q, a, distractor, pools, vocab,
                                                       FLAGS.max_sequence_length_q, FLAGS.max_sequence_length_a)
        predicteds = []
        for batch in data_helpers.batch_iter(canditates, batch_size=FLAGS.batch_size):
            feed_dict = {
                model.input_x_1: np.array(batch[:, 0].tolist()),
                model.input_x_2: np.array(batch[:, 1].tolist()),
                model.input_x_3: np.array(batch[:, 2].tolist()),
                model.input_x_4: np.array(batch[:, 3].tolist())
            }
            predicted = sess.run(model.gan_score, feed_dict)
            predicteds.extend(predicted)

        predicteds = np.array(predicteds) * FLAGS.sampled_temperature
        predicteds -= np.max(predicteds)
        exp_rating = np.exp(predicteds)
        prob = exp_rating / np.sum(exp_rating)
        prob = np.nan_to_num(prob) + 1e-7
        prob = prob / np.sum(prob)
        neg_samples = np.random.choice(pools, size=negative_size, p=prob, replace=False)
        for neg in neg_samples:
            samples.append((encode_sent(vocab, q, FLAGS.max_sequence_length_q),
                            encode_sent(vocab, a, FLAGS.max_sequence_length_a),
                            encode_sent(vocab, distractor, FLAGS.max_sequence_length_a),
                            encode_sent(vocab, neg, FLAGS.max_sequence_length_a)))
    return samples


def get_metrics(target_list, batch_scores, topk=10):
    '''
    Args:
        target_list: [1, 1, 0, 0, ...]. Could be all zeros.
        batch_scores: [0.9, 0.7, 0.2, ...]. Predicted relevance.
    '''
    length = min(len(target_list), len(batch_scores))
    if length == 0:
        return [0, 0, 0, 0]
    target_list = target_list[:length]
    batch_scores = batch_scores[:length]
    target_list = np.array(target_list)
    predict_list = np.argsort(batch_scores)[::-1]
    predict_list = target_list[predict_list]
    # RR and AP for MRR and MAP
    RR = 0.0
    avg_prec = 0.0
    precisions = []
    num_correct = 0.0
    prec1 = 0.0
    num_correct_total = 0.0
    for i in range(len(predict_list)):
        if predict_list[i] == 1:
            num_correct_total += 1
    for i in range(min(topk, len(predict_list))):
        if i == 0:
            if predict_list[i] == 1:
                prec1 = 1.0
        if predict_list[i] == 1:
            if RR == 0:
                RR += 1.0 / (i + 1)
            num_correct += 1
            precisions.append(num_correct / (i + 1))
    if len(precisions) > 0:
        avg_prec = sum(precisions) / len(precisions)
    if num_correct_total == 0.0:
        recall = 0.0
    else:
        recall = num_correct / num_correct_total
    return [RR, avg_prec, prec1, recall]


@log_time_delta
def dev_step(sess, model, devList, saveresults=False):
    # grouped by q
    testList_dict = {}
    for i in range(len(devList)):
        item = devList[i].split(' ')
        label, q = item[0], item[1]
        if q in testList_dict:
            testList_dict[q][0].append(int(label))
            testList_dict[q][1].append(devList[i])
        else:
            testList_dict[q] = [[int(label)], [devList[i]]]

    # save results
    saveresults_path = 'log/{}_results.txt'.format(FLAGS.prefix)
    if saveresults:
        with open(saveresults_path, 'w') as fout:
            fout.write('')

    # evaluation
    metrics_all = []
    for q, item in testList_dict.items():
        target_list = item[0]
        if np.sum(target_list) == 0:
            continue

        testList_sub = item[1]
        # batch_scores
        batch_scores = []
        for i in range(int(math.ceil(len(testList_sub) / FLAGS.batch_size))):
            if (i + 1) * FLAGS.batch_size > len(testList_sub):
                batch_size_real = len(testList_sub) - i * FLAGS.batch_size
            else:
                batch_size_real = FLAGS.batch_size
            x_test_1, x_test_2, x_test_3, x_test_4 = data_helpers.load_val_batch(testList_sub, vocab,
                                                                                 i*FLAGS.batch_size,
                                                                                 batch_size_real,
                                                                                 FLAGS.max_sequence_length_q,
                                                                                 FLAGS.max_sequence_length_a)
            feed_dict = {
                model.input_x_1: x_test_1,
                model.input_x_2: x_test_2,
                model.input_x_3: x_test_3,
                model.input_x_4: x_test_4  # x_test_3 equals x_test_4 for the test case
            }
            predicted = sess.run(model.score12, feed_dict)
            batch_scores.extend(predicted)
        # save results
        if saveresults:
            for j in range(len(batch_scores)):
                with open(saveresults_path, 'a') as fout:
                    fout.write('{} {} {}\n'.format(target_list[j],
                        testList_sub[j], batch_scores[j]))

        # MRR@10, MAP@10, Precision@1, Recall@10, etc.
        metrics = get_metrics(target_list, batch_scores, 10)
        metrics_all.append(metrics)
    metrics_all = np.array(metrics_all)
    # metrics_all = np.mean(metrics_all, axis=0)
    metrics_all = np.sum(metrics_all, axis=0) / len(testList_dict)
    return metrics_all.tolist()


@log_time_delta
def evaluation(sess, model, log, saver, num_epochs=0, split='dev',
        savemodel=True, justsave=False, saveresults=False):
    if isinstance(model, Discriminator.Discriminator):
        model_type = "Dis"
    else:
        model_type = "Gen"
    print(model_type)

    metrics_current = [0, 0, 0, 0]
    metrics_current = [str(x) for x in metrics_current]
    if justsave:
        filename = "model/{}_{}_{}_{}.model".format(FLAGS.prefix, model.model_type, num_epochs, '_'.join(metrics_current))
        saver.save(sess, filename)
        return

    assert split in ['dev', 'test', 'testall']
    if split == 'dev':
        metrics_current = dev_step(sess, model, devList, saveresults)
    elif split == 'test':
        metrics_current = dev_step(sess, model, testList, saveresults)
    elif split == 'testall':
        metrics_current = dev_step(sess, model, testallList, saveresults)

    line = "test: %d epoch: metric %s" % (num_epochs, metrics_current)
    print(line)
    log.write(line + "\n")
    log.flush()
    metrics_current = [str(x) for x in metrics_current]

    if savemodel:
        filename = "model/{}_{}_{}_{}.model".format(FLAGS.prefix, model.model_type, num_epochs, '_'.join(metrics_current))
        saver.save(sess, filename)


def main():
    with tf.Graph().as_default():
        with tf.device("/gpu:1"):
            # embeddings
            param = None
            if len(FLAGS.pretrained_embeddings_path) > 0:
                print('loading pretrained embeddings...')
                param = embd
            else:
                print('using randomized embeddings...')
                param = np.random.uniform(-0.05, 0.05, (len(vocab), FLAGS.embedding_dim))

            # models
            with tf.variable_scope('Dis'):
                discriminator = Discriminator.Discriminator(
                    sequence_length_q=FLAGS.max_sequence_length_q,
                    sequence_length_a=FLAGS.max_sequence_length_a,
                    batch_size=FLAGS.batch_size,
                    vocab_size=len(vocab),
                    embedding_size=FLAGS.embedding_dim,
                    hidden_size=FLAGS.hidden_size,
                    l2_reg_lambda=FLAGS.l2_reg_lambda,
                    learning_rate=FLAGS.learning_rate,
                    dropout_keep_prob=FLAGS.dropout_keep_prob,
                    padding_id=vocab[FLAGS.padding]
                )

            with tf.variable_scope('Gen'):
                generator = Generator.Generator(
                    sequence_length_q=FLAGS.max_sequence_length_q,
                    sequence_length_a=FLAGS.max_sequence_length_a,
                    batch_size=FLAGS.batch_size,
                    vocab_size=len(vocab),
                    embedding_size=FLAGS.embedding_dim,
                    hidden_size=FLAGS.hidden_size,
                    l2_reg_lambda=FLAGS.l2_reg_lambda,
                    sampled_temperature=FLAGS.sampled_temperature,
                    learning_rate=FLAGS.learning_rate,
                    dropout_keep_prob=FLAGS.dropout_keep_prob,
                    padding_id=vocab[FLAGS.padding]
                )

            session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                          log_device_placement=FLAGS.log_device_placement)
            sess = tf.Session(config=session_conf)
            with sess.as_default(), open(log_precision, "w") as log, open(log_loss, "w") as loss_log:
                # initialze or restore
                if len(FLAGS.pretrained_model_path) == 0:
                    print('initializing model...')
                    sess.run(tf.global_variables_initializer())
                    # pretrained embeddings or randomized embeddings
                    sess.run(discriminator.embedding_init,
                             feed_dict={discriminator.embedding_placeholder: param})
                    sess.run(generator.embedding_init,
                             feed_dict={generator.embedding_placeholder: param})
                else:
                    print('loading pretrained model...')
                    var_list = tf.global_variables()
                    var_list = [x for x in var_list if not x.name.startswith('Dis/output/Variable')]
                    var_list = [x for x in var_list if not x.name.startswith('Gen/Variable')]
                    restore_op, feed_dict = tf.contrib.framework.assign_from_checkpoint(
                        tf.train.latest_checkpoint(FLAGS.pretrained_model_path),
                        var_list,
                        True
                    )
                    sess.run(restore_op, feed_dict)

                # initial evaluation
                saver = tf.train.Saver(max_to_keep=None)
                # evaluation(sess, discriminator, log, saver, 0, 'dev', False)
                # evaluation(sess, generator, log, saver, 0, 'dev', False)

                baseline = 0.05
                for i in range(FLAGS.num_epochs):
                    # discriminator
                    if i > 0:
                        samples = generate_gan(sess, generator, FLAGS.gan_k)
                        for _index, batch in enumerate(data_helpers.batch_iter(samples,
                                                                               num_epochs=FLAGS.d_epochs_num,
                                                                               batch_size=FLAGS.batch_size,
                                                                               shuffle=True)):
                            feed_dict = {  # [q, a, distractor, negative sample]
                                discriminator.input_x_1: np.array(batch[:, 0].tolist()),
                                discriminator.input_x_2: np.array(batch[:, 1].tolist()),
                                discriminator.input_x_3: np.array(batch[:, 2].tolist()),
                                discriminator.input_x_4: np.array(batch[:, 3].tolist())
                            }
                            _, step, current_loss, accuracy, positive, negative = sess.run(
                                [discriminator.train_op, discriminator.global_step,
                                    discriminator.loss, discriminator.accuracy,
                                    discriminator.positive, discriminator.negative],
                                feed_dict
                            )

                            line = ("%s: Dis step %d, loss %f with acc %f, positive %f negative %f" % (
                                datetime.datetime.now().isoformat(),
                                step, current_loss, accuracy, positive, negative)
                            )
                            if _index % 100 == 0:
                                print(line)
                            loss_log.write(line+"\n")
                            loss_log.flush()
                        evaluation(sess, discriminator, log, saver, i,
                        'dev', True, False)

                    # generator
                    baseline_avg = []
                    for g_epoch in range(FLAGS.g_epochs_num):
                        for _index, pair in enumerate(raw):
                            q = pair[1]
                            a = pair[2]
                            distractor = pair[3]

                            # it's possible that true positive samples are selected
                            neg_alist_index = [j for j in range(len(alist))]
                            pos_num = min(4, len(raw_dict[q]))
                            sampled_index = np.random.choice(neg_alist_index,
                                                             size=FLAGS.pools_size - pos_num,
                                                             replace=False)
                            sampled_index = list(sampled_index)
                            pools = np.array(alist)[sampled_index]
                            # add the positive index
                            positive_index = [j for j in range(len(raw_dict[q]))]
                            positive_index = np.random.choice(positive_index, pos_num, replace=False).tolist()
                            pools = np.concatenate((pools, np.array(raw_dict[q])[positive_index]))

                            samples = data_helpers.loadCandidateSamples(q, a, distractor, pools, vocab,
                                                                        FLAGS.max_sequence_length_q,
                                                                        FLAGS.max_sequence_length_a)
                            predicteds = []
                            for batch in data_helpers.batch_iter(samples, batch_size=FLAGS.batch_size):
                                feed_dict = {
                                    generator.input_x_1: np.array(batch[:, 0].tolist()),
                                    generator.input_x_2: np.array(batch[:, 1].tolist()),
                                    generator.input_x_3: np.array(batch[:, 2].tolist()),
                                    generator.input_x_4: np.array(batch[:, 3].tolist())
                                }
                                predicted = sess.run(generator.gan_score, feed_dict)
                                predicteds.extend(predicted)

                            # generate FLAGS.gan_k negative samples
                            predicteds = np.array(predicteds) * FLAGS.sampled_temperature
                            predicteds -= np.max(predicteds)
                            exp_rating = np.exp(predicteds)
                            prob = exp_rating / np.sum(exp_rating)
                            prob = np.nan_to_num(prob) + 1e-7
                            prob = prob / np.sum(prob)
                            neg_index = np.random.choice(np.arange(len(pools)), size=FLAGS.gan_k, p=prob, replace=False)

                            subsamples = np.array(data_helpers.loadCandidateSamples(q, a, distractor, pools[neg_index], vocab,
                                                                                    FLAGS.max_sequence_length_q,
                                                                                    FLAGS.max_sequence_length_a))
                            feed_dict = {
                                discriminator.input_x_1: np.array(subsamples[:, 0].tolist()),
                                discriminator.input_x_2: np.array(subsamples[:, 1].tolist()),
                                discriminator.input_x_3: np.array(subsamples[:, 2].tolist()),
                                discriminator.input_x_4: np.array(subsamples[:, 3].tolist())
                            }
                            reward, l2_loss_d = sess.run([discriminator.reward, discriminator.l2_loss], feed_dict)
                            baseline_avg.append(np.mean(reward))
                            reward = reward - baseline

                            samples = np.array(samples)
                            feed_dict = {
                                generator.input_x_1: np.array(samples[:, 0].tolist()),
                                generator.input_x_2: np.array(samples[:, 1].tolist()),
                                generator.input_x_3: np.array(samples[:, 2].tolist()),
                                generator.input_x_4: np.array(samples[:, 3].tolist()),
                                generator.neg_index: neg_index,
                                generator.reward: reward
                            }
                            # should be softmax over all, but too computationally expensive
                            _, step, current_loss, positive, negative, score12, score13, l2_loss_g = sess.run(
                                [generator.gan_updates, generator.global_step,
                                    generator.gan_loss, generator.positive, generator.negative,
                                    generator.score12, generator.score13, generator.l2_loss],
                                feed_dict
                            )

                            line = ("%s: Gen step %d, loss %f l2 %f,%f positive %f negative %f, sample prob [%s, %f], reward [%f, %f]" % (
                                datetime.datetime.now().isoformat(), step,
                                current_loss, l2_loss_g, l2_loss_d, positive, negative, np.min(prob), np.max(prob), np.min(reward), np.max(reward)
                            ))
                            if _index % 100 == 0:
                                print(line)
                            loss_log.write(line+"\n")
                            loss_log.flush()

                        evaluation(sess, generator, log, saver, i *
                                FLAGS.g_epochs_num + g_epoch, 'dev', True,
                                False)
                        log.flush()
                    baseline = np.mean(baseline_avg)

                # final evaluation
                evaluation(sess, discriminator, log, saver, -1, 'test', False,
                        False, True)
                evaluation(sess, generator, log, saver, -1, 'test', False,
                        False, True)

if __name__ == '__main__':
    main()
