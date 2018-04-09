# coding=utf-8
import tensorflow as tf
from argparse import Namespace
from DecompAtt import DecompAtt


class Generator(DecompAtt):
    def __init__(self, sequence_length_q, sequence_length_a, batch_size, vocab_size,
                 embedding_size, hidden_size, dropout_keep_prob=1.0,
                 l2_reg_lambda=0.0, learning_rate=1e-2, sampled_temperature=1,
                 update_embeddings=True, is_training=True, padding_id=0):

        config = Namespace()
        config.sequence_length_q = sequence_length_q
        config.sequence_length_a = sequence_length_a
        config.batch_size = batch_size
        config.vocab_size = vocab_size
        config.padding_id = padding_id
        config.embedding_size = embedding_size
        config.hidden_size = hidden_size
        config.dropout_keep_prob = dropout_keep_prob
        config.l2_reg_lambda = l2_reg_lambda
        config.learning_rate = learning_rate
        self.model_type = 'Gen'
        self.sampled_temperature = sampled_temperature
        super(Generator, self).__init__(config, update_embeddings, is_training, self.model_type)

        self.reward = tf.placeholder(tf.float32, shape=[None], name='reward')
        self.neg_index = tf.placeholder(tf.int32, shape=[None], name='neg_index')

        self.gan_score = self.logits13
        self.batch_scores = tf.nn.softmax(self.logits13) + 1e-7
        self.prob = tf.gather(self.batch_scores, self.neg_index)
        self.gan_loss = -tf.reduce_mean(tf.log(self.prob) * self.reward) + self.l2_loss

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        grads_and_vars = optimizer.compute_gradients(self.gan_loss)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads_and_vars if grad is not None]
        self.gan_updates = optimizer.apply_gradients(capped_gvs, global_step=self.global_step)

        # classification, used for pre-training
        self.labels = tf.Variable([1] * batch_size + [0] * batch_size, dtype=tf.float32, trainable=False)
        self.cls_loss = self.l2_loss + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.logits))

        self.correct12 = tf.greater(self.score12, 0.5)
        self.correct13 = tf.less(self.score13, 0.5)
        self.accuracy = (tf.reduce_mean(tf.cast(self.correct12, "float")) + tf.reduce_mean(tf.cast(self.correct13, "float"))) / 2

        self.cls_step = tf.Variable(0, name="cls_step", trainable=False)
        cls_optimizer = tf.train.AdamOptimizer(self.learning_rate)
        cls_grads_and_vars = cls_optimizer.compute_gradients(self.cls_loss)
        cls_capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in cls_grads_and_vars if grad is not None]
        self.cls_updates = cls_optimizer.apply_gradients(cls_capped_gvs, global_step=self.cls_step)
