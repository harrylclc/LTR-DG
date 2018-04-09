import tensorflow as tf


class DecompAtt(object):
    """ Implements a Decomposable Attention model for NLI.
        http://arxiv.org/pdf/1606.01933v1.pdf """
    def __init__(self, config, update_embeddings=True, is_training=True, model_type='base'):
        self.config = config
        self.batch_size = config.batch_size
        self.hidden_size = config.hidden_size
        self.padding_id = config.padding_id
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        self.sequence_length_q = config.sequence_length_q
        self.sequence_length_a = config.sequence_length_a
        self.dropout_keep_prob = config.dropout_keep_prob
        self.l2_reg_lambda = config.l2_reg_lambda
        self.learning_rate = config.learning_rate
        self.update_embeddings = update_embeddings
        self.is_training = is_training
        self.model_type = model_type
        assert self.hidden_size == self.embedding_size, 'size should be the same'

        # regularizer
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.l2_reg_lambda)

        # placeholders for inputs [q, a, distractor, negative sample]
        self.input_x_1 = tf.placeholder(tf.int32, [None, self.sequence_length_q], name="input_x_1")
        self.input_x_2 = tf.placeholder(tf.int32, [None, self.sequence_length_a], name="input_x_2")
        self.input_x_3 = tf.placeholder(tf.int32, [None, self.sequence_length_a], name="input_x_3")
        self.input_x_4 = tf.placeholder(tf.int32, [None, self.sequence_length_a], name="input_x_4")

        self.premise = tf.concat([self.input_x_1, self.input_x_1], 0)
        self.answer = tf.concat([self.input_x_2, self.input_x_2], 0)
        self.hypothesis = tf.concat([self.input_x_3, self.input_x_4], 0)

        self.embedding = tf.get_variable('embedding', [self.vocab_size, self.embedding_size], dtype=tf.float32,
                                         regularizer=self.regularizer,
                                         trainable=update_embeddings)
        self.embedding_placeholder = tf.placeholder(tf.float32, [self.vocab_size, self.embedding_size])
        self.embedding_init = self.embedding.assign(self.embedding_placeholder)

        # mask embedding
        raw_mask_array = [[1.]] * self.padding_id + [[0.]] + [[1.]] * (self.vocab_size - self.padding_id - 1)
        self.mask_padding_lookup_table = tf.get_variable('mask_padding_lookup_table',
                                                         initializer=raw_mask_array,
                                                         dtype=tf.float32,
                                                         trainable=False)

        # (batch, sequence_length_p, hidden_size)
        premise_inputs = tf.nn.embedding_lookup(self.embedding, self.premise)
        premise_inputs_mask = tf.nn.embedding_lookup(self.mask_padding_lookup_table, self.premise)
        self.premise_inputs = tf.multiply(premise_inputs, premise_inputs_mask)

        # (batch, sequence_length_h, hidden_size)
        answer_inputs = tf.nn.embedding_lookup(self.embedding, self.answer)
        answer_inputs_mask = tf.nn.embedding_lookup(self.mask_padding_lookup_table, self.answer)
        self.answer_inputs = tf.multiply(answer_inputs, answer_inputs_mask)
        self.answer_inputs_sum = tf.reduce_sum(self.answer_inputs, 1)

        # (batch, sequence_length_h, hidden_size)
        hypothesis_inputs = tf.nn.embedding_lookup(self.embedding, self.hypothesis)
        hypothesis_inputs_mask = tf.nn.embedding_lookup(self.mask_padding_lookup_table, self.hypothesis)
        self.hypothesis_inputs = tf.multiply(hypothesis_inputs, hypothesis_inputs_mask)
        self.hypothesis_inputs_sum = tf.reduce_sum(self.hypothesis_inputs, 1)

        # run feed-forward networks
        with tf.variable_scope("F"):
            self.premise_F = self.feedforward_3d(self.premise_inputs)
            self.premise_F = tf.layers.batch_normalization(self.premise_F)
        with tf.variable_scope("F", reuse=True):
            self.hypothesis_F = self.feedforward_3d(self.hypothesis_inputs)
            self.hypothesis_F = tf.layers.batch_normalization(self.hypothesis_F)

        # normalize along sequence_length_h
        # (batch, sequence_length_q, sequence_length_a)
        self.dot1 = tf.matmul(self.premise_F, self.hypothesis_F, transpose_b=True)
        self.hypothesis_softmax = tf.reshape(self.dot1, [-1, self.sequence_length_a])
        self.hypothesis_softmax = tf.reshape(tf.nn.softmax(self.hypothesis_softmax),
                                             [-1, self.sequence_length_q, self.sequence_length_a])

        # normalize along sequence_length_p
        # (batch, sequence_length_a, sequence_length_q)
        self.dot2 = tf.transpose(self.dot1, [0, 2, 1])
        self.premise_softmax = tf.reshape(self.dot2, [-1, self.sequence_length_q])
        self.premise_softmax = tf.reshape(tf.nn.softmax(self.premise_softmax),
                                          [-1, self.sequence_length_a, self.sequence_length_q])

        # (batch, sequence_length_p, hidden_size)
        self.betas = tf.matmul(self.hypothesis_softmax, self.hypothesis_inputs)
        # (batch, sequence_length_h, hidden_size)
        self.alphas = tf.matmul(self.premise_softmax, self.premise_inputs)

        self.v1 = tf.concat([self.premise_inputs, self.betas], 2)
        self.v2 = tf.concat([self.hypothesis_inputs, self.alphas], 2)

        # run feed-forward networks
        with tf.variable_scope("G"):
            self.v1 = self.feedforward_3d(self.v1, self.hidden_size)
            self.v1 = tf.layers.batch_normalization(self.v1)
        with tf.variable_scope("G", reuse=True):
            self.v2 = self.feedforward_3d(self.v2, self.hidden_size)
            self.v2 = tf.layers.batch_normalization(self.v2)

        # aggregate
        self.v1_sum = tf.reduce_sum(self.v1, 1)  # (batch, self.hidden_size)
        self.v2_sum = tf.reduce_sum(self.v2, 1)  # (batch, self.hidden_size)
        self.v = tf.concat([self.v1_sum, self.v2_sum], 1)
        with tf.variable_scope("final_representation"):
            self.final_representation = self.feedforward_2d(self.v)
            self.final_representation = tf.layers.batch_normalization(self.final_representation)
        if self.dropout_keep_prob < 1.0 and self.is_training:
            self.final_representation = tf.nn.dropout(self.final_representation, self.dropout_keep_prob)

        # outputs to generate distribution
        output_w = tf.get_variable("output_w", [self.hidden_size*2, 1], regularizer=self.regularizer)
        output_b = tf.get_variable("output_b", [1], regularizer=self.regularizer, initializer=tf.zeros_initializer())
        self.logits1 = tf.matmul(self.final_representation, output_w) + output_b

        self.logits2 = self.cosine(self.answer_inputs_sum, self.hypothesis_inputs_sum)
        self.logits2 = tf.expand_dims(self.logits2, 1)

        combine_w = tf.get_variable("combine_w", [2, 1], regularizer=self.regularizer)
        combine_b = tf.get_variable("combine_b", [1], regularizer=self.regularizer, initializer=tf.zeros_initializer())
        self.logits = tf.matmul(tf.concat([self.logits1, self.logits2], 1), combine_w) + combine_b
        self.logits = tf.squeeze(self.logits)
        self.score = tf.sigmoid(self.logits)

        # scores
        self.logits12, self.logits13 = tf.split(self.logits, 2, 0)
        self.score12, self.score13 = tf.split(self.score, 2, 0)
        self.positive = tf.reduce_mean(self.score12)
        self.negative = tf.reduce_mean(self.score13)

        # l2
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=self.model_type)
        self.l2_loss = tf.reduce_sum(reg_variables)

    def linear_3d(self, input3D):
        length = input3D.get_shape()[1].value
        dimensions = input3D.get_shape()[2].value

        inputs = tf.reshape(input3D, [-1, length, 1, dimensions])
        W_proj = tf.get_variable("W_proj", [1, 1, dimensions, self.hidden_size], regularizer=self.regularizer)
        b_proj = tf.get_variable("b_proj", [self.hidden_size], regularizer=self.regularizer, initializer=tf.zeros_initializer())

        projection = tf.nn.conv2d(inputs, W_proj, [1, 1, 1, 1], "SAME")
        projection = tf.nn.relu(tf.nn.bias_add(projection, b_proj))
        return tf.reshape(projection, [-1, length, self.hidden_size])

    def feedforward_3d(self, input3D, dim=None):
        "implement feedforward for 3D tensor by two 2D convolutions with 2 different features"
        length = input3D.get_shape()[1].value
        dimensions = input3D.get_shape()[2].value
        if dim is None:
            dim = dimensions

        inputs = tf.reshape(input3D, [-1, length, 1, dimensions])
        k1 = tf.get_variable("W1", [1, 1, dimensions, dim], regularizer=self.regularizer)
        k2 = tf.get_variable("W2", [1, 1, dim, dim], regularizer=self.regularizer)
        b1 = tf.get_variable("b1", [dim], regularizer=self.regularizer, initializer=tf.zeros_initializer())
        b2 = tf.get_variable("b2", [dim], regularizer=self.regularizer, initializer=tf.zeros_initializer())

        features = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(inputs, k1, [1, 1, 1, 1], "SAME"), b1))
        features = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(features, k2, [1, 1, 1, 1], "SAME"), b2))
        return tf.reshape(features, [-1, length, dim])

    def feedforward_2d(self, input):
        hidden_dim = input.get_shape()[1].value

        hidden1_w = tf.get_variable("hidden1_w", [hidden_dim, hidden_dim], regularizer=self.regularizer)
        hidden1_b = tf.get_variable("hidden1_b", [hidden_dim], regularizer=self.regularizer, initializer=tf.zeros_initializer())

        hidden2_w = tf.get_variable("hidden2_w", [hidden_dim, hidden_dim], regularizer=self.regularizer)
        hidden2_b = tf.get_variable("hidden2_b", [hidden_dim], regularizer=self.regularizer, initializer=tf.zeros_initializer())

        hidden1 = tf.nn.relu(tf.matmul(input, hidden1_w) + hidden1_b)
        gate_output = tf.nn.relu(tf.matmul(hidden1, hidden2_w) + hidden2_b)
        return gate_output

    def cosine(self, q, a):
        pooled_len_1 = tf.sqrt(tf.reduce_sum(tf.multiply(q, q), 1))
        pooled_len_2 = tf.sqrt(tf.reduce_sum(tf.multiply(a, a), 1))

        pooled_mul_12 = tf.reduce_sum(tf.multiply(q, a), 1)
        score = tf.div(pooled_mul_12, tf.multiply(pooled_len_1, pooled_len_2) + 1e-8, name="cosines")
        return score

if __name__ == '__main__':
    import numpy as np
    from argparse import Namespace
    with tf.Graph().as_default():
        sess = tf.Session()

        config = Namespace()
        config.sequence_length_q = 100
        config.sequence_length_a = 100
        config.vocab_size = 10
        config.batch_size = 4
        config.hidden_size = 100
        config.embedding_size = 100
        config.dropout_keep_prob = 0.5

        model = DecompAtt(config, is_training=True)
        sess.run(tf.global_variables_initializer())

        input_x_1 = np.random.randint(config.vocab_size, size=(config.batch_size, config.sequence_length_q))
        input_x_2 = np.random.randint(config.vocab_size, size=(config.batch_size, config.sequence_length_a))
        input_x_3 = np.random.randint(config.vocab_size, size=(config.batch_size, config.sequence_length_a))

        feed_dict = {
            model.input_x_1: input_x_1,
            model.input_x_2: input_x_2,
            model.input_x_3: input_x_3
        }
        pos, neg, score12, score13 = sess.run([model.positive, model.negative,
                                               model.score12, model.score13], feed_dict)
        print(pos, neg, score12, score13)
