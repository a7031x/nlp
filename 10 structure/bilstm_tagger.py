import numpy as np
import sys
import os
import tensorflow as tf
import random
from collections import defaultdict

###################################################################
# Variables                                                       #
# When launching project or scripts from Visual Studio,           #
# input_dir and output_dir are passed as arguments automatically. #
# Users could set them from the project setting page.             #
###################################################################

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("input_dir", "input", "Input directory where training dataset and meta data are saved")
tf.app.flags.DEFINE_string("output_dir", "output", "Output directory where output such as logs are saved.")
tf.app.flags.DEFINE_string("model_dir", "model", "Model directory where final model files are saved.")

use_teacher_forcing = True
use_structure_perceptron = True
use_cost_augmented = True
use_hinge = True
use_schedule = True

# format of files: each line is "word1|tag1 word2|tag2 ..."
train_file = "tags/train.txt"
dev_file = "tags/dev.txt"

w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))

def read(fname):
    """
    Read tagged file
    """
    with open(os.path.join(FLAGS.input_dir, fname), 'r', encoding='utf8') as f:
        for line in f:
            words, tags = [], []
            for wt in line.strip().split():
                w, t = wt.split('|')
                words.append(w2i[w])
                tags.append(t2i[t])
            yield (words, tags)


class AlwaysTrueSampler:
    """
    An always true sampler, only sample fromtrue distribution.
    """

    def sample_true(self):
        return True

    def decay(self):
        pass


class ScheduleSampler:
    """
    A linear schedule sampler.
    """

    def __init__(self, start_rate=0.8, min_rate=0.2, decay_rate=0.1):
        self.min_rate = min_rate
        self.iter = 0
        self.decay_rate = decay_rate
        self.start_rate = start_rate
        self.reach_min = False
        self.sample_rate = start_rate

    def decay_func(self):
        if not self.reach_min:
            self.sample_rate = self.start_rate - self.iter * self.decay_rate
            if self.sample_rate < self.min_rate:
                self.reach_min = True
                self.sample_rate = self.min_rate

    def decay(self):
        self.iter += 1
        self.decay_func()
        print("Sample rate is now %.2f" % self.sample_rate)

    def sample_true(self):
        return random.random() < self.sample_rate

    @property
    def rate(self):
        return self.sample_rate


if use_schedule:
    sampler = ScheduleSampler()
else:
    sampler = AlwaysTrueSampler()

# Read the data
train = list(read(train_file))
unk_word = w2i["<unk>"]
w2i = defaultdict(lambda: unk_word, w2i)
unk_tag = t2i["<unk>"]
start_tag = t2i["<start>"]
t2i = defaultdict(lambda: unk_tag, t2i)
nwords = len(w2i)
ntags = len(t2i)
dev = list(read(dev_file))

# Model parameters
EMBED_SIZE = 64
TAG_EMBED_SIZE = 16
HIDDEN_SIZE = 128
MAX_LENGTH = 128


def calc_score(fw_cell, bw_cell, states, x, y, t, weight, bias, tag_embedding, sampler):
    t = tf.cond(tf.equal(t, -1), lambda: states[2], lambda: t if sampler.sample_true() else states[2])
    t = tf.nn.embedding_lookup(tag_embedding, t)
    x = tf.concat([x, t], axis=0)
    x = tf.reshape(x, [1, -1])
    fw_o, states[0] = fw_cell(x, states[0], scope='fw_cell')
    bw_o, states[1] = bw_cell(x, states[1], scope='bw_cell')
    output = tf.concat([fw_o, bw_o], axis=1)
    score = tf.matmul(output, weight) + bias
    score = tf.reshape(score, [-1])
    prediction = tf.argmax(score, axis=1)
    states[2] = prediction
    return score


def create_model(input, label):
    if label is not None:
        tag = tf.concat([[start_tag], label[:-1]], axis=0)
    else:
        tag = -tf.ones_like(input, dtype=tf.int32)

    embedding = tf.get_variable('embedding', shape=[nwords, EMBED_SIZE], dtype=tf.float32)
    input = tf.nn.embedding_lookup(embedding, input)
    tag_embedding = tf.get_variable('tag_embedding', shape=[ntags, TAG_EMBED_SIZE], dtype=tf.float32)

    fwd_lstm = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE / 2, forget_bias=0.0)
    bwd_lstm = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE / 2, forget_bias=0.0)

    weight = tf.get_variable(name='weight', shape=[HIDDEN_SIZE, ntags], dtype=tf.float32)
    bias = tf.get_variable(name='bias', shape=[ntags], dtype=tf.float32)
    reverse_input = tf.reverse(input, [0])

    states = [fwd_lstm.zero_state(1, tf.float32), bwd_lstm.zero_state(1, tf.float32), start_tag]
    fn = lambda x: calc_score(fwd_lstm, bwd_lstm, states, x[0], x[1], x[2], weight, bias, tag_embedding, sampler)
    scores = tf.map_fn(fn, (input, reverse_input, tag), dtype=tf.float32)
    return scores


def create_mle_loss(scores, tags):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=tags)
    return tf.reduce_sum(loss)


def hamming_cost(predictions, reference):
    diff = predictions - reference
    return tf.count_nonzero(diff, dtype=tf.float32)


def calc_sequence_score(scores, tags):
    ones = tf.one_hot(tags, int(scores.shape[1]), axis=1, dtype=tf.float32)
    scores = scores * ones
    return tf.reduce_sum(scores)


def hamming_augmented_decode(scores, reference):
    cost = tf.one_hot(reference, depth=int(scores.shape[1]), on_value=0.0, off_value=1.0, dtype=tf.float32)
    scores += cost
    result = tf.argmax(scores, axis=1, output_type=tf.int32)
    return result


def perceptron_loss(scores, reference):
    if use_cost_augmented:
        predictions = hamming_augmented_decode(scores, reference)
    else:
        predictions = tf.argmax(scores, axis=1)

    margin = -2

    reference_score = calc_sequence_score(scores, reference)
    prediction_score = calc_sequence_score(scores, predictions)
    if use_cost_augmented:
        # One could actually get the hamming augmented value during decoding, but we didn't do it here for
        # demonstration purpose.
        hamming = hamming_cost(predictions, reference)
        loss = prediction_score + hamming - reference_score
    else:
        loss = prediction_score - reference_score

    if use_hinge:
        loss = tf.nn.relu(loss - margin)

    return loss



def calc_loss(scores, tags):
    if use_structure_perceptron:
        return perceptron_loss(scores, tags)
    else:
        return mle(scores, tags)


def calc_correct(scores, tags):
    predicts = tf.argmax(scores, axis=1, output_type=tf.int32)
    correct = tf.reduce_sum(tf.cast(tf.equal(predicts, tags), tf.float32))
    return correct


def main(_):
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-1, 1)

        inputs = tf.placeholder(tf.int32, [None])
        tags = tf.placeholder(tf.int32, [None])

        with tf.name_scope('Train'):
            with tf.variable_scope('Model', reuse=None, initializer=initializer):
                train_scores = create_model(inputs, tags)
                train_losses = calc_loss(train_scores, tags)
                train_corrects = calc_correct(train_scores, tags)
                optimizer = tf.train.AdamOptimizer().minimize(train_losses)

        with tf.name_scope('Valid'):
            with tf.variable_scope('Model', reuse=True, initializer=initializer):
                valid_scores = create_model(inputs, None)
                valid_losses = calc_loss(valid_scores, tags)
                valid_corrects = calc_correct(valid_scores, tags)

        sv = tf.train.Supervisor(logdir=FLAGS.output_dir)
        with sv.managed_session() as sess:
            for itr in range(100):
                print('training {}...'.format(sampler.rate))
                random.shuffle(train)
                counter = 0
                for words, labels in train:
                    feed = {
                        inputs: words,
                        tags: labels
                    }
                    loss_val, correct_val, _ = sess.run([train_losses, train_corrects, optimizer], feed_dict=feed)
                    counter += 1
                    if counter % 100 == 0:
                        print('progress: {}/{}, loss: {}, correct: {}/{}'.format(counter, len(train), loss_val, int(correct_val), len(words)))
                    if counter % 1000 == 0:
                        sv.saver.save(sess, FLAGS.output_dir, global_step=itr*10000+counter)

                sampler.decay()

                print('evaluating...')
                counter = 0
                for words, labels in dev:
                    feed = {
                        inputs: words,
                        tags: labels
                    }
                    loss_val, correct_val = sess.run([valid_losses, valid_corrects], feed_dict=feed)
                    counter += 1
                    if counter % 100 == 0:
                        print('loss: {}, correct: {}/{}'.format(loss_val, int(correct_val), len(words)))


if __name__ == "__main__":
    tf.app.run()
