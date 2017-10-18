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
tf.app.flags.DEFINE_string("model_dir", "output", "Model directory where final model files are saved.")

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

    def __init__(self, start_rate=1, min_rate=0.2, decay_rate=0.1):
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


def calc_score(fw_cell, bw_cell, states, bw_state, x, y, t, weight, bias, tag_embedding, sampler):
    if t is None:
        t = states[2]
    else:
        if sampler.sample_true() is not True:
            t = states[2]
            
    x = tf.concate([x, t])
    fw_o, states[0] = fw_cell(x, states[0])
    bw_o, states[1] = bw_cell(x, states[1])
    output = tf.concate([fw_o, bw_o], axis=1)
    score = tf.matmul(output, weight) + bias
    prediction = tf.argmax(score)
    states[2] = prediction
    return score


def create_model(input, label):
    embedding = tf.Variable(tf.random_normal([nwords, EMBED_SIZE]), dtype=tf.float32)
    input = tf.embedding_lookup(embedding, input)
    tag_embedding = tf.Variable(tf.random_normal([ntags, TAG_EMBED_SIZE]), dtype=tf.float32)
    if use_schedule:
        sampler = ScheduleSampler()
    else:
        sampler = AlwaysTrueSampler()

    fwd_lstm = rnn.BasicLSTMCell(HIDDEN_SIZE / 2, forget_bias=0.0)
    bwd_lstm = rnn.BasicLSTMCell(HIDDEN_SIZE / 2, forget_bias=0.0)

    weight = tf.Variable(tf.random_normal([HIDDEN_SIZE, ntags]), dtype=tf.float32)
    bias = tf.Variable(tf.random_normal([ntags]), dtype=tf.float32)
    reverse_input = tf.reverse(input, 0)

    if label is not None:
        tag = tf.concate([start_tag, label[:-1]], axis=0)
        tag = tf.embedding_lookup(tag_embedding, tag)
    else:
        tag = [None] * MAX_LENGTH
        tag[0] = start_tag

    states = [fwd_lstm.zero_state(1, tf.float32), bwd_lstm.zero_state(1, tf.float32), start_tag]
    fn = lambda x, y, t: calc_score(fwd_lstm, bwd_lstm, states, x, y, t, weight, bias, tag_embedding, sampler)
    scores = tf.map_fn(fn, zip(input, reverse_input, tag))
    return scores


def main(_):
    # TODO: add your code here
    with tf.Session() as sess:
        welcome = sess.run(tf.constant("Hello, TensorFlow!"))
        print(welcome)
    exit(0)


if __name__ == "__main__":
    tf.app.run()
