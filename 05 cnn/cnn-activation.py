from collections import defaultdict
from collections import Counter
import numpy as np
import sys
import os
import random
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("input_dir", "input", "Input directory where training dataset and meta data are saved")
tf.app.flags.DEFINE_string("output_dir", "output", "Output directory where output such as logs are saved.")
tf.app.flags.DEFINE_string("model_dir", "output", "Model directory where final model files are saved.")

np.set_printoptions(linewidth=np.nan, threshold=np.nan)

# Functions to read in the corpus
w2i = defaultdict(lambda: len(w2i))
UNK = w2i["<unk>"]
def read_dataset(filename):
    with open(filename, "r") as f:
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            words = words.split(" ")
            yield (words, [w2i[x] for x in words], int(tag))

# Read in the data
train = list(read_dataset("classes/train.txt"))
w2i = defaultdict(lambda: UNK, w2i)
dev = list(read_dataset("classes/test.txt"))
nwords = len(w2i)
ntags = 5

EMB_SIZE = 64
WIN_SIZE = 3
FILTER_SIZE = 64


def create_model(input):
    w_emb = tf.Variable(tf.random_uniform([nwords, 1, 1, EMB_SIZE], -1.0, 1.0), dtype=tf.float32, name='w_emb')
    w_cnn = tf.Variable(tf.random_uniform([1, WIN_SIZE, EMB_SIZE, FILTER_SIZE], -1.0, 1.0), dtype=tf.float32, name='w_cnn')
    b_cnn = tf.Variable(tf.zeros([FILTER_SIZE]), dtype=tf.float32, name='b_cnn')
    w_sm = tf.Variable(tf.random_uniform([FILTER_SIZE, ntags], -1.0, 1.0), dtype=tf.float32, name='w_sm')
    b_sm = tf.Variable(tf.zeros([ntags]), dtype=tf.float32, name='b_sm')
    embed = tf.nn.embedding_lookup(w_emb, input, name='embed')
    cnn_in = tf.reshape(embed, [1, -1, 1, EMB_SIZE])
    cnn_out = tf.nn.conv2d(cnn_in, w_cnn, strides=[1, 1, 1, 1], padding='SAME')
    cnn_out = tf.nn.bias_add(cnn_out, b_cnn)
    pool_out = tf.reduce_max(cnn_out, 1)
    pool_out = tf.reshape(pool_out, [-1, FILTER_SIZE])
    relu = tf.nn.relu(pool_out)
    score = tf.matmul(relu, w_sm) + b_sm
    score = tf.reshape(score, [-1])
    return score


def create_loss(score, label):
    sparse = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=score)
    loss = tf.reduce_mean(sparse)
    return loss


def calc_predict_and_activations(scores):
    print('%d ||| %s' % (tag, ' '.join(words)))
    predict = np.argmax(scores)
    print('scores=%s, predict: %d' % (scores, predict))


def display_activations(words, activations):
    pad_begin = (WIN_SIZE - 1) / 2
    pad_end = WIN_SIZE - 1 - pad_begin
    words_padded = ['pad' for i in range(pad_begin)] + words + ['pad' for i in range(pad_end)]

    ngrams = []
    for act in activations:
        ngrams.append('[' + ', '.join(words_padded[act:act+WIN_SIZE]) + ']')

    return ngrams


def main(_):
    train, valid = read()
    input = tf.placeholder(tf.int32, shape=[None], name='input')
    label = tf.placeholder(tf.int32, name='tag')
    score = create_model(input)
    loss = create_loss(score, label)
    optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for iter in range(100):
            iteration_loss = 0
            for words, tag in train:
                if len(words) < WIN_SIZE:
                    words += [0] * (WIN_SIZE-len(words))
                feed = {
                    input: words,
                    label: tag
                }
                _, loss_val = sess.run([optimizer, loss], feed_dict=feed)
                iteration_loss += loss_val
            print('iteration: %r; loss: %.4f' % (iter, iteration_loss / len(train)))

        test_correct = 0.0
        for words, tag in dev:
            scores = calc_scores(words).npvalue()
            predict = np.argmax(scores)
            if predict == tag:
                test_correct += 1

if __name__ == "__main__":
    tf.app.run()
