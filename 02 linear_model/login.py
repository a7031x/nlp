from collections import defaultdict
import numpy as np
import sys
import os
import random
import tensorflow as tf

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
N = 2
w2i = defaultdict(lambda: len(w2i))
i2w = {}
S = w2i['<S>']
UNK = w2i['<UNK>']

def read_dataset(filename):
    global w2i
    with open(filename, 'r') as f:
        for line in f:
            yield [w2i[x] for x in line.strip().split(' ')]


def read():
    global w2i, i2w
    train_path = os.path.join(FLAGS.input_dir, 'ptb/train.txt')
    train = list(read_dataset(train_path))
    w2i = defaultdict(lambda: UNK, w2i)
    valid_path = os.path.join(FLAGS.input_dir, 'ptb/valid.txt')
    valid = list(read_dataset(valid_path))
    i2w = {v: k for k, v in w2i.items()}
    return train, valid


def create_model(input):
    nwords = len(w2i)
    w_emb = [tf.Variable(tf.random_uniform([nwords, nwords], -1.0, 1.0), dtype=tf.float32, name='w_emb') for _ in range(N)]
    lookup = [tf.nn.embedding_lookup(w_emb[i], input[i], name='lookup') for i in range(N)]
    b_sm = tf.Variable(tf.zeros([nwords]), dtype=tf.float32, name='b_sm')
    logits = b_sm
    for x in lookup:
        logits += x
    return logits


def create_loss(logits, labels):
    sparse = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.reduce_mean(sparse)
    return loss
   

def create_input(data):
    random.shuffle(data)
    article = []
    for x in data:
        article += x + [S]
    hist = [[article[x + y] for x in range(N)] for y in range(len(article) - N)]
    next = [article[x + 2] for x in range(len(article) - 2)]
    return hist, next


def create_ptb(data):
    hist, next = create_input(data)
    input = tf.convert_to_tensor(hist, dtype=tf.int32)
    labels = tf.convert_to_tensor(labels, dtype = tf.int32)
    return input, labels


def main(_):
    train, valid = read()
    hist, next = create_input(train)
    input = tf.placeholder(tf.int32, shape=[2], name='input')
    labels = tf.placeholder(tf.int32, name='labels')
    logits = create_model(input)
    loss = create_loss(logits, labels)
    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for iteration in range(100):
            random.shuffle(train)
            train_loss = 0.0
            sent_id = 0
            for x, y in zip(hist, next):
                feed_dict = {
                    input: x,
                    labels: y
                }
                _, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)
                train_loss += loss_val
                sent_id += 1
                if sent_id % 5000 == 0:
                    print('--finished %r/%r sentences, loss = %.4f' % (sent_id, len(hist), loss_val))
            print("iter %r: train loss/sent=%.4f" % (iteration, train_loss/len(train)))
    print('finished.')

    exit(0)


if __name__ == "__main__":
    tf.app.run()
