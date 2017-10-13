import numpy as np
from collections import defaultdict
import random
import sys
import os
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

w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
UNK = w2i["<unk>"]
EMB_SIZE = 64
HID_SIZE = 64
HID_LAY = 2

def read_dataset(filename):
    global w2i, t2i
    with open(filename, "r") as f:
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            yield ([w2i[x] for x in words.split(" ")], t2i[tag])


def read():
    global w2i
    train_path = os.path.join(FLAGS.input_dir, "classes/train.txt")
    train = list(read_dataset(train_path))
    w2i = defaultdict(lambda: UNK, w2i)
    test_path = os.path.join(FLAGS.input_dir, "classes/test.txt")
    dev = list(read_dataset(test_path))
    return train, dev


def create_model(input):
    nwords = len(w2i)
    ntags = len(t2i)
    w_emb = tf.Variable(tf.random_uniform([nwords, EMB_SIZE], -1.0, 1.0), dtype=tf.float32, name='w_emb')
    w_h_list = [tf.Variable(tf.truncated_normal([EMB_SIZE if i == 0 else HID_SIZE, HID_SIZE], stddev=0.1), dtype=tf.float32) for i in range(HID_LAY)]
    b_h_list = [tf.Variable(tf.zeros(shape=[HID_SIZE]), dtype=tf.float32) for i in range(HID_LAY)]
    w_sm = tf.Variable(tf.truncated_normal([HID_SIZE, ntags], stddev=0.1), dtype=tf.float32, name='w_sm')
    b_sm = tf.Variable(tf.zeros([ntags]), dtype=tf.float32, name='b_sm')
    temp = tf.nn.embedding_lookup(w_emb, input)
    temp = tf.reduce_sum(temp, 0)
    temp = tf.expand_dims(temp, 0)
    for i in range(HID_LAY):
        temp = tf.tanh(tf.matmul(temp, w_h_list[i])) + b_h_list[i]
    temp = tf.tanh(tf.matmul(temp, w_sm)) + b_sm
    r = tf.reshape(temp, [-1])
    return r


def create_loss(logits, labels):
    sparse = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.reduce_mean(sparse)
    return loss


def evaluate(logits, input, text_list):
    i2t = dict((v, k) for k, v in t2i.items())
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for text in text_list:
            r = sess.run(logits, feed_dict= {input: [w2i[x] for x in text.split(' ')]})
            r = [x[0] for x in sorted(zip(r, range(len(r))), key=lambda x: i2t[x[1]])]
            print(text + ': ' + ','.join([str(x) for x in r]))


def main(_):
    train, dev = read()
    input = tf.placeholder(tf.int32, shape=[None], name='input')
    labels = tf.placeholder(tf.int32, name='labels')

    logits = create_model(input)
    loss = create_loss(logits, labels)
    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for iteration in range(100):
            random.shuffle(train)
            train_loss = 0.0
            for words, tag in train:
                feed_dict = {
                    input: words,
                    labels: tag
                }
                _, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)
                train_loss += loss_val
            print("iter %r: train loss/sent=%.4f" % (iteration, train_loss/len(train)))
    print('finished.')
    evaluate(logits, input, [
        'i love you',
        'i hate you',
        "i don't hate you",
        "there is nothing i don't love in you",
        'you are good at nothing'
    ])


if __name__ == "__main__":
    tf.app.run()
