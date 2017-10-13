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

# Define the model
EMB_SIZE = 64
WIN_SIZE = 3
FILTER_SIZE = 64

# Functions to read in the corpus
w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
UNK = w2i["<unk>"]
def read_dataset(filename):
    with open(filename, "r") as f:
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            yield ([w2i[x] for x in words.split(" ")], t2i[tag])


def read():
    global w2i, i2w
    train_path = os.path.join(FLAGS.input_dir, 'classes/train.txt')
    train = list(read_dataset(train_path))
    w2i = defaultdict(lambda: UNK, w2i)
    valid_path = os.path.join(FLAGS.input_dir, 'classes/test.txt')
    valid = list(read_dataset(valid_path))
    i2w = {v: k for k, v in w2i.items()}

    return train, valid


def create_model(input):
    ntags = len(t2i)
    nwords = len(w2i)
    w_emb = tf.Variable(tf.random_uniform([nwords, 1, 1, EMB_SIZE], -1.0, 1.0), dtype=tf.float32, name='w_emb')
    w_cnn = tf.Variable(tf.random_uniform([1, WIN_SIZE, EMB_SIZE, FILTER_SIZE], -1.0, 1.0), dtype=tf.float32, name='w_cnn')
    b_cnn = tf.Variable(tf.zeros([FILTER_SIZE]), dtype=tf.float32, name='b_cnn')
    w_sm = tf.Variable(tf.random_uniform([FILTER_SIZE, ntags], -1.0, 1.0), dtype=tf.float32, name='w_sm')
    b_sm = tf.Variable(tf.zeros([ntags]), dtype=tf.float32, name='b_sm')
    embed = tf.nn.embedding_lookup(w_emb, input, name='embed')


def main(_):
    train, valid = read()


if __name__ == "__main__":
    tf.app.run()
