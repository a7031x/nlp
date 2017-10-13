from collections import defaultdict
import numpy as np
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

N=2 #length of window on each side (so N=2 gives a total window size of 5, as in t-2 t-1 t t+1 t+2)
EMB_SIZE = 128 # The size of the embedding

# We reuse the data reading from the language modeling class
w2i = defaultdict(lambda: len(w2i))
S = w2i["<s>"]
UNK = w2i["<unk>"]
i2w = {}

def read_dataset(filename):
    global w2i
    with open(filename, 'r') as f:
        for line in f:
            yield [w2i[x] for x in line.strip().split(' ')]


def read():
    global w2i, i2w
    train_path = os.path.join(FLAGS.input_dir, 'train.txt')
    train = list(read_dataset(train_path))
    w2i = defaultdict(lambda: UNK, w2i)
    valid_path = os.path.join(FLAGS.input_dir, 'valid.txt')
    valid = list(read_dataset(valid_path))
    i2w = {v: k for k, v in w2i.items()}
    return train, valid


def create_model(input):
    nwords = len(w2i)
    w_emb = tf.Variable(tf.random_uniform([nwords, EMB_SIZE], -1.0, 1.0), dtype=tf.float32, name='w_emb')
    w_w = tf.Variable(tf.truncated_normal([EMB_SIZE, nwords], stddev=0.1), dtype=tf.float32, name='w_w')
    lookup = tf.nn.embedding_lookup(w_emb, input, name='lookup')
    output = tf.matmul(lookup, w_w)
    logits = tf.nn.log_softmax(output)
    return logits


def create_loss(logits, input):
    #padding = tf.fill([N], S)
    #padding_input = tf.concat(0, [padding, input, padding])
    #return sparse
    #loss = tf.nn.embedding_lookup(logits, input)
    elements = tf.map_fn(lambda i, x: logits[i, x], input)
    return tf.reduce_sum(elements)


def main(_):
    train, valid = read()
    labels_location = os.path.join(FLAGS.output_dir, 'labels.txt')
    with open(labels_location, 'w') as labels_file:
      for i in range(len(i2w)):
        labels_file.write(i2w[i] + '\n')

    input = tf.placeholder(tf.int32, shape=[None], name='input')
    logits = create_model(input)
    loss = create_loss(logits, input)
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
    exit(0)


if __name__ == "__main__":
    tf.app.run()
