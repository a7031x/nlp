import numpy as np
import sys
import os
import tensorflow as tf
from tensorflow.contrib import rnn
import random
from collections import defaultdict

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("input_dir", "input", "Input directory where training dataset and meta data are saved")
tf.app.flags.DEFINE_string("output_dir", "output", "Output directory where output such as logs are saved.")
tf.app.flags.DEFINE_string("model_dir", "output", "Model directory where final model files are saved.")

w2i_src = defaultdict(lambda: len(w2i_src))
w2i_trg = defaultdict(lambda: len(w2i_trg))


def read_dataset(dataset):
    global w2i_src, w2i_trg
    file_en = os.path.join(FLAGS.input_dir, 'parallel/' + dataset + '.en')
    file_ja = os.path.join(FLAGS.input_dir, 'parallel/' + dataset + '.ja')

    with open(file_ja, 'r', encoding='utf8') as f_src, open(file_en, 'r', encoding='utf8') as f_trg:
        for line_src, line_trg in zip(f_src, f_trg):
            sent_src = [w2i_src[x] for x in line_src.strip().split()]
            sent_trg = [w2i_trg[x] for x in line_trg.strip().split()]
            yield (sent_src, sent_trg)


def read():
    global w2i_src, w2i_trg
    train = list(read_dataset('train'))
    unk_src = w2i_src["<unk>"]
    w2i_src = defaultdict(lambda: unk_src, w2i_src)
    unk_trg = w2i_trg["<unk>"]
    w2i_trg = defaultdict(lambda: unk_trg, w2i_trg)
    nwords_src = len(w2i_src)
    nwords_trg = len(w2i_trg)
    dev = list(read_dataset('dev'))
    return train, dev


EMBED_SIZE = 64
HIDDEN_SIZE = 128
BATCH_SIZE = 16
TIMESTEPS = 20
MAX_TIMESTEPS = 96

def create_matrix(input, nwords, name):
    embedding = tf.Variable(tf.random_uniform([nwords, EMBED_SIZE], -1.0, 1.0), dtype=tf.float32, name=name)
    input = tf.nn.embedding_lookup(embedding, input)
    lstm_fw_cell = rnn.BasicLSTMCell(HIDDEN_SIZE, forget_bias=1.0)
    lstm_bw_cell = rnn.BasicLSTMCell(HIDDEN_SIZE, forget_bias=1.0)
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, input, dtype=tf.float32, scope=name)
    mat = outputs[0][:, -1, :]
    return mat


def create_model(input_src, input_trg):
    mat_src = create_matrix(input_src, len(w2i_src), 'source')
    mat_trg = create_matrix(input_trg, len(w2i_trg), 'target')
    sim_mat = tf.matmul(mat_src, mat_trg, transpose_b=True)
    return sim_mat


def create_loss(sim_mat):
    labels = tf.convert_to_tensor(list(range(BATCH_SIZE)), dtype=tf.int32)
    sparse = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=sim_mat)
    loss = tf.reduce_sum(sparse)
    return loss


def fill_eos(sent, eos):
    ls = len(sent)
    if ls < MAX_TIMESTEPS:
        sent = [eos] * (MAX_TIMESTEPS - ls) + sent
    elif ls > MAX_TIMESTEPS:
        print('MAX_TIMESTEPS shall be at least {}'.format(ls))
        sys.exit()

    return sent


def run_epoch(sess, loss, data, input_src, input_trg, optimizer):
    if optimizer is not None:
        random.shuffle(data)
    this_loss = this_sents = 0
    total_loss = 0
    eos_src = w2i_src['。']
    eos_trg = w2i_src['.']

    for sid in range(0, len(data), BATCH_SIZE):
        if len(data) - sid < BATCH_SIZE:
            continue
        feed = {
            input_src: [fill_eos(data[x][0], eos_src) for x in range(sid, sid+BATCH_SIZE)],
            input_trg: [fill_eos(data[x][1], eos_trg) for x in range(sid, sid+BATCH_SIZE)]
        }
        if optimizer is None:
            loss_val = sess.run(loss, feed_dict=feed)
        else:
            loss_val, _ = sess.run([loss, optimizer], feed_dict=feed)

        total_loss += loss_val
        this_loss += loss_val
        this_sents += BATCH_SIZE
        if (sid // BATCH_SIZE + 1) % 20 == 0:
            print('loss/sent: %.7f' % (this_loss / this_sents))
            this_loss = this_sents = 0

    return total_loss


def main(_):
    train, dev = read()
    with tf.Graph().as_default():
        input_src = tf.placeholder(tf.int32, shape=[BATCH_SIZE, MAX_TIMESTEPS], name='input_src')
        input_trg = tf.placeholder(tf.int32, shape=[BATCH_SIZE, MAX_TIMESTEPS], name='input_trg')
        sim_mat = create_model(input_src, input_trg)
        loss = create_loss(sim_mat)
        optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
        sv = tf.train.Supervisor(logdir=FLAGS.output_dir)
        with sv.managed_session() as sess:
            for iter in range(100):
                print('training...')
                run_epoch(sess, loss, train, input_src, input_trg, optimizer)
                print('evaluating...')
                run_epoch(sess, loss, dev, input_src, input_trg, None)
                sv.saver.save(sess, FLAGS.output_dir, global_step=iter)

if __name__ == "__main__":
    tf.app.run()
