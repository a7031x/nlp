import numpy as np
import sys
import os
import random
from collections import defaultdict
from tensorflow.contrib import rnn
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
tf.app.flags.DEFINE_string('model_dir', 'model', 'Model directory where final model files are saved.')

#some of this code borrowed from Qinlan Shen's attention from the MT class last year
#much of the beginning is the same as the text retrieval
# format of files: each line is "word1 word2 ..." aligned line-by-line
train_src_file = "parallel/train.ja"
train_trg_file = "parallel/train.en"
dev_src_file = "parallel/dev.ja"
dev_trg_file = "parallel/dev.en"

w2i_src = defaultdict(lambda: len(w2i_src))
w2i_trg = defaultdict(lambda: len(w2i_trg))


def read(fname_src, fname_trg):
    """
    Read parallel files where each line lines up
    """
    fname_src = os.path.join(FLAGS.input_dir, fname_src)
    fname_trg = os.path.join(FLAGS.input_dir, fname_trg)
    with open(fname_src, "r", encoding='utf8') as f_src, open(fname_trg, "r", encoding='utf8') as f_trg:
        for line_src, line_trg in zip(f_src, f_trg):
            #need to append EOS tags to at least the target sentence
            sent_src = [w2i_src[x] for x in line_src.strip().split()] 
            sent_trg = [w2i_trg[x] for x in line_trg.strip().split()] 
            yield (sent_src, sent_trg)

# Read the data
train = list(read(train_src_file, train_trg_file))
unk_src = w2i_src["<unk>"]
eos_src = w2i_src['</s>']
w2i_src = defaultdict(lambda: unk_src, w2i_src)
unk_trg = w2i_trg["<unk>"]
eos_trg = w2i_trg['</s>']
sos_trg = w2i_trg['<s>']
w2i_trg = defaultdict(lambda: unk_trg, w2i_trg)
i2w_trg = {v: k for k, v in w2i_trg.items()}

nwords_src = len(w2i_src)
nwords_trg = len(w2i_trg)
dev = list(read(dev_src_file, dev_trg_file))

EMBED_SIZE = 128
HIDDEN_SIZE = 256
BATCH_SIZE = 64
TIMESTEPS = 20
MAX_TIMESTEPS = 96
KEEP_PROB = 0.75


def create_lstm_cell():
    return rnn.BasicLSTMCell(HIDDEN_SIZE, forget_bias=0.0, state_is_tuple=True)


def create_attn_cell():
    return rnn.DropoutWrapper(create_lstm_cell(), output_keep_prob=KEEP_PROB)


def create_encoder(input, length, is_training):
    with tf.name_scope('encoder'):
        embedding = tf.Variable(tf.random_uniform([nwords_src, EMBED_SIZE], -1.0, 1.0), dtype=tf.float32)
        input = tf.nn.embedding_lookup(embedding, input)
        lstm_cell = create_attn_cell() if is_training else create_lstm_cell()
        outputs, states = tf.nn.dynamic_rnn(lstm_cell, input, dtype=tf.float32, sequence_length=length, scope='source')
        batch_size = int(input.shape[0])
        last_indices = tf.subtract(length, tf.ones([batch_size], tf.int32));
        batch_ids = tf.convert_to_tensor(list(range(batch_size)))
        stacked_indices = tf.stack([batch_ids, last_indices])
        stacked_indices = tf.transpose(stacked_indices)
        outputs = tf.gather_nd(outputs, stacked_indices)
        return outputs, states


def create_decoder(src_output, input, length, is_training):
    with tf.name_scope('decoder'):
        lstm_cell = create_attn_cell() if is_training else create_lstm_cell()
        state = rnn.LSTMStateTuple(src_output, tf.tanh(src_output))
        embedding = tf.Variable(tf.random_uniform([nwords_trg, EMBED_SIZE], -1.0, 1.0), dtype=tf.float32)
        batch_size = int(input.shape[0])
        prefix = tf.convert_to_tensor([[eos_trg]] * batch_size)
        prefix_input = tf.concat([prefix, input], axis=1)
        prefix_input = prefix_input[:, :-1]
        prefix_input = tf.nn.embedding_lookup(embedding, prefix_input)
        outputs, states = tf.nn.dynamic_rnn(lstm_cell, prefix_input, initial_state=state, dtype=tf.float32, sequence_length=length, scope='target')
        weights = tf.Variable(tf.random_uniform([HIDDEN_SIZE, nwords_trg], -1, 1))
        bias = tf.Variable(tf.zeros([nwords_trg]))
        unstacked_outputs = tf.unstack(outputs, axis=0)
        unstacked_outputs = [tf.matmul(x, weights) + bias for x in unstacked_outputs]
        unstacked_lengths = tf.unstack(length)
        unstacked_outputs = [o[:l] for o, l in zip(unstacked_outputs, unstacked_lengths)]
        unstacked_labels = tf.unstack(input)
        unstacked_labels = [o[:l] for o, l in zip(unstacked_labels, unstacked_lengths)]
        return unstacked_outputs, unstacked_lengths, unstacked_labels


def create_loss(outputs, labels):
    with tf.name_scope('loss'):
        sparses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=o, labels=l) for o, l in zip(outputs, labels)]
        losses = [tf.reduce_sum(s) for s in sparses]
        loss = sum(losses)
        return loss


def create_optimizer(loss):
    with tf.name_scope('optimizer'):
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5)
        optimizer = tf.train.GradientDescentOptimizer(1.0)
        train_optimizer = optimizer.apply_gradients(zip(grads, tvars), global_step=tf.contrib.framework.get_or_create_global_step())
        return train_optimizer


def fill_eos(sent):
    ls = len(sent)
    if ls < MAX_TIMESTEPS:
        sent = sent + [0] * (MAX_TIMESTEPS - ls)
    elif ls > MAX_TIMESTEPS:
        print('MAX_TIMESTEPS shall be at least {}'.format(ls))
        sys.exit()

    return sent


def run_epoch(sess, loss, data, input_src, length_src, input_trg, length_trg, optimizer, unstacked_outputs):
    if optimizer is not None:
        random.shuffle(data)
    this_loss = this_sents = 0
    total_loss = 0

    for sid in range(0, len(data), BATCH_SIZE):
        if len(data) - sid < BATCH_SIZE:
            continue
        feed = {
            input_src: [fill_eos(data[x][0]) for x in range(sid, sid+BATCH_SIZE)],
            length_src: [len(data[x][0]) for x in range(sid, sid+BATCH_SIZE)],
            input_trg: [fill_eos(data[x][1]) for x in range(sid, sid+BATCH_SIZE)],
            length_trg: [len(data[x][1]) for x in range(sid, sid+BATCH_SIZE)]
        }
        if optimizer is None:
            loss_val = sess.run(loss, feed_dict=feed)
        else:
            loss_val, _, outputs_val = sess.run([loss, optimizer, unstacked_outputs], feed_dict=feed)
        wids_evl = [[np.argmax(x) for x in y] for y in outputs_val]
        wids_tag = [data[x][1] for x in range(sid, sid+BATCH_SIZE)]
        total_loss += loss_val
        this_loss += loss_val
        this_sents += BATCH_SIZE
        if (sid // BATCH_SIZE + 1) % 20 == 0:
            print([i2w_trg[x] for x in wids_evl[BATCH_SIZE // 2]])
            print([i2w_trg[x] for x in wids_tag[BATCH_SIZE // 2]])
            print('loss/sent: %.7f' % (this_loss / this_sents))
            this_loss = this_sents = 0

    return total_loss


def main(_):
    with tf.Graph().as_default():
        input_src = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None], name='input_src')
        input_trg = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None], name='input_trg')
        length_src = tf.placeholder(tf.int32, shape=[BATCH_SIZE], name='length_src')
        length_trg = tf.placeholder(tf.int32, shape=[BATCH_SIZE], name='length_trg')

        src_output, _ = create_encoder(input_src, length_src, True)
        trg_output, trg_lengths, trg_labels = create_decoder(src_output, input_trg, length_trg, True)
        trg_loss = create_loss(trg_output, trg_labels)
        optimizer = create_optimizer(trg_loss)

        sv = tf.train.Supervisor(logdir=FLAGS.output_dir)
        with sv.managed_session() as sess:
            for iter in range(100):
                print('training...')
                random.shuffle(train)
                run_epoch(sess, trg_loss, train, input_src, length_src, input_trg, length_trg, optimizer, trg_output)
                sv.saver.save(sess, FLAGS.output_dir, global_step=sv.global_step)
                
#need to evaluate trg_output to check the second axis

if __name__ == "__main__":
    tf.app.run()
