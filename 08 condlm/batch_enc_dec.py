import numpy as np
import sys
import os
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
tf.app.flags.DEFINE_string("model_dir", "output", "Model directory where final model files are saved.")

#some of this code borrowed from Qinlan Shen's attention from the MT class last year
#much of the beginning is the same as the text retrieval
# format of files: each line is "word1 word2 ..." aligned line-by-line
train_src_file = "parallel/train.ja"
train_trg_file = "parallel/train.en"
dev_src_file = "parallel/dev.ja"
dev_trg_file = "parallel/dev.en"

w2i_src = defaultdict(lambda: len(w2i_src))
w2i_trg = defaultdict(lambda: len(w2i_trg))


# Creates batches where all source sentences are the same length
def create_batches(sorted_dataset, max_batch_size):
    source = [x[0] for x in sorted_dataset]
    src_lengths = [len(x) for x in source]
    batches = []
    prev = src_lengths[0]
    prev_start = 0
    batch_size = 1
    for i in range(1, len(src_lengths)):
        if src_lengths[i] != prev or batch_size == max_batch_size:
            batches.append((prev_start, batch_size))
            prev = src_lengths[i]
            prev_start = i
            batch_size = 1
        else:
            batch_size += 1
    return batches


def read(fname_src, fname_trg):
    """
    Read parallel files where each line lines up
    """
    fname_src = os.path.join(FLAGS.input_dir, fname_src)
    fname_trg = os.path.join(FLAGS.input_dir, fname_trg)
    with open(fname_src, "r", encoding='utf8') as f_src, open(fname_trg, "r", encoding='utf8') as f_trg:
        for line_src, line_trg in zip(f_src, f_trg):
            #need to append EOS tags to at least the target sentence
            sent_src = [w2i_src[x] for x in line_src.strip().split() + ['</s>']] 
            sent_trg = [w2i_trg[x] for x in ['<s>'] + line_trg.strip().split() + ['</s>']] 
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

EMBED_SIZE = 64
HIDDEN_SIZE = 128
BATCH_SIZE = 16
TIMESTEPS = 20
MAX_TIMESTEPS = 96


def create_encoder(input, length):
    with tf.name_scope('encoder'):
        embedding = tf.Variable(tf.random_uniform([nwords_src, EMBED_SIZE], -1.0, 1.0), dtype=tf.float32)
        input = tf.nn.embedding_lookup(embedding, input)
        lstm_cell = rnn.BasicLSTMCell(HIDDEN_SIZE, forget_bias=1.0)
        outputs, states = tf.nn.dynamic_rnn(lstm_cell, input, dtype=tf.float32, sequence_length=length, scope='source')
        last_indices = tf.subtract(length, tf.ones([BATCH_SIZE], tf.int32));
        batch_ids = tf.convert_to_tensor(list(range(BATCH_SIZE)))
        stacked_indices = tf.stack([batch_ids, last_indices])
        stacked_indices = tf.transpose(stacked_indices)
        outputs = tf.gather_nd(outputs, stacked_indices)
        return outputs, states


def create_decoder(src_output, input, length):
    with tf.name_scope('decoder'):
        lstm_cell = rnn.BasicLSTMCell(HIDDEN_SIZE, forget_bias=1.0)
        state = rnn.LSTMStateTuple(src_output, tf.tanh(src_output))
        embedding = tf.Variable(tf.random_uniform([nwords_trg, EMBED_SIZE], -1.0, 1.0), dtype=tf.float32)
        prefix = tf.convert_to_tensor([[eos_trg]] * BATCH_SIZE)
        prefix_input = tf.concat([prefix, input], axis=1)
        prefix_input = tf.nn.embedding_lookup(embedding, prefix_input)
        outputs, states = tf.nn.dynamic_rnn(lstm_cell, prefix_input, initial_state=state, dtype=tf.float32, sequence_length=length, scope='target')
        weights = tf.Variable(tf.random_uniform([HIDDEN_SIZE, nwords_trg], -1, 1))
        bias = tf.Variable(tf.zeros([nwords_trg]))
        unstacked_outputs = tf.unstack(outputs, axis=0)
        unstacked_outputs = [tf.matmul(x, weights) + bias for x in unstacked_outputs]
        outputs = tf.stack(unstacked_outputs)
        return outputs, states


def create_loss(outputs, labels, lengths):
#outputs 16xNonex7083
#labels 16xNone
#lengths 16
    sparse = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs, labels=labels)
#sparse 16xNone
    loss = tf.reduce_sum(sparse, axis=[0, 1])
    return loss


def create_optimizer(loss):
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5)
    optimizer = tf.train.GradientDescentOptimizer(1.0)
    train_optimizer = optimizer.apply_gradients(zip(grads, tvars), global_step=tf.contrib.framework.get_or_create_global_step())
    return train_optimizer


def main(_):
    with tf.Graph().as_default():
        input_src = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None], name='input_src')
        input_trg = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None], name='input_trg')
        length_src = tf.placeholder(tf.int32, shape=[BATCH_SIZE], name='length_src')
        length_trg = tf.placeholder(tf.int32, shape=[BATCH_SIZE], name='length_trg')

        src_output, _ = create_encoder(input_src, length_src)
        trg_output, _ = create_decoder(src_output, input_trg, length_trg)
        trg_loss = create_loss(trg_output, input_trg, length_trg)
        optimizer = create_optimizer(trg_loss)

        sv = tf.train.Supervisor(logdir=FLAGS.output_dir)
    #    with sv.managed_session() as sess:
    #        for iter in range(100):
                
#need to evaluate trg_output to check the second axis

if __name__ == "__main__":
    tf.app.run()
