import numpy as np
import sys
import os
import tensorflow as tf
import inspect
import time
from collections import defaultdict

###################################################################
# Variables                                                       #
# When launching project or scripts from Visual Studio,           #
# input_dir and output_dir are passed as arguments automatically. #
# Users could set them from the project setting page.             #
###################################################################

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input_dir', 'input', 'Input directory where training dataset and meta data are saved')
tf.app.flags.DEFINE_string('output_dir', 'output', 'Output directory where output such as logs are saved.')

# We reuse the data reading from the language modeling class
w2i = defaultdict(lambda: len(w2i))
S = w2i['<s>']
UNK = w2i['<unk>']
i2w = {}
valid_samples = []
word_counts = defaultdict(int)

def read_dataset(filename):
    global w2i
    with open(filename, 'r') as f:
        lines = []
        for line in f:
            line = line.strip().split(' ')
            for word in line:
                word_counts[w2i[word]] += 1
            lines.extend(line)
        return [w2i[x] for x in lines]


def read():
    global w2i, i2w
    train_path = os.path.join(FLAGS.input_dir, 'ptb/train.txt')
    train = read_dataset(train_path)
    w2i = defaultdict(lambda: UNK, w2i)
    valid_path = os.path.join(FLAGS.input_dir, 'ptb/valid.txt')
    valid = read_dataset(valid_path)
    test_path = os.path.join(FLAGS.input_dir, 'ptb/test.txt')
    test = read_dataset(test_path)
    i2w = {v: k for k, v in w2i.items()}
    return train, valid, test


EMBED_SIZE = 64
NUMBER_LAYERS = 2
KEEP_PROB = 0.5
BATCH_SIZE = 20
NUMBER_STEPS = 35
HIDDEN_SIZE = 1000
INIT_SCALE = 0.05
MAX_GRAD_NORM = 5
MAX_EPOCH = 6
MAX_MAX_EPOCH = 39
LR_DECAY = 0.8
LEARNING_RATE = 1.0


class PTBInput(object):
    def __init__(self, data, name):
        self._epoch_size = ((len(data) // BATCH_SIZE) - 1) // NUMBER_STEPS
        self._input, self._label = self.ptb_producer(data, BATCH_SIZE, NUMBER_STEPS, name)

    def ptb_producer(self, raw_data, batch_size, num_steps, name):
        raw_data = tf.convert_to_tensor(raw_data, dtype=tf.int32, name=name)
        data_len = tf.size(raw_data)
        batch_len = data_len // batch_size
        data = tf.reshape(raw_data[0 : batch_size * batch_len], [batch_size, batch_len])
        epoch_size = (batch_len - 1) // num_steps
        epoch_size = tf.identity(epoch_size)

        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.strided_slice(data, [0, i * num_steps], [batch_size, (i + 1) * num_steps])
        x.set_shape([batch_size, num_steps])

        y = tf.strided_slice(data, [0, i * num_steps + 1], [batch_size, (i + 1) * num_steps + 1])
        y.set_shape([batch_size, num_steps])
        return x, y

    def produce(self):
        return self._input, self._label, self._epoch_size


class PTBModel(object):
    def __init__(self, input, label, is_training):
        self._input = input
        batch_size = input.shape[0].value
        if is_training:
            attn_cell = self.create_attn_cell
        else:
            attn_cell = self.create_lstm_cell
        cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(NUMBER_LAYERS)], state_is_tuple=True)
        self._state = cell.zero_state(batch_size, tf.float32)

        nwords = len(w2i)
        embedding = tf.get_variable('embedding', [nwords, HIDDEN_SIZE], dtype=tf.float32)
        inputs = tf.nn.embedding_lookup(embedding, input)
        outputs = []
        state = self._state
        with tf.variable_scope('RNN'):
            for time_step in range(NUMBER_STEPS):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, HIDDEN_SIZE])

        softmax_w = tf.get_variable('softmax_w', [HIDDEN_SIZE, nwords], dtype=tf.float32)
        softmax_b = tf.get_variable('softmax_b', [nwords], dtype=tf.float32)
        logits = tf.add(tf.matmul(output, softmax_w), softmax_b, name='logits')
        self._logits = logits

        if label is None:
            return

        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(label, [-1])],
            [tf.ones([batch_size * NUMBER_STEPS], dtype=tf.float32)])

        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._state = state

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), MAX_GRAD_NORM)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())
        self._new_lr = tf.placeholder(tf.float32, shape=[], name='new_learning_rate')
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def create_lstm_cell(self):
        if 'reuse' in inspect.getargspec(tf.contrib.rnn.BasicLSTMCell.__init__).args:
            return tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE, forget_bias=0.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
        else:
            return tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE, forget_bias=0.0, state_is_tuple=True)

    def create_attn_cell(self):
        return tf.contrib.rnn.DropoutWrapper(self.create_lstm_cell(), output_keep_prob = KEEP_PROB)

    @property
    def input(self):
        return self._input

    @property
    def state(self):
        return self._state

    @property
    def cost(self):
        return self._cost

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

    @property
    def logits(self):
        return self._logits


def run_epoch(session, model, epoch_size, eval_op=None, verbose=True):
    '''Runs the model on the given data.'''
    start_time = time.time()
    costs = 0.0
    iters = 0

    fetches = {
        'cost': model.cost
    }
    if eval_op is not None:
        fetches['eval_op'] = eval_op

    for step in range(epoch_size):
        vals = session.run(fetches)
        cost = vals['cost']

        costs += cost
        iters += NUMBER_STEPS

        if verbose and step % 10 == 0:
            print('--%.3f perplexity: %.3f speed: %.0f wps' % (step / epoch_size, np.exp(costs / iters), iters * BATCH_SIZE / (time.time() - start_time)))

    return np.exp(costs / iters)


def main(_):
    initializer = tf.random_uniform_initializer(-INIT_SCALE, INIT_SCALE)
    train, valid, test = read()

    with tf.Graph().as_default():

        with tf.name_scope('Train'):
            train_input, train_label, train_epoch_size = PTBInput(data=train, name='TrainInput').produce()
            with tf.variable_scope('Model', reuse=None, initializer=initializer):
                train_model = PTBModel(is_training=True, input=train_input, label=train_label)

        with tf.name_scope('Valid'):
            valid_input, valid_label, valid_epoch_size = PTBInput(data=valid, name='ValidInput').produce()
            with tf.variable_scope('Model', reuse=True, initializer=initializer):
                valid_model = PTBModel(is_training=False, input=valid_input, label=valid_label)

        with tf.name_scope('Test'):
            test_input, test_label, test_epoch_size = PTBInput(data=test, name='TestInput').produce()
            with tf.variable_scope('Model', reuse=True, initializer=initializer):
                valid_model = PTBModel(is_training=False, input=test_input, label=test_label)

        sv = tf.train.Supervisor(logdir=FLAGS.output_dir)
        with sv.managed_session() as session:
            for i in range(MAX_MAX_EPOCH):
                lr_decay = LR_DECAY ** max(i + 1 - MAX_EPOCH, 0.0)
                train_model.assign_lr(session, LEARNING_RATE * lr_decay)
                print('Epoch: %d Learning rate: %.3f' % (i + 1, session.run(train_model.lr)))

                train_perplexity = run_epoch(session, train_model, train_epoch_size, eval_op=train_model.train_op, verbose=True)
                print('Epoch: %d Train Perplexity: %.3f' % (i + 1, train_perplexity))

                valid_perplexity = run_epoch(session, valid_model, valid_epoch_size)
                print('Epoch: %d Valid Perplexity: %.3f' % (i + 1, valid_perplexity))
    
                sv.saver.save(session, FLAGS.output_dir, global_step=sv.global_step)

            test_perplexity = run_epoch(session, test_model, test_epoch_size)
            print('Test Perplexity: %.3f' % test_perplexity)



if __name__ == '__main__':
    tf.app.run()
